#!/usr/bin/env python3
"""
Train reflectance reconstruction head on OlmoEarth base model with W&B logging.

Proper training loop with:
- Epoch-based training with validation
- W&B integration for experiment tracking
- Checkpoint saving and resuming
- Gradient accumulation support
- Mixed precision training
- Learning rate scheduling
"""

import os
import sys
import json
import logging
import time
from pathlib import Path
from datetime import datetime
from typing import Optional, Tuple
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from torch.amp import autocast
from torch.cuda.amp import GradScaler
from upath import UPath
import wandb

from olmoearth_pretrain.data.dataset import OlmoEarthDataset, GetItemArgs
from olmoearth_pretrain.data.constants import Modality
from olmoearth_pretrain.model_loader import ModelID, load_model_from_id
from olmoearth_pretrain.train.masking import MaskedOlmoEarthSample, MaskValue

# Add repo to path
sys.path.insert(0, '/home/rob/repo/olmoearth_pretrain')




# Get W&B API Key from environment variable
WANDB_API_KEY = os.environ.get('WANDB_API_KEY')


# Configuration
@dataclass
class TrainingConfig:
    """Training configuration."""
    # Paths
    dataset_path: str = (
        "s3://cc-dataocean/scratch/20251114_olmo_example/"
        "h5py_data_w_missing_timesteps_zstd_3_128_x_4/"
        "cdl_gse_landsat_openstreetmap_raster_"
        "sentinel1_sentinel2_l2a_srtm_worldcereal_worldcover_worldpop_wri_canopy_height_map/"
        "1138828"
    )
    checkpoint_dir: str = "./checkpoints"
    
    # Model
    model_id: ModelID = ModelID.OLMOEARTH_V1_BASE
    freeze_encoder: bool = True
    
    # Training
    num_epochs: int = 50  # Extended for convergence study
    batch_size: int = 1
    gradient_accumulation_steps: int = 1
    learning_rate: float = 1e-4  # Slightly higher for faster learning
    max_grad_norm: float = 100.0  # Increased to allow larger gradient updates
    warmup_steps: int = 100
    
    # Data
    patch_size: int = 8  # Changed from 1 to 8 to reduce memory usage
    num_samples_per_epoch: int = 20  # Process this many samples per epoch
    train_val_split: float = 0.8
    max_total_samples: int = None  # Use all available samples (no limit)
    # Note: With 50 epochs × 20 samples/epoch = 1000 iterations
    # On 18,000 unique samples, each sample is seen only once (or a few times)
    max_dataset_files: int = None  # Scan all H5 files to discover all samples
    
    # Mixed precision and device
    use_mixed_precision: bool = False  # Disable for stability with large loss values
    device: str = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    # W&B
    wandb_project: str = "olmoearth-reconstruction"
    wandb_entity: str = None
    use_wandb: bool = True  # Enabled for monitoring convergence
    
    # Checkpointing
    save_interval: int = 1  # Save every N epochs
    checkpoint_path: Optional[str] = None  # Resume from checkpoint
    
    # Head architecture
    hidden_multiplier: float = 1.0  # Multiplier for hidden layer size (768 * hidden_multiplier)
    debug_logging: bool = False  # Enable verbose logging for debugging
class ReflectanceReconstructionHead(nn.Module):
    """Learnable head for spatial image reconstruction from OlmoEarth tokens.
    
    Takes flattened spatial-temporal tokens and reconstructs full spectral images
    with correct spatial patterns, not just mean values.
    """
    
    def __init__(self, embedding_dim: int = 768, num_bands: int = 12, hidden_multiplier: float = 1.0):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_bands = num_bands
        
        hidden_dim = int(embedding_dim * hidden_multiplier)
        
        # MLP head that outputs full spectral bands directly (not scaled)
        self.head = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_bands),
        )
        
        # Don't initialize with large values - let network learn from scratch with standard init
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Reconstruct reflectance from OlmoEarth tokens.
        
        Args:
            x: [B, H, W, T, C, D] or flattened tokens
            
        Returns:
            Reflectance: [B, H, W, T, C, num_bands] or same shape as input with num_bands
        """
        original_shape = x.shape[:-1]
        x_flat = x.reshape(-1, self.embedding_dim)
        reflectance = self.head(x_flat)
        reflectance = reflectance.reshape(*original_shape, self.num_bands)
        
        # Output raw reflectance values - no scaling
        return reflectance


class ReflectanceReconstructionTrainer:
    """Trainer for reflectance reconstruction."""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = torch.device(config.device)
        self.global_step = 0
        self.start_epoch = 0
        self.total_samples_processed = 0
        self.current_epoch = 0
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        # Suppress verbose logging from external libraries
        logging.getLogger('botocore').setLevel(logging.WARNING)
        logging.getLogger('aiobotocore').setLevel(logging.WARNING)
        logging.getLogger('fsspec').setLevel(logging.WARNING)
        logging.getLogger('urllib3').setLevel(logging.WARNING)
        logging.getLogger('asyncio').setLevel(logging.WARNING)
        
        # Create checkpoint directory
        Path(config.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        
        # Initialize W&B
        self.wandb_enabled = config.use_wandb
        if self.wandb_enabled:
            try:
                if WANDB_API_KEY:
                    wandb.login(key=WANDB_API_KEY, relogin=False)
                    self.logger.info(f"W&B login successful")
                else:
                    self.logger.warning("W&B API key not available, skipping W&B initialization")
                    self.wandb_enabled = False
                
                if self.wandb_enabled:
                    run = wandb.init(
                        project=config.wandb_project,
                        entity=config.wandb_entity,
                        config=config.__dict__,
                        save_code=True,
                    )
                    self.logger.info(f"✅ W&B initialized successfully: {wandb.run.name}")
                    self.logger.info(f"   Dashboard URL: {wandb.run.url}")
            except Exception as e:
                self.logger.error(f"❌ W&B initialization failed: {e}")
                import traceback
                self.logger.error(traceback.format_exc())
                self.logger.warning("Continuing training without W&B logging")
                self.wandb_enabled = False
        else:
            self.logger.info(f"W&B disabled (use_wandb={config.use_wandb})")
                
        # Load models
        self._load_models()
        
        # Setup training components
        self._setup_training()
        
        # Resume from checkpoint if specified
        if config.checkpoint_path:
            self._load_checkpoint(config.checkpoint_path)
    
    def _load_models(self):
        """Load base model and create reconstruction head."""
        self.logger.info("Loading base OlmoEarth model...")
        self.base_model = load_model_from_id(self.config.model_id)
        self.base_model = self.base_model.to(self.device)
        if self.config.freeze_encoder:
            for param in self.base_model.parameters():
                param.requires_grad = False
            self.logger.info("Base model encoder frozen")
        
        # Count parameters
        total_params = sum(p.numel() for p in self.base_model.parameters())
        trainable_params = sum(p.numel() for p in self.base_model.parameters() if p.requires_grad)
        self.logger.info(f"Base model: {total_params:,} total, {trainable_params:,} trainable")
        
        # Create reconstruction head with configurable hidden multiplier
        self.head = ReflectanceReconstructionHead(
            embedding_dim=768,
            num_bands=12,
            hidden_multiplier=self.config.hidden_multiplier
        ).to(self.device)
        head_params = sum(p.numel() for p in self.head.parameters() if p.requires_grad)
        self.logger.info(f"Reconstruction head: {head_params:,} trainable parameters (hidden_multiplier={self.config.hidden_multiplier})")
    
    def _setup_training(self):
        """Setup optimizer, scheduler, and loss function."""
        self.optimizer = torch.optim.AdamW(
            self.head.parameters(),
            lr=self.config.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=1e-3,
        )
        
        # Scheduler
        total_steps = self.config.num_epochs * self.config.num_samples_per_epoch // self.config.batch_size
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=total_steps // 2,
            T_mult=1,
            eta_min=1e-7,
        )
        
        # Loss function
        self.criterion = nn.MSELoss()
        
        # Gradient scaler for mixed precision
        self.scaler = GradScaler() if self.config.use_mixed_precision else None
        
        self.logger.info(f"Optimizer: AdamW (lr={self.config.learning_rate})")
        self.logger.info(f"Scheduler: CosineAnnealingWarmRestarts")
        self.logger.info(f"Loss: MSELoss")
    
    def _load_dataset(self):
        """Load OlmoEarth training dataset."""
        self.logger.info(f"Loading dataset from {self.config.dataset_path}")
        
        # Load all modalities - model handles None values for missing modalities
        training_modalities = [
            "cdl", "gse", "landsat", "latlon", "openstreetmap_raster",
            "sentinel1", "sentinel2_l2a", "srtm", "worldcereal", "worldcover",
            "era5_10", "worldpop", "wri_canopy_height_map"
        ]
        
        self.dataset = OlmoEarthDataset(
            h5py_dir=UPath(self.config.dataset_path),
            training_modalities=training_modalities,
            dtype=np.float32,
            normalize=False,
        )
        
        # Prepare dataset - scans S3 once to discover samples and cache metadata
        self.logger.info("Preparing dataset (discovering samples from S3, this may take a few minutes)...")
        start_time = time.time()
        self.dataset.prepare()
        elapsed = time.time() - start_time
        self.logger.info(f"✓ Dataset prepared with {len(self.dataset)} samples in {elapsed:.1f}s")
    
    def _get_sample_with_masking(self, sample_idx: int) -> Optional[Tuple]:
        """Get a sample with random S2 timesteps masked for reconstruction.
        
        We randomly mask 1-12 timesteps of Sentinel-2 and use unmasked timesteps + other modalities
        to reconstruct the masked ones. This trains the model to fill in missing S2 data.
        """
        load_start = time.time()
        _, sample = self.dataset[GetItemArgs(idx=sample_idx, patch_size=self.config.patch_size, sampled_hw_p=128)]
        load_time = time.time() - load_start
        
        # Check that sample has S2 data
        if not hasattr(sample, 'sentinel2_l2a') or sample.sentinel2_l2a is None:
            return None
        
        # Get S2 data before masking (for ground truth of masked timesteps)
        s2_data = sample.sentinel2_l2a  # (H, W, T, C_s2) - numpy array
        num_timesteps = s2_data.shape[2]  # T dimension
        
        # Debug: Check raw S2 data (only if debug_logging enabled)
        if self.config.debug_logging:
            self.logger.info(f"  Raw S2 shape: {s2_data.shape}, dtype: {s2_data.dtype}")
            for t in range(num_timesteps):
                t_data = s2_data[:, :, t, :]
                self.logger.info(f"    Timestep {t}: min={np.min(t_data):.6f}, max={np.max(t_data):.6f}, std={np.std(t_data):.6f}, mean={np.mean(t_data):.6f}")
        
        if num_timesteps < 2:
            return None
        
        # Randomly choose 1-min(12, T-1) timesteps to mask
        # Keep at least 1 timestep unmasked for context
        # IMPORTANT: Only mask timesteps that have real data (non-zero or non-uniform)
        num_to_mask = np.random.randint(1, min(13, num_timesteps))
        
        # Find timesteps with actual data (not all zeros or uniform)
        valid_timesteps = []
        for t in range(num_timesteps):
            timestep_data = s2_data[:, :, t, :]
            # Check if timestep has variation and is not all zeros
            if np.std(timestep_data) > 1e-6 and np.max(timestep_data) > 0:
                valid_timesteps.append(t)
        
        # If we don't have enough valid timesteps, skip this sample
        if len(valid_timesteps) < 2:
            self.logger.warning(f"  Skipping sample - not enough valid timesteps. Available: {len(valid_timesteps)}")
            return None
        
        # Mask up to min(12, num_valid_timesteps-1) timesteps to keep at least 1 unmasked for context
        num_to_mask = min(np.random.randint(1, min(13, num_timesteps)), len(valid_timesteps) - 1)
        masked_indices = np.random.choice(valid_timesteps, size=num_to_mask, replace=False)
        
        # Extract GT for masked timesteps
        s2_gt_list = [s2_data[:, :, t, :].copy() for t in masked_indices]  # List of (H, W, C_s2)
        
        # Debug: Log GT extraction (only if debug_logging enabled)
        if self.config.debug_logging and len(s2_gt_list) > 0:
            first_gt = s2_gt_list[0]
            self.logger.info(f"  GT EXTRACTION DEBUG - shape: {first_gt.shape}, min: {np.min(first_gt):.6f}, max: {np.max(first_gt):.6f}, mean: {np.mean(first_gt):.6f}, std: {np.std(first_gt):.6f}, masked_indices: {masked_indices}")
        
        # Convert regular sample to MaskedOlmoEarthSample with all modalities
        masked_sample = MaskedOlmoEarthSample.from_olmoearthsample(sample)
        
        # Mask out the S2 data for the selected timesteps
        if masked_sample.sentinel2_l2a is not None:
            # Convert to numpy for manipulation
            s2_data = masked_sample.sentinel2_l2a.numpy() if isinstance(masked_sample.sentinel2_l2a, torch.Tensor) else masked_sample.sentinel2_l2a
            s2_mask = masked_sample.sentinel2_l2a_mask.numpy() if isinstance(masked_sample.sentinel2_l2a_mask, torch.Tensor) else masked_sample.sentinel2_l2a_mask
            
            if s2_mask is None:
                s2_mask = np.ones_like(s2_data)
            
            # Start with all timesteps visible to encoder
            s2_mask[:] = MaskValue.ONLINE_ENCODER.value
            
            # Mask out the selected timesteps - set to DECODER so encoder knows to predict these
            for t in masked_indices:
                s2_data[:, :, t, :] = 0  # Zero out the data values
                s2_mask[:, :, t, :] = MaskValue.DECODER.value  # Mark as decoder-only tokens
            
            masked_sample = masked_sample._replace(
                sentinel2_l2a=s2_data,
                sentinel2_l2a_mask=s2_mask
            )
        
        # Convert all numpy arrays to tensors on device - batch processing
        masked_sample_dict = {}
        for key, val in masked_sample._asdict().items():
            if val is None:
                masked_sample_dict[key] = None
            else:
                # Convert to tensor if not already
                if not isinstance(val, torch.Tensor):
                    val = torch.from_numpy(val)
                
                # Determine dtype
                if 'timestamp' in key:
                    tensor_val = val.long().to(self.device)
                elif 'latlon' in key or 'mask' in key or val.dtype in [torch.float16, torch.float32, torch.float64]:
                    tensor_val = val.float().to(self.device)
                else:
                    tensor_val = val.float().to(self.device)
                
                # Add batch dimension if needed
                if key == 'timestamps' and tensor_val.ndim == 2:
                    tensor_val = tensor_val.unsqueeze(0)
                elif key == 'latlon' and tensor_val.ndim == 1:
                    tensor_val = tensor_val.unsqueeze(0)
                elif 'mask' in key and tensor_val.ndim == 4:
                    tensor_val = tensor_val.unsqueeze(0)
                elif tensor_val.ndim == 4 and key != 'timestamps':
                    tensor_val = tensor_val.unsqueeze(0)
                
                masked_sample_dict[key] = tensor_val
        
        masked_sample = MaskedOlmoEarthSample(**masked_sample_dict)
        
        return (masked_sample, s2_gt_list, np.array(masked_indices), load_time, num_to_mask, num_timesteps)
    
    def train_epoch(self, sample_indices: list) -> dict:
        """Train for one epoch.
        
        Args:
            sample_indices: List of sample indices to use for this epoch
            
        Returns:
            Dictionary with epoch metrics
        """
        self.base_model.eval()
        self.head.train()
        
        epoch_metrics = {
            'loss': [],
            'learning_rate': [],
            'per_band_correlations': [[] for _ in range(12)],  # Track each band separately
        }
        
        processed = 0
        skipped = 0
        
        self.optimizer.zero_grad()
        accumulated_loss = 0.0
        batch_losses = []  # Track per-batch losses for progress
        batch_correlations = []  # Track per-batch correlations
        
        for batch_idx, sample_idx in enumerate(sample_indices):
            # Stop early if we've processed enough for this epoch
            if batch_idx >= self.config.num_samples_per_epoch:
                break
            sample_start = time.time()
            
            # Load sample with timing
            load_start = time.time()
            sample_data = self._get_sample_with_masking(sample_idx)
            load_time = time.time() - load_start
            if sample_data is None:
                skipped += 1
                continue
            
            masked_sample, s2_gt_list, masked_indices, load_time, num_to_mask, num_timesteps = sample_data
            
            # Log sample info (sparse logging - only every Nth sample)
            if batch_idx % 10 == 0:
                self.logger.info(f"  Sample {batch_idx+1}: load={load_time:.2f}s, masked={num_to_mask}/{num_timesteps} timesteps")
            
            # Forward through frozen base model
            model_start = time.time()
            with torch.no_grad():
                encoder_output, decoder_output, pooled_output, reconstructed, extra_metrics = self.base_model(
                    masked_sample,
                    self.config.patch_size
                )
            model_time = time.time() - model_start
            
            # encoder_output is a TokensAndMasks object with encoded modality tokens
            # S2 is encoded with ONLINE_ENCODER mask, so we get S2 tokens
            if encoder_output is None or not hasattr(encoder_output, 'sentinel2_l2a') or encoder_output.sentinel2_l2a is None:
                self.logger.info("    No S2 tokens from encoder, skipping")
                skipped += 1
                continue
            
            s2_tokens = encoder_output.sentinel2_l2a
            B, H, W, T, C_s2, D = s2_tokens.shape
            
            # Extract tokens from UNMASKED timesteps only
            # The encoder has already processed all timesteps and generated context-aware
            # representations. We only need the unmasked S2 tokens as our feature input.
            t_mask = torch.ones(T, dtype=torch.bool, device=self.device)
            t_mask[masked_indices] = False
            
            # Get input tokens from unmasked timesteps
            s2_tokens_unmasked = s2_tokens[:, :, :, t_mask, :, :]  # (B, H, W, T_unmasked, C_s2, D)
            input_tokens = s2_tokens_unmasked.reshape(-1, D)  # Flatten for head processing
            
            num_unmasked = (T - len(masked_indices))
            
            # Process through reconstruction head
            head_start = time.time()
            device_type = str(self.device).split(':')[0] if ':' in str(self.device) else str(self.device)
            with autocast(device_type=device_type, enabled=self.config.use_mixed_precision):
                predicted_reflectance_flat = self.head(input_tokens)  # (B*H*W*T_unmasked*C_s2, 12)
                
                # Reshape predictions back to per-timestep format
                # (B*H*W*T_unmasked*C_s2, 12) -> (B, H, W, T_unmasked, C_s2, 12) -> (B, H, W, T_unmasked, 12)
                pred_per_token = predicted_reflectance_flat.reshape(B, H, W, num_unmasked, C_s2, 12)
                pred_per_unmasked = pred_per_token.mean(dim=4)  # (B, H, W, T_unmasked, 12) - average over spectral tokens
                
                # Average across all unmasked timesteps to get a single estimate
                pred_estimate = pred_per_unmasked.mean(dim=3)  # (B, H, W, 12)
                
                # Compute loss against all masked timesteps
                losses_per_masked = []
                spatial_corr_losses = []
                correlations_all = []
                
                for gt_idx, (t_idx, s2_gt_t) in enumerate(zip(masked_indices, s2_gt_list)):
                    # Use single prediction (averaged across unmasked timesteps) for all masked timesteps
                    # The reconstruction head learns to predict spectral patterns from unmasked context
                    pred_t = pred_estimate  # (B, H, W, 12)
                    
                    # Pool GT from full resolution (128, 128) to token resolution (H, W)
                    gt_s2_raw = torch.from_numpy(s2_gt_t).to(self.device).float()  # (H_full, W_full, C_s2)
                    
                    # Pool GT from its native resolution to token resolution (H, W)
                    gt_s2_raw = torch.from_numpy(s2_gt_t).to(self.device).float()  # (H_full, W_full, C_s2)
                    
                    # Compute kernel size based on actual GT resolution
                    h_full, w_full = gt_s2_raw.shape[:2]
                    kernel_size_h = h_full // H if H > 0 else 1
                    kernel_size_w = w_full // W if W > 0 else 1
                    kernel_size_h = max(1, kernel_size_h)
                    kernel_size_w = max(1, kernel_size_w)
                    
                    # DEBUG first GT
                    if gt_idx == 0 and batch_idx < 1:
                        self.logger.info(f"    GT pooling debug:")
                        self.logger.info(f"      GT raw shape: {gt_s2_raw.shape}")
                        self.logger.info(f"      H (token resolution): {H}, W: {W}")
                        self.logger.info(f"      kernel_size: ({kernel_size_h}, {kernel_size_w})")
                    
                    # Pool each band separately
                    # Shape: (H_full, W_full, 12) -> pool spatial dims for each band
                    gt_pooled_list = []
                    for band_idx in range(12):
                        band_data = gt_s2_raw[:, :, band_idx]  # (H_full, W_full)
                        band_data_unsqueezed = band_data.unsqueeze(0).unsqueeze(0)  # (1, 1, H_full, W_full)
                        band_pooled = torch.nn.functional.avg_pool2d(band_data_unsqueezed, kernel_size=(kernel_size_h, kernel_size_w))
                        gt_pooled_list.append(band_pooled.squeeze(0).squeeze(0))  # (H, W)
                    
                    gt_pooled = torch.stack(gt_pooled_list, dim=-1)  # (H, W, 12)
                    
                    # DEBUG pooling result
                    if gt_idx == 0 and batch_idx < 1:
                        self.logger.info(f"      GT pooled shape: {gt_pooled.shape}")
                        self.logger.info(f"      GT pooled band 0 - min: {gt_pooled[:, :, 0].min():.2f}, max: {gt_pooled[:, :, 0].max():.2f}, mean: {gt_pooled[:, :, 0].mean():.2f}")
                    
                    # IMPORTANT: Keep all 12 spectral bands with their spatial structure
                    gt_masked_t = gt_pooled  # (H, W, 12) - all bands preserved
                    gt_masked_t = gt_masked_t.unsqueeze(0)  # (1, H, W, 12) - add batch
                    gt_masked_t = gt_masked_t.repeat(B, 1, 1, 1)  # (B, H, W, 12) - repeat for batch
                    
                    # Normalize GT - IMPORTANT: normalize per-band, ONLY across spatial dimensions (H, W)
                    # Don't include batch dimension in statistics
                    gt_masked_t_norm = gt_masked_t.clone()
                    for band in range(12):
                        for b in range(B):
                            band_vals = gt_masked_t[b, :, :, band]  # (H, W) - only spatial
                            band_mean = band_vals.mean()
                            band_std = band_vals.std()
                            if band_std > 1e-6:
                                gt_masked_t_norm[b, :, :, band] = (band_vals - band_mean) / band_std
                            else:
                                # If std is zero, the band is constant - can't learn anything
                                gt_masked_t_norm[b, :, :, band] = band_vals - band_mean
                                if gt_idx == 0 and batch_idx < 1:
                                    self.logger.warning(f"    Band {band} has zero variance! Raw band min/max: {band_vals.min()}/{band_vals.max()}")
                    
                    # Compute MSE for this timestep
                    # Normalize predictions per-band ONLY across spatial dimensions to match GT
                    pred_norm = pred_t.clone()
                    for band in range(12):
                        for b in range(B):
                            pred_band = pred_t[b, :, :, band]  # (H, W)
                            pred_mean = pred_band.mean()
                            pred_std = pred_band.std()
                            if pred_std > 1e-6:
                                pred_norm[b, :, :, band] = (pred_band - pred_mean) / pred_std
                            else:
                                pred_norm[b, :, :, band] = pred_band - pred_mean
                    
                    mse_t = self.criterion(pred_norm, gt_masked_t_norm)
                    losses_per_masked.append(mse_t)
                    
                    # Compute spatial correlations for each band
                    # Correlation across spatial locations (H*W pixels) for each band
                    pred_flat = pred_norm.reshape(-1, 12)  # (B*H*W, 12) - normalized predictions
                    gt_flat = gt_masked_t_norm.reshape(-1, 12)  # (B*H*W, 12)
                    
                    band_spatial_corrs = []
                    for band in range(12):
                        gt_vals = gt_flat[:, band]  # (B*H*W,)
                        pred_vals = pred_flat[:, band]  # (B*H*W,)
                        if gt_vals.std() > 1e-6 and pred_vals.std() > 1e-6:
                            gt_centered = gt_vals - gt_vals.mean()
                            pred_centered = pred_vals - pred_vals.mean()
                            cov = (gt_centered * pred_centered).mean()
                            spatial_corr = cov / (gt_vals.std() * pred_vals.std() + 1e-8)
                            band_spatial_corrs.append(spatial_corr.item())
                            correlations_all.append(spatial_corr.item())
                        else:
                            band_spatial_corrs.append(0.0)
                            correlations_all.append(0.0)
                    
                    # Spatial correlation loss for this timestep
                    avg_spatial_corr = np.mean(band_spatial_corrs)
                    spatial_corr_loss = 1.0 - avg_spatial_corr  # Loss: 1 - correlation
                    spatial_corr_losses.append(spatial_corr_loss)
                    
                    # DEBUG: Log prediction statistics
                    if gt_idx < 1 and batch_idx < 2:  # Only for first masked timestep of first 2 samples
                        pred_flat_np = pred_t.detach().cpu().numpy().reshape(-1, 12)
                        pred_std_per_band = [np.std(pred_flat_np[:, b]) for b in range(12)]
                        gt_flat_np = gt_flat.detach().cpu().numpy() if isinstance(gt_flat, torch.Tensor) else gt_flat
                        gt_std_per_band = [np.std(gt_flat_np[:, b]) for b in range(12)]
                        
                        # MORE DETAILED DEBUG INFO
                        self.logger.info(f"    GT DATA DEBUG:")
                        self.logger.info(f"      GT shape: {gt_flat_np.shape}")
                        self.logger.info(f"      GT min/max (band 0): {np.min(gt_flat_np[:, 0]):.6f} / {np.max(gt_flat_np[:, 0]):.6f}")
                        self.logger.info(f"      GT mean (band 0): {np.mean(gt_flat_np[:, 0]):.6f}")
                        self.logger.info(f"      GT unique values count (band 0): {len(np.unique(gt_flat_np[:, 0]))}")
                        self.logger.info(f"      Pred min/max (band 0): {np.min(pred_flat_np[:, 0]):.6f} / {np.max(pred_flat_np[:, 0]):.6f}")
                        
                        self.logger.info(f"    SPATIAL ANALYSIS (timestep {t_idx}):")
                        self.logger.info(f"      Pred std per band (first 3): {[f'{s:.6f}' for s in pred_std_per_band[:3]]}")
                        self.logger.info(f"      GT std per band (first 3):   {[f'{s:.6f}' for s in gt_std_per_band[:3]]}")
                        self.logger.info(f"      Correlations (first 3 bands): {[f'{c:.6f}' for c in band_spatial_corrs[:3]]}")
                        self.logger.info(f"      Avg correlation: {avg_spatial_corr:.6f}")
                
                # Average loss across all masked timesteps
                mse_loss = torch.stack(losses_per_masked).mean()
                mse_loss_scaled = mse_loss  # Keep original scale
                
                # Average spatial correlation loss across timesteps
                spatial_corr_loss_mean = np.mean(spatial_corr_losses) if spatial_corr_losses else 0.0
                spatial_corr_loss = torch.tensor(spatial_corr_loss_mean, device=pred_estimate.device, requires_grad=True)
                
                mean_corr = np.mean(correlations_all) if correlations_all else 0.0
                
                # Track per-band spatial correlations for epoch
                for band_idx in range(12):
                    band_corrs = [correlations_all[gt_idx*12 + band_idx] for gt_idx in range(len(masked_indices)) if gt_idx*12 + band_idx < len(correlations_all)]
                    if band_corrs:
                        epoch_metrics['per_band_correlations'][band_idx].append(np.mean(band_corrs))
                
                # Combined loss: MSE + per-band spatial correlation
                # Both components important: accuracy (MSE) + spatial structure (correlation)
                alpha = 0.3  # 30% MSE, 70% spatial correlation - prioritize learning spatial patterns
                loss = alpha * mse_loss_scaled + (1.0 - alpha) * spatial_corr_loss * 100.0
                
                # Track loss and correlation for this batch
                batch_losses.append(loss.item() * self.config.gradient_accumulation_steps)  # Un-scale for logging
                if mean_corr is not None:
                    batch_correlations.append(mean_corr)
                
                # DEBUG: Log loss components for first few samples
                if batch_idx < 3:
                    self.logger.info(f"  LOSS DEBUG (batch {batch_idx}): MSE={mse_loss_scaled:.6f}, SpatialCorr={spatial_corr_loss:.6f}, Combined={loss*self.config.gradient_accumulation_steps:.6f}")
                    self.logger.info(f"  LOSS COMPONENTS: MSE contribution={alpha*mse_loss_scaled:.6f}, Corr contribution={(1.0-alpha)*spatial_corr_loss*100.0:.6f}")
                    self.logger.info(f"  CORRELATIONS (first 3 bands): {[f'{c:.4f}' for c in correlations_all[:3]]}")
                    self.logger.info(f"  Mean correlation this batch: {mean_corr:.4f}")
                    # Log gradient magnitudes
                    total_grad_norm = 0.0
                    for p in self.head.parameters():
                        if p.grad is not None:
                            total_grad_norm += p.grad.data.norm(2).item() ** 2
                    total_grad_norm = total_grad_norm ** 0.5
                    self.logger.info(f"  GRADIENT: Total norm before step={total_grad_norm:.6e}")
                
                # Scale loss for gradient accumulation
                loss = loss / self.config.gradient_accumulation_steps
            head_time = time.time() - head_start
            
            # Backward pass
            if self.scaler:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # DEBUG: Check if we have gradients in head
            if batch_idx < 2:
                has_grads = False
                for name, p in self.head.named_parameters():
                    if p.grad is not None and p.grad.abs().sum() > 0:
                        has_grads = True
                        self.logger.info(f"  PARAM {name}: grad sum={p.grad.abs().sum():.6e}, param sum={p.abs().sum():.6e}")
                if not has_grads:
                    self.logger.warning("  WARNING: No gradients found in head parameters!")
            
            accumulated_loss += loss.item()
            processed += 1
            self.total_samples_processed += 1  # Track globally
            
            # Gradient accumulation
            if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                # Gradient clipping
                if self.scaler:
                    self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.head.parameters(),
                    self.config.max_grad_norm
                )
                
                # Optimizer step
                if self.scaler:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                
                self.scheduler.step()
                self.optimizer.zero_grad()
                
                epoch_metrics['loss'].append(accumulated_loss)
                epoch_metrics['learning_rate'].append(self.optimizer.param_groups[0]['lr'])
                accumulated_loss = 0.0
                
                self.global_step += 1
                
                # Log progress every N accumulation steps
                if self.global_step % 10 == 0:
                    recent_loss = np.mean(batch_losses[-10:]) if batch_losses else 0.0
                    recent_corr = np.mean(batch_correlations[-10:]) if batch_correlations else 0.0
                    self.logger.info(f"  Step {self.global_step}: Loss={recent_loss:.6f}, AvgCorr={recent_corr:.4f}")
            
            sample_time = time.time() - sample_start
        
        # Compute epoch stats
        metrics = {
            'avg_loss': np.mean(epoch_metrics['loss']) if epoch_metrics['loss'] else 0.0,
            'processed': processed,
            'skipped': skipped,
            'learning_rate': epoch_metrics['learning_rate'][-1] if epoch_metrics['learning_rate'] else self.config.learning_rate,
        }
        
        # Compute per-band average correlations for epoch
        for band_idx in range(12):
            band_corrs = epoch_metrics['per_band_correlations'][band_idx]
            if band_corrs:
                metrics[f'train/corr_band_{band_idx:02d}'] = np.mean(band_corrs)
        
        # Overall correlation across all bands and samples
        all_corrs = [c for band_corrs in epoch_metrics['per_band_correlations'] for c in band_corrs]
        if all_corrs:
            metrics['train/corr_mean'] = np.mean(all_corrs)
            metrics['train/corr_std'] = np.std(all_corrs)
        
        self.logger.info(f"\n{'='*70}")
        self.logger.info(f"✅ EPOCH {self.current_epoch + 1} COMPLETE:")
        self.logger.info(f"   Avg Loss: {metrics['avg_loss']:.6f}")
        self.logger.info(f"   Processed: {processed}, Skipped: {skipped}")
        self.logger.info(f"   Learning Rate: {metrics['learning_rate']:.2e}")
        if 'train/corr_mean' in metrics:
            self.logger.info(f"   Mean Correlation: {metrics['train/corr_mean']:.4f} (±{metrics['train/corr_std']:.4f})")
            # Show per-band correlations
            band_corrs_str = ", ".join([f"B{i:02d}:{metrics.get(f'train/corr_band_{i:02d}', 0.0):.3f}" for i in range(0, 12, 2)])
            self.logger.info(f"   Per-Band Correlations (even bands): {band_corrs_str}")
        self.logger.info(f"{'='*70}\n")
        
        return metrics
    
    def validate(self, sample_indices: list) -> dict:
        """Validate by masking random 1-12 timesteps of S2 and predicting them from others.
        
        Uses same loss as training: 50% MSE + 50% spatial correlation loss.
        Tracks per-band spatial correlations for monitoring.
        """
        self.base_model.eval()
        self.head.eval()
        
        val_losses = []
        val_spatial_corr_losses = []
        val_per_band_correlations = [[] for _ in range(12)]  # Track each band
        processed = 0
        skipped = 0
        
        with torch.no_grad():
            for val_idx, sample_idx in enumerate(sample_indices):
                self.logger.info(f"Validating sample {val_idx+1}/{len(sample_indices)} (idx={sample_idx})")
                sample_data = self._get_sample_with_masking(sample_idx)
                if sample_data is None:
                    skipped += 1
                    continue
                
                masked_sample, s2_gt_list, masked_indices, load_time, num_to_mask, num_timesteps = sample_data
                
                # Call the full model (LatentMIM) - returns 5 values
                encoder_output, decoder_output, pooled_output, reconstructed, extra_metrics = self.base_model(
                    masked_sample,
                    self.config.patch_size
                )
                
                # Check encoder output
                if encoder_output is None:
                    self.logger.error("    Encoder output is None, skipping validation sample")
                    skipped += 1
                    continue
                
                if not hasattr(encoder_output, 'sentinel2_l2a'):
                    self.logger.error(f"    Encoder output missing sentinel2_l2a attribute")
                    skipped += 1
                    continue
                    
                if encoder_output.sentinel2_l2a is None:
                    self.logger.error("    S2 tokens from encoder are None, skipping validation sample")
                    skipped += 1
                    continue
                
                s2_tokens = encoder_output.sentinel2_l2a
                B, H, W, T, C_s2, D = s2_tokens.shape
                self.logger.info(f"    S2 tokens shape: {s2_tokens.shape}")
                
                # Extract tokens from UNMASKED timesteps only
                t_mask = torch.ones(T, dtype=torch.bool, device=self.device)
                t_mask[masked_indices] = False
                
                # Get input tokens from unmasked timesteps
                num_unmasked = T - len(masked_indices)
                s2_tokens_unmasked = s2_tokens[:, :, :, t_mask, :, :]  # (B, H, W, T_unmasked, C_s2, D)
                input_tokens = s2_tokens_unmasked.reshape(-1, D)
                
                device_type = str(self.device).split(':')[0] if ':' in str(self.device) else str(self.device)
                with autocast(device_type=device_type, enabled=self.config.use_mixed_precision):
                    predicted_reflectance_flat = self.head(input_tokens)
                    
                    # Reshape predictions
                    pred_per_token = predicted_reflectance_flat.reshape(B, H, W, num_unmasked, C_s2, 12)
                    pred_per_unmasked = pred_per_token.mean(dim=4)  # (B, H, W, T_unmasked, 12)
                    pred_estimate = pred_per_unmasked.mean(dim=3)  # (B, H, W, 12)
                    self.logger.info(f"    Pred estimate shape: {pred_estimate.shape}")
                    
                    # Compute loss against all masked timesteps
                    losses_per_masked = []
                    spatial_corr_losses = []
                    correlations_all = []
                    
                    for gt_idx, (t_idx, s2_gt_t) in enumerate(zip(masked_indices, s2_gt_list)):
                        # Use single prediction for all masked timesteps
                        pred_t = pred_estimate  # (B, H, W, 12)
                        
                        # Prepare GT for masked timestep
                        # s2_gt_t shape: (H_full, W_full, C_s2)
                        gt_s2_raw = torch.from_numpy(s2_gt_t).to(self.device).float()
                        
                        # Pool GT from full resolution to token resolution
                        gt_s2_perm = gt_s2_raw.permute(2, 0, 1).unsqueeze(1)  # (C_s2, 1, H_full, W_full)
                        kernel_size = 128 // H
                        gt_pooled = torch.nn.functional.avg_pool2d(gt_s2_perm, kernel_size=kernel_size)  # (C_s2, 1, H, W)
                        
                        # Keep all 12 spectral bands with their spatial structure
                        gt_pooled = gt_pooled.squeeze(1).permute(1, 2, 0)  # (H, W, C_s2)
                        gt_masked_t = gt_pooled.unsqueeze(0)  # (1, H, W, 12)
                        gt_masked_t = gt_masked_t.repeat(B, 1, 1, 1)  # (B, H, W, 12)
                        
                        # Normalize both to same scale - per-band normalization ONLY across spatial dimensions
                        gt_masked_t_norm = gt_masked_t.clone()
                        for band in range(12):
                            for b in range(B):
                                band_vals = gt_masked_t[b, :, :, band]  # (H, W) - only spatial
                                band_mean = band_vals.mean()
                                band_std = band_vals.std()
                                if band_std > 0:
                                    gt_masked_t_norm[b, :, :, band] = (band_vals - band_mean) / band_std
                                else:
                                    gt_masked_t_norm[b, :, :, band] = band_vals - band_mean
                        
                        # Compute MSE for this timestep
                        # Normalize predictions per-band ONLY across spatial dimensions to match GT
                        pred_norm = pred_t.clone()
                        for band in range(12):
                            for b in range(B):
                                pred_band = pred_t[b, :, :, band]  # (H, W)
                                pred_mean = pred_band.mean()
                                pred_std = pred_band.std()
                                if pred_std > 1e-6:
                                    pred_norm[b, :, :, band] = (pred_band - pred_mean) / pred_std
                                else:
                                    pred_norm[b, :, :, band] = pred_band - pred_mean
                        
                        mse_loss = self.criterion(pred_norm, gt_masked_t_norm)
                        self.logger.info(f"    Timestep {gt_idx}: MSE Loss = {mse_loss.item():.6f}")
                        losses_per_masked.append(mse_loss.item())
                        
                        # Compute spatial correlations for each band
                        pred_flat = pred_norm.reshape(-1, 12)  # (B*H*W, 12) - normalized predictions
                        gt_flat = gt_masked_t_norm.reshape(-1, 12)
                        
                        band_spatial_corrs = []
                        for band in range(12):
                            gt_vals = gt_flat[:, band]
                            pred_vals = pred_flat[:, band]
                            if gt_vals.std() > 1e-6 and pred_vals.std() > 1e-6:
                                gt_centered = gt_vals - gt_vals.mean()
                                pred_centered = pred_vals - pred_vals.mean()
                                cov = (gt_centered * pred_centered).mean()
                                spatial_corr = cov / (gt_vals.std() * pred_vals.std() + 1e-8)
                                band_spatial_corrs.append(spatial_corr.item())
                                correlations_all.append(spatial_corr.item())
                                # Track per-band correlations
                                val_per_band_correlations[band].append(spatial_corr.item())
                            else:
                                band_spatial_corrs.append(0.0)
                                correlations_all.append(0.0)
                                val_per_band_correlations[band].append(0.0)
                        
                        # Spatial correlation loss for this timestep
                        avg_spatial_corr = np.mean(band_spatial_corrs)
                        spatial_corr_loss = 1.0 - avg_spatial_corr
                        spatial_corr_losses.append(spatial_corr_loss)
                    
                    if losses_per_masked:
                        # Same loss combination as training
                        mse_loss_mean = np.mean(losses_per_masked)  # Keep original scale
                        spatial_corr_loss_mean = np.mean(spatial_corr_losses)
                        
                        # Balance MSE + spatial correlation
                        alpha = 0.3  # 30% MSE, 70% correlation - match training weighting
                        combined_loss = alpha * mse_loss_mean + (1.0 - alpha) * spatial_corr_loss_mean * 100.0
                        
                        self.logger.info(f"    Avg validation loss (combined): {combined_loss:.6f}")
                        val_losses.append(combined_loss)
                        val_spatial_corr_losses.append(spatial_corr_loss_mean)
                    
                    processed += 1
        
        self.base_model.train()
        self.head.train()
        
        val_loss = np.mean(val_losses) if val_losses else 0.0
        val_spatial_corr = np.mean(val_spatial_corr_losses) if val_spatial_corr_losses else 0.0
        
        # Compute per-band average correlations
        val_per_band_corr_means = {}
        for band_idx in range(12):
            if val_per_band_correlations[band_idx]:
                val_per_band_corr_means[f'val/corr_band_{band_idx:02d}'] = np.mean(val_per_band_correlations[band_idx])
        
        # Overall correlation metrics
        all_val_corrs = [c for band_corrs in val_per_band_correlations for c in band_corrs]
        if all_val_corrs:
            val_per_band_corr_means['val/corr_mean'] = np.mean(all_val_corrs)
            val_per_band_corr_means['val/corr_std'] = np.std(all_val_corrs)
        
        self.logger.info(f"  Val Summary - Processed: {processed}, Skipped: {skipped}, Avg Loss: {val_loss:.6f}, Spatial Corr Loss: {val_spatial_corr:.6f}")
        if val_per_band_corr_means:
            self.logger.info(f"  Val Correlations - Mean: {val_per_band_corr_means.get('val/corr_mean', 0.0):.6f}, Std: {val_per_band_corr_means.get('val/corr_std', 0.0):.6f}")
        
        return {
            'val_loss': val_loss,
            'val_spatial_corr_loss': val_spatial_corr,
            'val_processed': processed,
            'val_skipped': skipped,
            **val_per_band_corr_means,
        }
    
    def _save_checkpoint(self, epoch: int, metrics: dict):
        """Save training checkpoint."""
        checkpoint_path = Path(self.config.checkpoint_dir) / f"epoch_{epoch:03d}.pt"
        
        checkpoint = {
            'epoch': epoch,
            'global_step': self.global_step,
            'model_state': self.head.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'scheduler_state': self.scheduler.state_dict(),
            'metrics': metrics,
            'config': self.config.__dict__,
        }
        
        torch.save(checkpoint, checkpoint_path)
        self.logger.info(f"Saved checkpoint to {checkpoint_path}")
    
    def _load_checkpoint(self, checkpoint_path: str):
        """Load training checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.head.load_state_dict(checkpoint['model_state'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state'])
        
        self.start_epoch = checkpoint['epoch'] + 1
        self.global_step = checkpoint['global_step']
        
        self.logger.info(f"Loaded checkpoint from {checkpoint_path}")
        self.logger.info(f"Resuming from epoch {self.start_epoch}, step {self.global_step}")
    
    def train(self):
        """Main training loop."""
        self.logger.info("Starting training loop")
        
        # Time dataset loading
        dataset_start = time.time()
        self._load_dataset()
        dataset_time = time.time() - dataset_start
        self.logger.info(f"Dataset loading took {dataset_time:.2f}s")
        
        # Use dataset's prepared sample indices (already renumbered 0-N)
        self.logger.info("Getting available sample indices...")
        num_samples = len(self.dataset)
        valid_indices = list(range(num_samples))  # Use dataset's internal indexing
        
        self.logger.info(f"Found {len(valid_indices)} available samples")

        if not valid_indices:
            self.logger.error("Could not find any samples!")
            return
        
        # Use first N samples if max_total_samples is set
        if self.config.max_total_samples and self.config.max_total_samples < len(valid_indices):
            valid_indices = valid_indices[:self.config.max_total_samples]
            self.logger.info(f"Using first {len(valid_indices)} samples (max_total_samples={self.config.max_total_samples})")
        
        # Split into train/val (but keep at least 1 training sample)
        num_train = max(1, int(len(valid_indices) * self.config.train_val_split))
        train_indices = valid_indices[:num_train]
        val_indices = valid_indices[num_train:]
        
        self.logger.info(f"Training samples: {len(train_indices)}, Validation samples: {len(val_indices)}")
        
        best_val_loss = float('inf')
        
        for epoch in range(self.start_epoch, self.config.num_epochs):
            self.logger.info(f"\n{'='*70}")
            self.logger.info(f"Epoch {epoch+1}/{self.config.num_epochs}")
            self.logger.info(f"{'='*70}")
            
            # Shuffle training indices for each epoch (different sample order each time)
            shuffled_train_indices = train_indices.copy()
            np.random.shuffle(shuffled_train_indices)
            
            # Training
            train_start = time.time()
            train_metrics = self.train_epoch(shuffled_train_indices)
            train_time = time.time() - train_start
            
            self.logger.info(
                f"Train - Loss: {train_metrics['avg_loss']:.6f}, "
                f"LR: {train_metrics['learning_rate']:.2e}, "
                f"Processed: {train_metrics['processed']}, "
                f"Skipped: {train_metrics['skipped']}, "
                f"Time: {train_time:.2f}s"
            )
            
            # Validation (skip if num_samples_per_epoch is very small for quick testing)
            if self.config.num_samples_per_epoch <= 3:
                self.logger.info("Skipping validation for quick test")
                val_metrics = {'val_loss': 0.0, 'val_processed': 0, 'val_skipped': 0}
            else:
                val_metrics = self.validate(val_indices)
            self.logger.info(
                f"Val - Loss: {val_metrics['val_loss']:.6f}, "
                f"Processed: {val_metrics['val_processed']}, "
                f"Skipped: {val_metrics['val_skipped']}, "
                f"Mean Corr: {val_metrics.get('val/corr_mean', 0.0):.6f}"
            )
            
            # Log to W&B
            if self.wandb_enabled:
                wandb_dict = {
                    'epoch': epoch,
                    'train/loss': train_metrics['avg_loss'],
                    'train/learning_rate': train_metrics['learning_rate'],
                    'val/loss': val_metrics['val_loss'],
                    'global_step': self.global_step,
                }
                
                # Add per-band correlations for training
                for band_idx in range(12):
                    key = f'train/corr_band_{band_idx:02d}'
                    if key in train_metrics:
                        wandb_dict[key] = train_metrics[key]
                
                # Add overall correlation metrics for training
                if 'train/corr_mean' in train_metrics:
                    wandb_dict['train/corr_mean'] = train_metrics['train/corr_mean']
                    wandb_dict['train/corr_std'] = train_metrics['train/corr_std']
                
                # Add validation correlations
                for band_idx in range(12):
                    key = f'val/corr_band_{band_idx:02d}'
                    if key in val_metrics:
                        wandb_dict[key] = val_metrics[key]
                
                # Add overall validation correlation metrics
                if 'val/corr_mean' in val_metrics:
                    wandb_dict['val/corr_mean'] = val_metrics['val/corr_mean']
                    wandb_dict['val/corr_std'] = val_metrics['val/corr_std']
                
                try:
                    wandb.log(wandb_dict)
                    if epoch % 5 == 0:  # Log success every 5 epochs
                        self.logger.info(f"✅ Logged to W&B (epoch {epoch+1})")
                except Exception as e:
                    self.logger.error(f"❌ Failed to log to W&B: {e}")
                    import traceback
                    self.logger.error(traceback.format_exc())
            
            # Save checkpoint
            if (epoch + 1) % self.config.save_interval == 0:
                metrics = {**train_metrics, **val_metrics}
                self._save_checkpoint(epoch, metrics)
            
            # Track best validation loss
            if val_metrics['val_loss'] < best_val_loss:
                best_val_loss = val_metrics['val_loss']
                self.logger.info(f"New best validation loss: {best_val_loss:.6f}")
        
        self.logger.info("\n" + "="*70)
        self.logger.info("✅ Training complete!")
        self.logger.info("="*70)
        
        if self.wandb_enabled:
            wandb.finish()


def main():
    """Main entry point."""
    config = TrainingConfig(
        patch_size=8,
        num_epochs=50,
        num_samples_per_epoch=20,
        batch_size=2,
        gradient_accumulation_steps=1,
        learning_rate=1e-4,
        max_grad_norm=100.0,
        max_total_samples=1000,  # Use all available samples
        warmup_steps=100,
        use_wandb=True,
        hidden_multiplier=0.8,  # Adjust to control parameters: 0.5 (~370K), 0.25 (~150K), 1.0 (~600K), 2.0 (~2.4M)
    )

    trainer = ReflectanceReconstructionTrainer(config)
    trainer.train()


if __name__ == '__main__':
    main()

