"""utility Functions for hyper parameter sweeps."""

EXIT_CONFIG_TYPES = ["zero", "half", "full", "varied"]


def build_token_exit_config(
    config_type: str, modality_names: list[str], encoder_depth: int
) -> str:
    """Build the token exit config for an experiment."""
    if config_type not in EXIT_CONFIG_TYPES:
        raise ValueError(f"Invalid config type: {config_type}")
    if config_type == "zero":
        return " ".join(
            f"--train_module.token_exit_cfg.{modality_name}=0"
            for modality_name in modality_names
        )
    elif config_type == "half":
        return " ".join(
            f"--train_module.token_exit_cfg.{modality_name}={encoder_depth // 2}"
            for modality_name in modality_names
        )
    elif config_type == "full":
        return " ".join(
            f"--train_module.token_exit_cfg.{modality_name}={encoder_depth}"
            for modality_name in modality_names
        )
    elif config_type == "varied":
        varied_args = []
        for modality_name in modality_names:
            if modality_name not in ["latlon", "worldcover"]:
                varied_args.append(
                    f"--train_module.token_exit_cfg.{modality_name}={encoder_depth}"
                )
            else:
                varied_args.append(f"--train_module.token_exit_cfg.{modality_name}=0")
        return " ".join(varied_args)
    else:
        raise ValueError(f"Invalid config type: {config_type}")


MODEL_SIZE_ARGS = {
    "nano": {
        "decoder_depth": 4,
        "encoder_embedding_size": 128,
        "decoder_embedding_size": 128,
        "encoder_depth": 4,
        "encoder_num_heads": 8,
        "decoder_num_heads": 8,
        "mlp_ratio": 4.0,
    },
    "tiny": {
        "decoder_depth": 12,
        "encoder_embedding_size": 192,
        "decoder_embedding_size": 192,
        "encoder_depth": 12,
        "encoder_num_heads": 3,
        "decoder_num_heads": 3,
        "mlp_ratio": 4.0,
    },
    "tiny_more_heads": {
        "decoder_depth": 12,
        "encoder_embedding_size": 192,
        "decoder_embedding_size": 192,
        "encoder_depth": 12,
        "encoder_num_heads": 8,
        "decoder_num_heads": 8,
        "mlp_ratio": 4.0,
    },
    "base": {
        "decoder_depth": 12,
        "encoder_embedding_size": 768,
        "decoder_embedding_size": 768,
        "encoder_depth": 12,
        "encoder_num_heads": 12,
        "decoder_num_heads": 12,
        "mlp_ratio": 4.0,
    },
    "large": {
        "decoder_depth": 24,
        "encoder_embedding_size": 1024,
        "decoder_embedding_size": 1024,
        "encoder_depth": 24,
        "encoder_num_heads": 16,
        "decoder_num_heads": 16,
        "mlp_ratio": 4.0,
    },
    "giga": {
        "decoder_depth": 40,
        "encoder_embedding_size": 1536,
        "decoder_embedding_size": 1536,
        "encoder_depth": 40,
        "encoder_num_heads": 24,
        "decoder_num_heads": 24,
        "mlp_ratio": 4.0,
    },
    "tiny_shallow_decoder": {
        "decoder_depth": 4,
        "encoder_embedding_size": 192,
        "decoder_embedding_size": 192,
        "encoder_depth": 12,
        "encoder_num_heads": 3,
        "decoder_num_heads": 3,
        "mlp_ratio": 4.0,
    },
    "base_shallow_decoder": {
        "decoder_depth": 4,
        "encoder_embedding_size": 768,
        "decoder_embedding_size": 768,
        "encoder_depth": 12,
        "encoder_num_heads": 12,
        "decoder_num_heads": 12,
        "mlp_ratio": 4.0,
    },
    "large_shallow_decoder": {
        "decoder_depth": 4,
        "encoder_embedding_size": 1024,
        "decoder_embedding_size": 1024,
        "encoder_depth": 24,
        "encoder_num_heads": 16,
        "decoder_num_heads": 16,
        "mlp_ratio": 4.0,
    },
    "giga_shallow_decoder": {
        "decoder_depth": 4,
        "encoder_embedding_size": 1536,
        "decoder_embedding_size": 1536,
        "encoder_depth": 40,
        "encoder_num_heads": 24,
        "decoder_num_heads": 24,
        "mlp_ratio": 4.0,
    },
    "tiny_super_shallow_decoder": {
        "decoder_depth": 2,
        "encoder_embedding_size": 192,
        "decoder_embedding_size": 192,
        "encoder_depth": 12,
        "encoder_num_heads": 3,
        "decoder_num_heads": 3,
        "mlp_ratio": 4.0,
    },
    "base_super_shallow_decoder": {
        "decoder_depth": 2,
        "encoder_embedding_size": 768,
        "decoder_embedding_size": 768,
        "encoder_depth": 12,
        "encoder_num_heads": 12,
        "decoder_num_heads": 12,
        "mlp_ratio": 4.0,
    },
    "large_super_shallow_decoder": {
        "decoder_depth": 2,
        "encoder_embedding_size": 1024,
        "decoder_embedding_size": 1024,
        "encoder_depth": 24,
        "encoder_num_heads": 16,
        "decoder_num_heads": 16,
        "mlp_ratio": 4.0,
    },
    "giga_super_shallow_decoder": {
        "decoder_depth": 2,
        "encoder_embedding_size": 1536,
        "decoder_embedding_size": 1536,
        "encoder_depth": 40,
        "encoder_num_heads": 24,
        "decoder_num_heads": 24,
        "mlp_ratio": 4.0,
    },
}
