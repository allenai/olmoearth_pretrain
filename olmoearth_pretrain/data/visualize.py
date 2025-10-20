"""Code for loading and Visualizing Samples from the OlmoEarth Pretrain Dataset."""

import logging
import os
from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from einops import rearrange
from matplotlib.figure import Figure
from upath import UPath

from olmoearth_pretrain.data.constants import Modality
from olmoearth_pretrain.data.dataset import GetItemArgs, OlmoEarthDataset
from olmoearth_pretrain.data.utils import convert_to_db

logger = logging.getLogger(__name__)

WORLDCOVER_LEGEND = {
    10: ("#006400", "Tree cover"),
    20: ("#ffbb22", "Shrubland"),
    30: ("#ffff4c", "Grassland"),
    40: ("#f096ff", "Cropland"),
    50: ("#fa0000", "Built-up"),
    60: ("#b4b4b4", "Bare / sparse vegetation"),
    70: ("#f0f0f0", "Snow and ice"),
    80: ("#0064c8", "Permanent water bodies"),
    90: ("#0096a0", "Herbaceous wetland"),
    95: ("#00cf75", "Mangroves"),
    100: ("#fae6a0", "Moss and lichen"),
}


def visualize_sample(
    dataset: OlmoEarthDataset,
    sample_index: int,
    out_dir: str | Path | UPath,
) -> Figure:
    """Visualize a sample from the OlmoEarth Pretrain Dataset in a grid format.

    - Each modality is placed on its own row.
    - Each band of that modality is shown in columns of that row.
    - If the modality is LATLON, plot the coordinate on a base map.
    - If the modality is WORLDCOVER, use a discrete colormap and show a legend.
    Saves the resulting figure to a .png file.
    """
    wc_classes_sorted = sorted(WORLDCOVER_LEGEND.keys())  # [10, 20, 30, ...]
    wc_colors = [WORLDCOVER_LEGEND[val][0] for val in wc_classes_sorted]
    # Construct a discrete colormap and corresponding norm
    wc_cmap = mcolors.ListedColormap(wc_colors)
    wc_bounds = wc_classes_sorted + [
        wc_classes_sorted[-1] + 1
    ]  # e.g., up to 101 if last is 100
    wc_norm = mcolors.BoundaryNorm(wc_bounds, wc_cmap.N)
    logger.info(f"Visualizing sample index: {sample_index}")

    args = GetItemArgs(
        idx=sample_index,
        patch_size=1,
        sampled_hw_p=256,
    )
    _, sample = dataset[args]
    modalities = sample.modalities
    if not modalities:
        logger.warning("No modalities found in this sample.")
        return
    total_rows = len(modalities)
    # At least 1 column, but also handle the largest band_order among the modalities
    max_bands = 1
    for modality_name in modalities:
        if modality_name == "timestamps":
            continue
        modality_spec = Modality.get(modality_name)
        if modality_spec != Modality.LATLON:
            max_bands = max(max_bands, len(modality_spec.band_order))

    fig, axes = plt.subplots(
        nrows=total_rows + 1,
        ncols=max_bands,
        figsize=(5 * max_bands, 5 * total_rows),
        squeeze=False,
    )

    # Load lat/lon data, e.g. [lat, lon]
    assert sample.latlon is not None
    latlon_data = sample.latlon
    lat = float(latlon_data[0])
    lon = float(latlon_data[1])

    # Remove old Axes & re-add Cartopy
    fig.delaxes(axes[0, 0])

    axes[0, 0] = fig.add_subplot(
        total_rows,
        max_bands,
        1,
        projection=ccrs.PlateCarree(),
    )
    ax_map = axes[0, 0]

    ax_map.add_feature(cfeature.COASTLINE)
    ax_map.add_feature(cfeature.LAND, alpha=0.1)
    ax_map.add_feature(cfeature.OCEAN, alpha=0.1)
    ax_map.scatter(lon, lat, transform=ccrs.PlateCarree(), c="red", s=60)

    ax_map.set_global()
    ax_map.set_extent([-180, 180, -90, 90], crs=ccrs.PlateCarree())
    ax_map.gridlines()
    ax_map.set_title(f"{Modality.LATLON.name.upper()} (Lat: {lat:.2f}, Lon: {lon:.2f})")

    # Hide unused columns
    for empty_col in range(max_bands - 1, 0, -1):
        axes[0, empty_col].axis("off")
    sample_dict = sample.as_dict()
    for row_idx, (modality_name, modality_data) in enumerate(
        sample_dict.items(), start=1
    ):
        assert modality_data is not None
        if modality_name == "timestamps" or modality_name == Modality.LATLON.name:
            continue
        logger.info(f"Plotting modality: {modality_name}")
        modality_spec = Modality.get(modality_name)
        # 4B. Plot other modalities
        if modality_spec == Modality.SENTINEL1:
            modality_data = convert_to_db(modality_data)
        logger.info(f"Modality data shape (loaded): {modality_data.shape}")

        # If temporal [H, W, T, C], take first time step
        if modality_spec.is_spatial:
            modality_data = modality_data[:, :, 0]
            logger.info(
                f"Modality data shape after first time step: {modality_data.shape}"
            )

        # Rearrange to [C, H, W]
        modality_data = rearrange(modality_data, "h w c -> c h w")
        logger.info(f"Modality data shape after rearranging: {modality_data.shape}")

        for band_i, band_name in enumerate(modality_spec.band_order):
            ax = axes[row_idx, band_i]
            channel_data = modality_data[band_i]  # shape [H, W]

            # 4B(i). If WorldCover, use discrete colormap & legend
            if modality_spec == Modality.WORLDCOVER:
                _ = ax.imshow(channel_data, cmap=wc_cmap, norm=wc_norm)
                ax.set_title(f"{modality_spec.name.upper()} - {band_name}")

                # Create legend patches
                patches = []
                for val in wc_classes_sorted:
                    color_hex, label_txt = WORLDCOVER_LEGEND[val]
                    patch = mpatches.Patch(color=color_hex, label=label_txt)
                    patches.append(patch)

                # Place legend to the right of the axes (outside the main image area)
                ax.legend(
                    handles=patches,
                    loc="center left",
                    bbox_to_anchor=(1, 0.5),  # Adjust as needed, e.g. (1.05, 0.5)
                    borderaxespad=0.0,
                    fontsize=8,
                )
                ax.axis("off")

            else:
                _ = ax.imshow(channel_data, cmap="viridis")
                ax.set_title(f"{modality_spec.name.upper()} — {band_name}")
                ax.axis("off")

        # Hide any unused columns if fewer bands than max_bands
        used_cols = len(modality_spec.band_order)
        for empty_col in range(used_cols, max_bands):
            axes[row_idx, empty_col].axis("off")

    plt.tight_layout()
    fig.subplots_adjust(
        wspace=0.3, hspace=0.3, left=0.05, right=0.95, top=0.95, bottom=0.05
    )
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"sample_{sample_index}.png")
    fig.savefig(out_path)
    logger.info(f"Saved visualization to {out_path}")
    logger.info(f"type(fig): {type(fig)}")
    return fig
