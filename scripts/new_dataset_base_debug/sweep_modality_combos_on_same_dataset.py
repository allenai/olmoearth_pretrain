"""This script is used to sweep different combinations of training modalities.

for latent_mim_debug.py.
"""

import argparse
import os
import subprocess  # nosec

from helios.data.constants import Modality, ModalitySpec

OLD_MODALITIES = [
    Modality.SENTINEL2_L2A,
    Modality.SENTINEL1,
    Modality.WORLDCOVER,
]

OLD_MODALITIES_SRTM = OLD_MODALITIES + [Modality.SRTM]

OLD_MODALITIES_SRTM_LANDSAT = OLD_MODALITIES_SRTM + [Modality.LANDSAT]

OLD_MODALITIES_SRTM_LANDSAT_OPENSTREETMAP = OLD_MODALITIES_SRTM_LANDSAT + [
    Modality.OPENSTREETMAP_RASTER
]

OLD_MODALITIES_LANDSAT = OLD_MODALITIES + [Modality.LANDSAT]

OLD_MODALITIES_OPENSTREETMAP = OLD_MODALITIES + [Modality.OPENSTREETMAP_RASTER]


# Combine all the combinations we want to test
MODALITY_COMBINATIONS = (
    OLD_MODALITIES,
    OLD_MODALITIES_SRTM,
    OLD_MODALITIES_SRTM_LANDSAT,
    OLD_MODALITIES_SRTM_LANDSAT_OPENSTREETMAP,
    OLD_MODALITIES_LANDSAT,
    OLD_MODALITIES_OPENSTREETMAP,
)


def format_training_modalities(modalities: list[ModalitySpec]) -> str:
    """Format the training modalities as a single comma-separated list."""
    # These modalities are used exactly as they are in the base script
    # when defining TRAINING_MODALITIES
    formatted_modalities = []
    for m in modalities:
        # Convert modality name to lowercase to match format in base script
        modality_value = m.name.lower()
        formatted_modalities.append(f'"{modality_value}"')

    # Join with commas
    modality_list = ",".join(formatted_modalities)
    # Return as a command line argument
    return f"--common.training_modalities=[{modality_list}]"


def main() -> None:
    """Main function to run the script."""
    parser = argparse.ArgumentParser(
        description="Sweep different combinations of training modalities."
    )
    parser.add_argument(
        "--base_script_path",
        type=str,
        help="Path to the base Python script to run with different modality combinations",
    )
    parser.add_argument(
        "--cluster",
        type=str,
        default="ai2/jupiter-cirrascale-2",
        help="Cluster to run on",
    )
    args = parser.parse_args()

    # Ensure the base script exists
    if not os.path.exists(args.base_script_path):
        raise FileNotFoundError(f"Base script not found: {args.base_script_path}")

    # Base command template using the provided script path
    print(f"Base script path: {args.base_script_path}")
    base_command = (
        f"python3 {args.base_script_path} "
        f"launch {{run_name}} {args.cluster} "
        f"--dataset.h5py_dir=/weka/dfive-default/helios/dataset/presto/h5py_data/landsat_naip_openstreetmap_raster_sentinel1_sentinel2_l2a_srtm_worldcover/118861 "
        f"{{modality_args}}"
    )

    # Iterate over all combinations of modalities
    for modality_combo in MODALITY_COMBINATIONS:
        # Generate a descriptive name for the run
        modality_combo_name = "_".join([m.name.lower() for m in modality_combo])
        script_name = os.path.basename(args.base_script_path).replace(".py", "")
        run_name = f"new_2filtering1_{script_name}_modalities_{modality_combo_name}"

        # Format the modality arguments
        modality_args = format_training_modalities(modality_combo)

        # Construct full command
        command = base_command.format(
            run_name=run_name,
            modality_args=modality_args,
        )

        print(f"Launching: {command}")

        # Execute the command
        subprocess.run(command, shell=True, check=True)  # nosec


if __name__ == "__main__":
    main()
