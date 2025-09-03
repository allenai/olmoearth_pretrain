"""Sweep different combinations of modalities on the same dataset."""

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

OLD_MODALITIES_SRTM_OPENSTREETMAP = OLD_MODALITIES_SRTM + [
    Modality.OPENSTREETMAP_RASTER,
]


# Combine all the combinations we want to test
MODALITY_COMBINATIONS = (
    # OLD_MODALITIES_SRTM,
    OLD_MODALITIES_SRTM_LANDSAT,
    OLD_MODALITIES_SRTM_LANDSAT_OPENSTREETMAP,
    OLD_MODALITIES_LANDSAT,
    # OLD_MODALITIES_SRTM_OPENSTREETMAP,
    # OLD_MODALITIES_OPENSTREETMAP,
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
    base_command = (
        "python3 scripts/2025_04_23_benchmarking_ladder/base_galileo_max.py "
        "launch {run_name} ai2/jupiter-cirrascale-2 "
        "--train_module.contrastive_config.loss_config.type=InfoNCE "
        "--train_module.contrastive_config.loss_config.weight={contrastive_weight} "
        "--launch.priority=urgent "
        "--launch.num_gpus=8 "
        "{modality_args}"
    )
    print(len(MODALITY_COMBINATIONS))
    # Iterate over all combinations of modalities
    for modality_combo in MODALITY_COMBINATIONS:
        # Generate a descriptive name for the run
        modality_combo_name = "_".join([m.name.lower() for m in modality_combo])
        run_name = f"8_ddp_galileo_base_contrastive_0.05_{modality_combo_name}"

        # Format the modality arguments
        modality_args = format_training_modalities(modality_combo)

        # Construct full command
        command = base_command.format(
            run_name=run_name,
            modality_args=modality_args,
            contrastive_weight=0.05,
        )

        print(f"Launching: {command}")

        # Execute the command
        subprocess.run(command, shell=True, check=True)  # nosec


if __name__ == "__main__":
    main()
