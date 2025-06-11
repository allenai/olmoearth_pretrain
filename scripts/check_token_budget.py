"""Script to check the token budget for a given sample."""

import argparse

import pandas as pd
import torch

from helios.data.constants import Modality
from helios.data.dataset import HeliosSample


def create_dummy_sample(modalities: list[str], H: int, W: int, T: int) -> HeliosSample:
    """Creates a dummy HeliosSample for testing."""
    sample_data = {}

    # We always need timestamps for self.time
    sample_data["timestamps"] = torch.zeros((T, 3))

    for modality_name in modalities:
        modality_spec = Modality.get(modality_name)
        shape = []
        if modality_spec.is_spatial:
            shape.extend([H, W])
        if modality_spec.is_multitemporal:
            shape.append(T)
        else:
            # for space-only varying, T is 1
            if modality_spec.is_spatial:
                shape.append(1)

        shape.append(modality_spec.num_bands)

        # Adjust for unbatched sample shape expectations.
        # subset() expects unbatched data, e.g., (H, W, T, C) or (H, W, C)
        # For space-only, the time dim is squeezed out if it is 1.
        if modality_spec.is_space_only_varying:
            shape = (H, W, modality_spec.num_bands)

        sample_data[modality_name] = torch.zeros(shape)

    return HeliosSample(**sample_data)


def main():
    """Main function to run the analysis."""
    parser = argparse.ArgumentParser(
        description="Check token budget for a given sample."
    )
    parser.add_argument("--h_w_p_min", type=int, default=4, help="Min value for h_w_p.")
    parser.add_argument(
        "--h_w_p_max", type=int, default=17, help="Max value for h_w_p."
    )
    parser.add_argument(
        "--token_budget_min", type=int, default=1000, help="Min token budget."
    )
    parser.add_argument(
        "--token_budget_max", type=int, default=20001, help="Max token budget."
    )
    parser.add_argument(
        "--token_budget_step", type=int, default=500, help="Token budget step."
    )
    args = parser.parse_args()

    # Modalities to include in the dummy sample.
    # A mix of spacetime and space-only varying modalities.
    modalities_to_test = [
        Modality.SENTINEL2_L2A.name,
        Modality.WORLDCOVER.name,
        Modality.SENTINEL1.name,
        Modality.LANDSAT.name,
        Modality.OPENSTREETMAP_RASTER.name,
        Modality.SRTM.name,
        Modality.LATLON.name,
    ]

    H, W, T = 256, 256, 12
    sample = create_dummy_sample(modalities_to_test, H, W, T)

    h_w_p_values = list(range(args.h_w_p_min, args.h_w_p_max))
    token_budgets = list(
        range(args.token_budget_min, args.token_budget_max, args.token_budget_step)
    )

    results = {}
    for h_w_p in h_w_p_values:
        max_t_values = []
        for budget in token_budgets:
            try:
                max_t = sample._get_max_t_within_token_budget(
                    h_w_p=h_w_p, max_tokens_per_instance=budget
                )
                max_t_values.append(max_t)
            except ValueError as e:
                print(f"Error for h_w_p={h_w_p} and budget={budget}: {e}")
                max_t_values.append(0)
        results[f"h_w_p={h_w_p}"] = max_t_values

    df = pd.DataFrame(results, index=[f"{b}" for b in token_budgets])
    df.index.name = "Token Budget"

    print("This table shows the maximum number of time steps (t) possible for a given")
    print("h_w_p (patch tokens in height/width) and a total token budget.\n")
    print(f"Sample created with modalities: {modalities_to_test}")
    print(f"Base dimensions: H={H}, W={W}, T={T}\n")
    print(df)
    # write to csv
    df.to_csv("token_budget_results.csv")


if __name__ == "__main__":
    main()
