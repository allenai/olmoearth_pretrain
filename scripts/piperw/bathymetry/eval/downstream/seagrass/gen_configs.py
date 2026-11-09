"""Seagrass configs for the v1.3-tiny arms, copied from round 1's *_fixdata configs.

Each round-1 base-arm config (fold x method x seed, train + polygon eval) is copied with only
the bundle path and run name changed, so every other setting matches round 1.
"""

from pathlib import Path

HERE = Path(
    "/weka/dfive-default/agent/piperbots/2026-10-29-piperw-bathymetry-direct-supervision/downstream/seagrass"
)
OLD = Path(
    "/weka/dfive-default/agent/piperbots/2026-10-26-piperw-bathymetry-target/seagrass/configs"
)
OLD_BUNDLE = "/weka/dfive-default/agent/piperbots/2026-10-26-piperw-bathymetry-target/bundles/base"
BUNDLES = HERE.parent / "bundles"
ARMS = ["sdbprobe", "sdbsup", "btsup", "sdbbtsup"]


def main() -> None:
    """Write one config per arm, fold, method and seed from round 1's configs."""
    out = HERE / "configs"
    out.mkdir(exist_ok=True)
    runs = []
    for src in sorted(OLD.glob("bathy_base_*_fixdata.yaml")):
        base_run = src.stem  # bathy_base_<fold>_<method>_s<seed>_fixdata
        for arm in ARMS:
            run = base_run.replace("bathy_base_", f"bathy_v13{arm}_")
            for suffix in ("", "_polyeval"):
                text = (OLD / f"{base_run}{suffix}.yaml").read_text()
                assert (
                    text.count(OLD_BUNDLE) == 1
                    and text.count(f"run_name: {base_run}{suffix}\n") == 1
                )
                text = text.replace(OLD_BUNDLE, str(BUNDLES / f"v13_{arm}"))
                text = text.replace(
                    f"run_name: {base_run}{suffix}\n", f"run_name: {run}{suffix}\n"
                )
                (out / f"{run}{suffix}.yaml").write_text(text)
            runs.append(run)
    (HERE / "configs" / "runs.txt").write_text("\n".join(runs) + "\n")
    print(f"{len(runs)} runs -> {out}")


if __name__ == "__main__":
    main()
