"""Compare the per-band read weight of several checkpoints' pixel projections.

Each modality's patch embedding starts with a per-pixel linear map
``pixel_proj.0`` of shape ``[hidden, n_bands]``, applied identically at every
pixel before patchification. Column j is the only path band j has into the
model, so ``||W[:, j]||`` is how hard the encoder reads that band, and the
Frobenius norm over a modality's whole matrix is how hard it reads that sensor.

Three arms of the new-maps recipe are compared against the d768 DN reference,
and each contrast has its own built-in control:

- **d768 reflectance** differs from the reference only in Landsat radiometry, so
  its S2 and S1 divergence is pure seed/trajectory noise and sets the scale the
  Landsat difference has to be read against.
- **d128 DN** differs only in register-bottleneck width, and every modality sees
  byte-identical data, so its divergence on *all three* sensors is a second,
  independent noise scale -- what a change downstream of the patch embedding
  does to the patch embedding.

``pixel_proj.0`` is ``[64, n_bands]`` in both widths (the bottleneck is further
down the encoder), so the matrices stay directly comparable across arms.

Two caveats the numbers cannot express on their own:

- Absolute norms are only comparable across arms because all normalize inputs
  to roughly unit scale before the projection (each arm with its own computed
  norm config). Within-modality shares are reported alongside precisely because
  they are invariant to overall weight growth and to that scaling.
- ``pixel_proj.0`` is followed by a ReLU and more layers, so a column norm
  bounds a band's influence rather than measuring it end to end.

Reads the DCP shards directly: torch's own reader rejects these checkpoints
(``_StorageInfo`` has no ``transform_descriptors``), but each tensor is stored
as a self-contained ``torch.save`` blob at a known offset.

Usage:
    python scripts/tools/compare_band_read_weights.py --steps 667200
    python scripts/tools/compare_band_read_weights.py --step-stride 25000 --plot out.png
"""

import argparse
import io
import json
import os

import torch
import torch.distributed.checkpoint as dcp

from olmoearth_pretrain.data.constants import Modality

CKPT_ROOT = "/weka/dfive-default/helios/checkpoints/yawenzzzz"
D768_RUN = "regbtl_v1_2_gdyn_d768_wideread_regsup_w0p1_newsampling_psuniform_newmaps"
D128_RUN = "regbtl_v1_2_gdyn_d128_wideread_regsup_w0p1_newsampling_psuniform_newmaps"

# The reference is first; every divergence is measured against it. Order here is
# the order everything is plotted and tabulated in.
REFERENCE = "d768 DN"
ARMS: dict[str, dict[str, str]] = {
    REFERENCE: {"run": D768_RUN, "color": "#4C72B0", "linestyle": "-"},
    "d768 refl": {
        "run": f"{D768_RUN}_landsat_refl",
        "color": "#DD8452",
        "linestyle": "--",
    },
    "d128 DN": {"run": D128_RUN, "color": "#55A868", "linestyle": "-."},
}
OTHERS = [name for name in ARMS if name != REFERENCE]

MODALITIES = ("sentinel2_l2a", "sentinel1", "landsat")

# S2 bandsets carry native resolutions; grouping bands this way is what makes
# the 60 m atmospheric bands' behaviour legible against the 10 m ones.
S2_RESOLUTION = {
    "B02": 10,
    "B03": 10,
    "B04": 10,
    "B08": 10,
    "B05": 20,
    "B06": 20,
    "B07": 20,
    "B8A": 20,
    "B11": 20,
    "B12": 20,
    "B01": 60,
    "B09": 60,
}


def weight_key(modality: str, encoder: str) -> str:
    """State-dict key of a modality's per-pixel projection weight."""
    return (
        f"model.{encoder}.patch_embeddings.per_modality_embeddings."
        f"{modality}.{modality}__0.pixel_proj.0.weight"
    )


def read_tensors(ckpt_dir: str, keys: list[str]) -> dict[str, torch.Tensor]:
    """Pull specific tensors out of a DCP checkpoint by direct blob read."""
    shard_dir = os.path.join(ckpt_dir, "model_and_optim")
    metadata = dcp.FileSystemReader(shard_dir).read_metadata()
    wanted = set(keys)
    located = {
        index.fqn: info
        for index, info in metadata.storage_data.items()
        if getattr(index, "fqn", None) in wanted
    }
    missing = wanted - set(located)
    if missing:
        raise KeyError(f"{ckpt_dir}: keys absent from checkpoint: {sorted(missing)}")

    out = {}
    for key, info in located.items():
        with open(os.path.join(shard_dir, info.relative_path), "rb") as f:
            f.seek(info.offset)
            blob = f.read(info.length)
        out[key] = torch.load(io.BytesIO(blob), weights_only=True)
    return out


def load_arm(ckpt_dir: str, encoder: str) -> dict[str, torch.Tensor]:
    """Every modality's pixel_proj weight from one checkpoint."""
    keys = [weight_key(m, encoder) for m in MODALITIES]
    tensors = read_tensors(ckpt_dir, keys)
    out = {}
    for modality in MODALITIES:
        w = tensors[weight_key(modality, encoder)].float()
        bands = list(Modality.get(modality).band_order)
        if len(bands) != w.shape[1]:
            raise ValueError(
                f"{modality}: {len(bands)} bands in band_order but "
                f"{w.shape[1]} columns in pixel_proj.0.weight"
            )
        out[modality] = w
    return out


def summarize(w: torch.Tensor, modality: str) -> dict:
    """Per-band column norms and the matrix total."""
    return {
        "bands": list(Modality.get(modality).band_order),
        "norms": [float(x) for x in w.norm(dim=0)],
        "total": float(w.norm()),
    }


def collect(steps: list[int], encoder: str) -> dict:
    """Measure every arm at every step, plus each arm's divergence from the reference.

    The divergence is the point of the exercise. All arms start from the same
    init, and every contrast holds at least S2 and S1 byte-identical, so those
    sensors give the drift floor that a real radiometry effect has to clear.
    """
    out: dict = {
        "arms": {name: {} for name in ARMS},
        "divergence": {name: {} for name in OTHERS},
    }
    for step in steps:
        dirs = {
            name: os.path.join(CKPT_ROOT, cfg["run"], f"step{step}")
            for name, cfg in ARMS.items()
        }
        absent = [name for name, d in dirs.items() if not os.path.isdir(d)]
        if absent:
            print(f"  skip step{step} (absent in {', '.join(absent)})", flush=True)
            continue
        weights = {name: load_arm(d, encoder) for name, d in dirs.items()}
        for name, arm in weights.items():
            out["arms"][name][step] = {m: summarize(arm[m], m) for m in MODALITIES}
        ref = weights[REFERENCE]
        for name in OTHERS:
            out["divergence"][name][step] = {
                m: float((ref[m] - weights[name][m]).norm() / ref[m].norm())
                for m in MODALITIES
            }
        print(f"  read step{step}", flush=True)
    return out


def common_steps(data: dict) -> list[int]:
    """Steps present in every arm, in order."""
    per_arm = [set(steps) for steps in data["arms"].values()]
    return sorted(set.intersection(*per_arm)) if per_arm else []


def report(data: dict, step: int) -> None:
    """Print the per-band table for every arm at one step, against the reference."""
    arms = {name: data["arms"][name].get(step) for name in ARMS}
    if any(a is None for a in arms.values()):
        return

    print(f"\n=== per-band read weight at step {step} (||W[:, band]||)")
    for modality in MODALITIES:
        stats = {name: arm[modality] for name, arm in arms.items()}
        ref = stats[REFERENCE]

        totals = "   ".join(
            f"{name} {stats[name]['total']:.3f}"
            + (
                ""
                if name == REFERENCE
                else f" ({100 * (stats[name]['total'] / ref['total'] - 1):+.1f}%)"
            )
            for name in ARMS
        )
        print(f"\n-- {modality}   total (Frobenius): {totals}")

        # Share of the summed column norms (not the Frobenius total): this is
        # the spectral budget split, invariant to overall weight growth.
        sums = {name: sum(s["norms"]) for name, s in stats.items()}
        header = f"   {'band':6s}"
        for name in ARMS:
            header += f" {name:>10s}"
        for name in OTHERS:
            header += f" {name + ' %':>12s}"
        for name in OTHERS:
            header += f" {name + ' pp':>13s}"
        print(header)

        for i, band in enumerate(ref["bands"]):
            row = f"   {band:6s}"
            for name in ARMS:
                row += f" {stats[name]['norms'][i]:10.4f}"
            for name in OTHERS:
                delta = stats[name]["norms"][i] / ref["norms"][i] - 1
                row += f" {100 * delta:+11.1f}%"
            ref_share = ref["norms"][i] / sums[REFERENCE]
            for name in OTHERS:
                share = stats[name]["norms"][i] / sums[name]
                row += f" {100 * (share - ref_share):+13.2f}"
            print(row)


def report_divergence(data: dict, step: int) -> None:
    """Print each arm's whole-matrix divergence from the reference at one step."""
    print(
        f"\n=== divergence from {REFERENCE} at step {step} "
        f"(||W_ref - W_arm|| / ||W_ref||)"
    )
    print(f"   {'arm':12s}" + "".join(f" {m:>16s}" for m in MODALITIES))
    for name in OTHERS:
        row = data["divergence"][name].get(step)
        if row is None:
            continue
        print(f"   {name:12s}" + "".join(f" {100 * row[m]:15.1f}%" for m in MODALITIES))


def plot(data: dict, path: str) -> None:
    """Per-band comparison, spectral budget, trajectories and the drift floor."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    steps = common_steps(data)
    last = steps[-1]
    ks = [s / 1000 for s in steps]
    names = list(ARMS)
    colors = {name: ARMS[name]["color"] for name in names}
    styles = {name: ARMS[name]["linestyle"] for name in names}

    fig, axes = plt.subplots(4, 3, figsize=(19, 16))
    fig.suptitle(
        "Per-pixel read weight ||W[:, band]|| of pixel_proj.0 across three "
        f"new-maps arms, step {last}\n"
        f"Divergence is measured against {REFERENCE}; S2 and S1 see "
        "byte-identical data in every arm, so their movement is the drift floor",
        fontsize=13,
    )

    # Grouped bars: offsets spread the arms symmetrically about each band tick.
    span = 0.8
    width = span / len(names)
    offsets = [-span / 2 + width * (i + 0.5) for i in range(len(names))]

    for col, modality in enumerate(MODALITIES):
        stats = {name: data["arms"][name][last][modality] for name in names}
        bands = stats[REFERENCE]["bands"]
        x = list(range(len(bands)))
        cmap = plt.get_cmap("viridis")
        band_colors = [cmap(i / max(len(bands) - 1, 1)) for i in range(len(bands))]

        ax = axes[0][col]
        for name, off in zip(names, offsets):
            ax.bar(
                [i + off for i in x],
                stats[name]["norms"],
                width,
                label=name,
                color=colors[name],
            )
        ax.set_xticks(x)
        ax.set_xticklabels(bands, rotation=90, fontsize=8)
        ax.set_title(f"{modality}: per-band read weight")
        ax.set_ylabel("||W[:, band]||")
        if col == 0:
            ax.legend(fontsize=8)

        # Spectral budget: share of the summed column norms, so overall weight
        # growth cancels and only the split between bands shows.
        ax = axes[1][col]
        for name, off in zip(names, offsets):
            total = sum(stats[name]["norms"])
            ax.bar(
                [i + off for i in x],
                [100 * v / total for v in stats[name]["norms"]],
                width,
                label=name,
                color=colors[name],
            )
        ax.axhline(
            100 / len(bands),
            color="k",
            ls=":",
            lw=0.8,
            label=f"equal split (1/{len(bands)})",
        )
        ax.set_xticks(x)
        ax.set_xticklabels(bands, rotation=90, fontsize=8)
        ax.set_title(f"{modality}: spectral budget (share of summed norms)")
        ax.set_ylabel("share (%)")
        if col == 0:
            ax.legend(fontsize=7)

        # Trajectories are coloured by band and styled by arm, so a band's three
        # curves stay visually tied together.
        ax = axes[2][col]
        for i, band in enumerate(bands):
            for name in names:
                ax.plot(
                    ks,
                    [data["arms"][name][s][modality]["norms"][i] for s in steps],
                    styles[name],
                    color=band_colors[i],
                    lw=1.1,
                )
            ax.annotate(
                band,
                (ks[-1], stats[REFERENCE]["norms"][i]),
                fontsize=6,
                color=band_colors[i],
                xytext=(2, 0),
                textcoords="offset points",
            )
        style_key = ", ".join(f"{styles[n]} {n}" for n in names)
        ax.set_title(f"{modality}: per-band trajectory ({style_key})", fontsize=9)
        ax.set_xlabel("training step (thousands)")
        ax.set_ylabel("||W[:, band]||")

    ax = axes[3][0]
    for name in OTHERS:
        for modality, marker in zip(MODALITIES, ("o", "s", "^")):
            ax.plot(
                ks,
                [100 * data["divergence"][name][s][modality] for s in steps],
                styles[name],
                color=colors[name],
                lw=1.3,
                marker=marker,
                markevery=max(len(ks) // 8, 1),
                markersize=4,
                label=f"{name}: {modality}",
            )
    ax.set_title(
        f"whole-matrix divergence from {REFERENCE}\n"
        "(S2/S1 curves are the noise scale for each contrast)",
        fontsize=9,
    )
    ax.set_xlabel("training step (thousands)")
    ax.set_ylabel("||W_ref - W_arm|| / ||W_ref||  (%)")
    ax.legend(fontsize=6, ncol=2)

    ax = axes[3][1]
    for modality, mcolor in zip(MODALITIES, ("#4C72B0", "#55A868", "#DD8452")):
        for name in names:
            ax.plot(
                ks,
                [data["arms"][name][s][modality]["total"] for s in steps],
                styles[name],
                color=mcolor,
                lw=1.1,
                label=f"{modality} {name}",
            )
    ax.set_title("total read weight per sensor over training", fontsize=10)
    ax.set_xlabel("training step (thousands)")
    ax.set_ylabel("Frobenius norm")
    ax.legend(fontsize=5, ncol=2)

    ax = axes[3][2]
    pos = list(range(len(MODALITIES)))
    totals = {
        name: [data["arms"][name][last][m]["total"] for m in MODALITIES]
        for name in names
    }
    for name, off in zip(names, offsets):
        ax.bar(
            [p + off for p in pos], totals[name], width, label=name, color=colors[name]
        )
    for p, modality in enumerate(MODALITIES):
        ref_total = totals[REFERENCE][p]
        peak = max(totals[name][p] for name in names)
        text = "\n".join(
            f"{name}: {100 * (totals[name][p] / ref_total - 1):+.1f}%"
            for name in OTHERS
        )
        ax.annotate(text, (p, peak), ha="center", va="bottom", fontsize=7)
    ax.set_xticks(pos)
    ax.set_xticklabels(MODALITIES, rotation=20, fontsize=8)
    ax.set_title(f"sensor-level read weight at step {last}")
    ax.set_ylabel("Frobenius norm")
    ax.margins(y=0.18)
    ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    print(f"\nwrote {path}")


def main() -> None:
    """Compare every arm's read weights against the reference."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--steps",
        default="667200",
        help="comma-separated steps, or 'stride' with --step-stride",
    )
    parser.add_argument(
        "--step-stride",
        type=int,
        default=None,
        help="sample every N steps from 0 to --max-step, plus final",
    )
    parser.add_argument("--max-step", type=int, default=667200)
    parser.add_argument("--final-step", type=int, default=667200)
    parser.add_argument(
        "--encoder", default="encoder", choices=["encoder", "target_encoder"]
    )
    parser.add_argument("--plot", default=None)
    parser.add_argument("--json", default=None)
    args = parser.parse_args()

    if args.step_stride:
        steps = list(range(0, args.max_step + 1, args.step_stride))
        if args.final_step not in steps:
            steps.append(args.final_step)
    else:
        steps = [int(s) for s in args.steps.split(",")]

    data = collect(steps, args.encoder)
    report(data, args.final_step)
    report_divergence(data, args.final_step)
    if args.json:
        with open(args.json, "w") as f:
            json.dump(data, f, indent=2)
        print(f"wrote {args.json}")
    if args.plot:
        plot(data, args.plot)


if __name__ == "__main__":
    main()
