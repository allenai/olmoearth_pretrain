"""Find where the quantized-FOV flex path produces NaN (GPU), vs the dense reference."""

import sys

import torch

sys.path.insert(0, "scripts/tools")
from lighthouse_aoi_inference import (  # noqa: E402
    WindowSamples,
    _batched,
    _crop,
    build_window_dataset,
)

from olmoearth_pretrain.model_loader import load_pretrain_checkpoint  # noqa: E402
from olmoearth_pretrain.nn.lighthouse_rc import RCLighthouseSettings  # noqa: E402


def main() -> None:
    """Flex vs dense per quantum on a small real crop, with first-NaN hooks."""
    ckpt, dataset = sys.argv[1], sys.argv[2]
    dev = torch.device("cuda")
    enc = load_pretrain_checkpoint(ckpt, device=dev).encoder.eval()
    _, sample, _ = WindowSamples(
        build_window_dataset(dataset, "predict", ["sundarbans_tidal"])
    )[0]
    full = _batched(sample, dev)
    crop = _crop(full, slice(300, 364), slice(300, 364))
    seen: list[str] = []

    def hook(name):
        def f(_m, _i, out):
            if torch.isnan(out).any() and not seen:
                seen.append(name)
                print(
                    f"   first NaN after {name}: {int(torch.isnan(out).any(-1).sum())} rows of {out.shape[1]}",
                    flush=True,
                )

        return f

    hs = [
        b.attn.proj.register_forward_hook(hook(f"vit{i}.attn.proj"))
        for i, b in enumerate(enc.blocks)
    ]
    hs += [
        b.mlp.register_forward_hook(hook(f"vit{i}.mlp"))
        for i, b in enumerate(enc.blocks)
    ]
    p = enc.perceiver
    hs += [
        b.attn.proj.register_forward_hook(hook(f"read{i}.attn.proj"))
        for i, b in enumerate(p.read_blocks)
    ]
    hs += [
        b.attn.proj.register_forward_hook(hook(f"lat{i}.attn.proj"))
        for i, b in enumerate(p.latent_blocks)
    ]
    for quantum, column, dtype in (
        (1, False, "int64"),
        (1, True, "int64"),
        (4, False, "int64"),
        (8, False, "int64"),
        (1, True, "int32"),
        (1, False, "int32"),
    ):
        outs = {}
        tag = f"q{quantum} column={column} {dtype}"
        for dense in (True, False):
            seen.clear()
            enc.lighthouse = RCLighthouseSettings(
                16,
                fov_quantum=quantum,
                dense=dense,
                column_mask=column,
                code_dtype=dtype,
            )
            try:
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    o = enc(crop, patch_size=1, input_res=10, fast_pass=False)
                torch.cuda.synchronize()
            except Exception as e:  # noqa: BLE001
                print(f"{tag} dense={dense}: ERROR {type(e).__name__}", flush=True)
                return
            finally:
                enc.lighthouse = None
            r = o["student_registers"][0].float()
            outs[dense] = r
            print(
                f"{tag} dense={dense}: NaN px {int(torch.isnan(r).any(-1).sum())}",
                flush=True,
            )
        cos = torch.nn.functional.cosine_similarity(outs[False], outs[True], dim=-1)
        print(
            f"{tag} flex vs dense cos min {cos.nan_to_num(-9).min().item():.6f} mean {cos.nan_to_num(-9).mean().item():.6f}",
            flush=True,
        )
    for h in hs:
        h.remove()


if __name__ == "__main__":
    main()
