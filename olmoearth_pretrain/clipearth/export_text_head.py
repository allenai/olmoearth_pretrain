"""Export a native aligner's text head for in-browser text search.

Writes the JSON the Studio ``text-search`` lab loads: which transformers.js model to embed
the query with, its prompt prefix and output, and the dense layers that map that
embedding into OlmoEarth's native embedding space (row-major float16, base64).

Run:
  python -m olmoearth_pretrain.clipearth.export_text_head
      --aligner <run>/aligner.pt --browser_model <hf repo> --out <file>.json
"""

from __future__ import annotations

import argparse
import base64
import json

import numpy as np
import torch


def encode(array: torch.Tensor) -> str:
    """Base64 of the array's little-endian float16 bytes, row-major."""
    data = array.detach().cpu().numpy().astype("<f2")
    return base64.b64encode(np.ascontiguousarray(data).tobytes()).decode()


def encode32(array: np.ndarray) -> str:
    """Base64 of the array's little-endian float32 bytes, row-major."""
    return base64.b64encode(np.ascontiguousarray(array, dtype="<f4").tobytes()).decode()


def main() -> None:
    """Write the text head JSON."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--aligner", required=True)
    parser.add_argument("--browser_model", required=True)
    parser.add_argument("--dtype", default="q4")
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--calibration",
        action="append",
        default=[],
        help="<dims>=<calibrate.py output>; adds match cutoffs for that width",
    )
    parser.add_argument(
        "--background",
        help="npz of mean_<dims>, cov_<dims>; needed for z-score calibrations",
    )
    args = parser.parse_args()

    ckpt = torch.load(args.aligner, map_location="cpu", weights_only=False)
    train_args = ckpt["args"]
    if not train_args.get("native"):
        raise ValueError("only native aligners score text against stored embeddings")
    linears = [
        (key.rsplit(".", 1)[0], value)
        for key, value in ckpt["state_dict"].items()
        if key.startswith("text.") and key.endswith("weight")
    ]
    layers = [
        {
            "inFeatures": weight.shape[1],
            "outFeatures": weight.shape[0],
            "weight": encode(weight),
            "bias": encode(ckpt["state_dict"][f"{name}.bias"]),
        }
        for name, weight in linears
    ]
    onnx = train_args["text_model"].startswith("onnx:")
    head = {
        "textModel": args.browser_model,
        "dtype": args.dtype,
        "prefix": train_args.get("text_prefix", ""),
        "output": "sentence_embedding" if onnx else "cls",
        "modelFileName": train_args.get("text_onnx_file", "model.onnx")
        .removesuffix(".onnx")
        .removesuffix(f"_{args.dtype}"),
        "textDims": train_args.get("text_dims"),
        "activation": "gelu" if len(layers) > 1 else None,
        "layers": layers,
        "matryoshkaDims": [128]
        + [int(d) for d in train_args.get("matryoshka", "").split(",") if d],
    }
    calibration = {}
    background = {}
    stats = np.load(args.background) if args.background else None
    for item in args.calibration:
        dims, path = item.split("=", 1)
        report = json.load(open(path))
        score = report.get("score", "cosine")
        calibration[dims] = {
            "score": score,
            "cutoffs": [
                {"falsePositiveRate": float(rate), "value": cut["cosine"]}
                for rate, cut in report["unseen_at_false_positive_rate"].items()
            ],
        }
        if score == "z":
            if stats is None:
                raise ValueError("z-score calibrations need --background")
            background[dims] = {
                "mean": encode32(stats[f"mean_{dims}"]),
                "cov": encode32(stats[f"cov_{dims}"]),
            }
    if calibration:
        head["calibration"] = calibration
    if background:
        head["background"] = background
    with open(args.out, "w") as f:
        json.dump(head, f, separators=(",", ":"))


if __name__ == "__main__":
    main()
