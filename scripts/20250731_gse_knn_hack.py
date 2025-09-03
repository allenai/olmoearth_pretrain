"""GSE embeddings on EuroSat with KNN.

Use the exported GSE windows for Eurosat and run KNN,
to get somewhat comparable results to our exising KNN
runs.

This achieves an accuracy score of 0.9447852760736196.
"""

import json

import numpy as np
import rioxarray
import torch
from sklearn.metrics import accuracy_score
from torch import nn
from tqdm import tqdm
from upath import UPath

EUROSAT_PATH = UPath(
    "/weka/dfive-default/rslearn-eai/datasets/eurosat/rslearn_dataset/windows/default"
)
BAND_HASH = "610a2ee7942f0f42b7a9bddea505048ac5b6d68739b0d848c0b30e36b201d01a"


SPLIT_TO_INT = {"train": 0, "val": 1}


def _run_knn_for_k(
    train_embeddings: torch.Tensor,
    train_labels: torch.Tensor,
    test_embeddings: torch.Tensor,
    num_classes: int,
    k: int,
    device: torch.device,
    skip_idx: bool,
) -> torch.Tensor:
    train_embeddings = train_embeddings.to(device)
    test_embeddings = test_embeddings.to(device)
    train_labels = train_labels.to(device)
    cos = nn.CosineSimilarity(dim=-1)
    all_preds = []
    for idx in range(test_embeddings.shape[0]):
        test_embedding = test_embeddings[idx].unsqueeze(dim=0)
        test_embedding = (
            test_embeddings[idx].unsqueeze(dim=0).repeat(train_embeddings.shape[0], 1)
        )
        sims = cos(test_embedding, train_embeddings)
        top_k = torch.topk(sims, k=k)
        if skip_idx:
            top_k_values = top_k.values[1:]
            top_k_indices = top_k.indices[1:]
        else:
            top_k_values = top_k.values
            top_k_indices = top_k.indices

        fetched_labels = train_labels[top_k_indices]
        fetched_onehots = nn.functional.one_hot(fetched_labels, num_classes=num_classes)
        distances = top_k_values.clone().div_(0.07).exp_()
        weighted_sum_onehots = (distances.unsqueeze(dim=1) * fetched_onehots).sum(dim=0)
        prediction = torch.argmax(weighted_sum_onehots)
        all_preds.append(prediction)

    return torch.LongTensor(all_preds).cpu()


def _load_images_and_labels_and_split_from_folder(
    path: UPath,
) -> tuple[np.ndarray, int, str] | None:
    tif_file = path / f"layers/gse/{BAND_HASH}/geotiff.tif"
    if not tif_file.exists():
        return None
    else:
        x = rioxarray.open_rasterio(tif_file).values

        with (path / "metadata.json").open("r") as f:
            metadata = json.load(f)
            label = metadata["options"]["category"]
            split = metadata["options"]["split"]

    # take the mean along h, w
    return np.mean(x, axis=(1, 2)), SPLIT_TO_INT[split], label


def run_knn():
    """Run knn."""
    all_folders = [f for f in EUROSAT_PATH.glob("*") if f.is_dir()]
    xs, splits, labels = [], [], []
    for folder in tqdm(all_folders):
        output = _load_images_and_labels_and_split_from_folder(folder)
        if output:
            x, split, label = output
            xs.append(x)
            splits.append(split)
            labels.append(label)

    # concat
    x_t = torch.from_numpy(np.stack(xs, axis=0))
    splits = torch.tensor(splits)
    unique_classes = set(labels)
    classes_to_int = {val: idx for idx, val in enumerate(unique_classes)}
    print(classes_to_int)
    y = torch.tensor([classes_to_int[c] for c in labels])

    x_train = x_t[splits == 0]
    x_val = x_t[splits == 1]
    y_train = y[splits == 0]
    y_val = y[splits == 1]

    preds = _run_knn_for_k(
        x_train,
        y_train,
        x_val,
        num_classes=len(unique_classes),
        k=20,
        device=x_t.device,
        skip_idx=False,
    )
    print(accuracy_score(y_true=y_val, y_pred=preds))


if __name__ == "__main__":
    run_knn()
