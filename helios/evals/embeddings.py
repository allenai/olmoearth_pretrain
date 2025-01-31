import torch


def get_embeddings(data_loader, model, device):
    embeddings = []
    labels = []

    model = model.eval()
    with torch.no_grad():
        for batch in data_loader:
            batch_labels = batch.pop("target")
            if "s1" in batch:
                batch["s1"] = batch["s1"].to(device).to(torch.bfloat16)
            if "s2" in batch:
                batch["s2"] = batch["s2"].to(device).to(torch.bfloat16)
            if "months" in batch:
                batch["months"] = batch["months"].to(device).long()

            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                batch_embeddings = model(**batch)  # (bsz, dim)

            embeddings.append(batch_embeddings.to(torch.bfloat16).cpu())
            labels.append(batch_labels)

    embeddings = torch.cat(embeddings, dim=0)  # (N, dim)
    labels = torch.cat(labels, dim=0)  # (N)

    return embeddings, labels
