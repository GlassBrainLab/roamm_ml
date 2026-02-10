#!/usr/bin/env python

import os
import re
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    balanced_accuracy_score,
    cohen_kappa_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
)
from datasets import Dataset, DatasetDict
from transformers import AutoModel, set_seed
from functools import partial
from tqdm.auto import tqdm

# ==========================
# Config
# ==========================
data_root = "/gpfs1/pi/djangraw/mindless_reading/data"
sfreq = 200
window_seconds = 2
window_size = int(sfreq * window_seconds)

# model_size = "base"
model_size = "large"
if model_size == "base":
    hidden_dim = 512
elif model_size == "large":
    hidden_dim = 1250

batch_size = 64
n_epochs = 20
lr = 1e-3
random_seed = 42

set_seed(random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ==========================
# Collect subjects
# ==========================
all_subjects = sorted(
    d for d in os.listdir(data_root)
    if d.startswith("s") and os.path.isdir(os.path.join(data_root, d))
)

X_all, y_all = [], []
chan_list = None

# ==========================
# Load data
# ==========================
for subject_id in all_subjects:
    subject_dir = os.path.join(
        data_root,
        subject_id,
        "ml_data",
        f"{256 * window_seconds}window_datasets",
    )

    if not os.path.isdir(subject_dir):
        continue

    # Load channel list once
    if chan_list is None:
        col_file = os.path.join(subject_dir, f"{subject_id}_col_names.npy")
        col_names = np.load(col_file, allow_pickle=True)
        chan_list = col_names[:64].tolist()
        chan_list = ["AFz" if ch == "Afz" else ch for ch in chan_list]

    data_file = os.path.join(subject_dir, f"{subject_id}_{window_size}windowed_eeg_data.npy")
    labels_file = os.path.join(subject_dir, f"{subject_id}_{256 * window_seconds}windowed_labels.npy")

    if not (os.path.exists(data_file) and os.path.exists(labels_file)):
        continue

    data = np.load(data_file)
    labels = np.load(labels_file)

    X_all.append(data)
    y_all.append(labels)

X_all = np.concatenate(X_all, axis=0)
y_all = np.concatenate(y_all, axis=0)

print("Final dataset shape:", X_all.shape, y_all.shape)

# ==========================
# Train / Val / Test split
# ==========================
X_train, X_tmp, y_train, y_tmp = train_test_split(
    X_all, y_all, test_size=0.2, stratify=y_all, random_state=random_seed
)
X_val, X_test, y_val, y_test = train_test_split(
    X_tmp, y_tmp, test_size=0.5, stratify=y_tmp, random_state=random_seed
)

dataset = DatasetDict({
    "train": Dataset.from_dict({"data": list(X_train), "labels": list(y_train)}),
    "val": Dataset.from_dict({"data": list(X_val), "labels": list(y_val)}),
    "test": Dataset.from_dict({"data": list(X_test), "labels": list(y_test)}),
})

dataset.set_format("torch", columns=["data", "labels"])
print(dataset)

# ==========================
# Load model & positions
# ==========================
pos_bank = AutoModel.from_pretrained(
    "brain-bzh/reve-positions", trust_remote_code=True
)
model = AutoModel.from_pretrained(
    f"brain-bzh/reve-{model_size}", trust_remote_code=True
)

positions = pos_bank(chan_list).to(device)

# Replace final layer
ch_num = len(chan_list)
dim = ch_num * window_seconds * hidden_dim

model.final_layer = torch.nn.Sequential(
    torch.nn.Flatten(),
    torch.nn.RMSNorm(dim),
    torch.nn.Dropout(0.1),
    torch.nn.Linear(dim, 2),
)

model.to(device)

# ==========================
# Dataloaders
# ==========================
def collate(batch, positions):
    x_data = torch.stack([x["data"] for x in batch])
    y_label = torch.tensor([x["labels"] for x in batch])
    pos = positions.repeat(len(batch), 1, 1)
    return {"sample": x_data, "label": y_label.long(), "pos": pos}

collate_fn = partial(collate, positions=positions)

train_loader = torch.utils.data.DataLoader(
    dataset["train"], batch_size=batch_size, shuffle=True, collate_fn=collate_fn
)
val_loader = torch.utils.data.DataLoader(
    dataset["val"], batch_size=batch_size, shuffle=False, collate_fn=collate_fn
)
test_loader = torch.utils.data.DataLoader(
    dataset["test"], batch_size=batch_size, shuffle=False, collate_fn=collate_fn
)

# ==========================
# Training utils
# ==========================
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.final_layer.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="max", factor=0.5, patience=3
)

def train_one_epoch(model, loader):
    model.train()
    pbar = tqdm(loader, desc="Training")
    for batch in pbar:
        x, y, pos = (
            batch["sample"].to(device),
            batch["label"].to(device),
            batch["pos"].to(device),
        )
        optimizer.zero_grad()
        with torch.amp.autocast(
            device_type="cuda" if torch.cuda.is_available() else "cpu",
            dtype=torch.float16,
        ):
            out = model(x, pos)
            loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        pbar.set_postfix(loss=loss.item())

def eval_model(model, loader):
    model.eval()
    y_true, y_pred, y_prob = [], [], []

    with torch.inference_mode():
        for batch in tqdm(loader, desc="Evaluating"):
            x, y, pos = (
                batch["sample"].to(device),
                batch["label"].to(device),
                batch["pos"].to(device),
            )
            with torch.amp.autocast(
                device_type="cuda" if torch.cuda.is_available() else "cpu",
                dtype=torch.float16,
            ):
                out = model(x, pos)

            pred = torch.argmax(out, dim=1)
            y_true.append(y)
            y_pred.append(pred)
            y_prob.append(out)

    y_true = torch.cat(y_true).cpu().numpy()
    y_pred = torch.cat(y_pred).cpu().numpy()
    y_prob = torch.cat(y_prob).cpu().numpy()

    return {
        "acc": (y_true == y_pred).mean(),
        "balanced_acc": balanced_accuracy_score(y_true, y_pred),
        "kappa": cohen_kappa_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred, average="weighted"),
        "auroc": roc_auc_score(y_true, y_prob[:, 1]),
        "auc_pr": average_precision_score(y_true, y_prob[:, 1]),
    }

# ==========================
# Training loop
# ==========================
best_val = 0.0
best_state = None

for epoch in range(n_epochs):
    print(f"\nEpoch {epoch + 1}/{n_epochs}")
    train_one_epoch(model, train_loader)
    metrics = eval_model(model, val_loader)

    if metrics["balanced_acc"] > best_val:
        best_val = metrics["balanced_acc"]
        best_state = model.final_layer.state_dict()

    scheduler.step(metrics["balanced_acc"])
    print("Val metrics:", metrics)

# ==========================
# Test
# ==========================
model.final_layer.load_state_dict(best_state)
test_metrics = eval_model(model, test_loader)

print("\n==== TEST RESULTS ====")
for k, v in test_metrics.items():
    print(f"{k}: {v:.4f}")
