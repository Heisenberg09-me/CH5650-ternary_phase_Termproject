import os
import pickle
import numpy as np
import torch

from types import SimpleNamespace
from mlphase.data.split import split_data
from mlphase.data.data import CustomDataset
from mlphase.model.train import train_cls_reg
from mlphase.model.model import ChainSoftmax
from mlphase.model.loss import wu_loss
import torch.nn as nn
from torch.utils.data import DataLoader

# ✅ Hardcoded paths
DATA_DIR = "/content/ml-ternary-phase/data"
RESULT_DIR = "/content/ml-ternary-phase/result_pickle"

os.makedirs(RESULT_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main(fold_idx):
    # === Configuration ===
    args = SimpleNamespace()
    args.fold_idx = fold_idx
    args.sub_fold = 0  # not used here but required
    args.data_path = os.path.join(DATA_DIR, "data_clean.pickle")
    args.result_path = RESULT_DIR
    args.random_seed = 42
    args.mask = 0
    args.use_split = 0
    args.use_delta_mu = 0
    args.use_free_energy = 0
    args.model_type = "base"

    # === Load split data ===
    dataset = split_data(
        fold=fold_idx,
        sub_fold=0,
        n_folds=5,
        sample_ratio=0.1,
        random_seed=args.random_seed,
        DATA_DIR=DATA_DIR,
    )

    train_loader = DataLoader(dataset["train"], batch_size=5000, shuffle=True)
    val_loader = DataLoader(dataset["val"], batch_size=5000)
    test_loader = DataLoader(dataset["test"], batch_size=5000)

    # === Build model and optimizer ===
    model = ChainSoftmax(DEVICE).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    cls_criterion = nn.CrossEntropyLoss()
    reg_crit = wu_loss
    reg_criterion_size = 9

    # === Train model ===
    result = train_cls_reg(
        model,
        train_loader,
        optimizer,
        cls_criterion,
        reg_crit,
        reg_criterion_size,
        DEVICE,
    )

    # === Save result ===
    result_path = os.path.join(RESULT_DIR, f"outer_result_fold{fold_idx}.pickle")
    with open(result_path, "wb") as f:
        pickle.dump(result, f)

    print(f"✅ Saved outer fold {fold_idx} result to {result_path}")


# ✅ Replace SLURM-based execution with manual loop over folds
if __name__ == "__main__":
    for idx in range(1, 6):  # Fold indices: 1 to 5
        print(f"\n==============================")
        print(f"▶️ Starting Outer Fold {idx}")
        print(f"==============================\n")
        main(idx)
