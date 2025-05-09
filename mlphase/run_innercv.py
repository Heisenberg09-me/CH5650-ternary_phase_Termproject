import os
import sys
import pickle
from types import SimpleNamespace

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# ✅ Fix import paths
sys.path.append("/content/ml-ternary-phase/mlphase/model")
sys.path.append("/content/ml-ternary-phase/mlphase/data")

# ✅ Imports
from train import train_cls_reg
from model import ChainSoftmax
from loss import wu_loss
from split import split_data

# ✅ Hardcoded paths
PICKLE_DATA_PATH = "/content/ml-ternary-phase/data_clean.pickle"
PICKLE_INNER_PATH = "/content/ml-ternary-phase/result_pickle"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("✅ Running inner cross-validation (Colab version)...")

for fold_idx in range(1,6):
    print(f"\n▶️ Starting fold {fold_idx}...")

    # Dummy args
    args = SimpleNamespace()
    args.fold_idx = fold_idx
    args.data_path = PICKLE_DATA_PATH
    args.result_path = PICKLE_INNER_PATH
    args.random_seed = 42
    args.mask = 0
    args.use_split = 0
    args.use_delta_mu = 0
    args.use_free_energy = 0
    args.model_type = 'base'

    # ✅ Load dataset using split_data
    dataset = split_data(
        fold=fold_idx,
        sub_fold=0,
        n_folds=5,
        sample_ratio=0.1,
        random_seed=args.random_seed,
        DATA_DIR= "/content/ml-ternary-phase/data"
    )
    data_loader = DataLoader(dataset["train"], batch_size=5000, shuffle=True)

    # ✅ Model, optimizer, loss
    model = ChainSoftmax(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    cls_criterion = nn.CrossEntropyLoss()
    reg_crit = wu_loss
    reg_criterion_size = 9

    # ✅ Train
    result = train_cls_reg(
        model, data_loader, optimizer,
        cls_criterion, reg_crit, reg_criterion_size, DEVICE
    )

    # ✅ Save
    os.makedirs(PICKLE_INNER_PATH, exist_ok=True)
    with open(f"{PICKLE_INNER_PATH}/inner_result_fold{fold_idx}.pickle", "wb") as f:
        pickle.dump(result, f)

    print(f"✅ Fold {fold_idx} completed and saved.")
