import os
import numpy as np
from timeit import default_timer as timer
from sklearn.model_selection import StratifiedKFold
from mlphase.data import load_data
from mlphase.data.data import CustomDataset


def split_data(fold, sub_fold, n_folds=5, sample_ratio=0.1, random_seed=42, DATA_DIR="/scratch/gpfs/sj0161/mlphase/data/"):
    t0 = timer()

    # ✅ Step 1: Load data
    file = "/content/ml-ternary-phase/data_clean.pickle"
    x, yc, yr, phase_idx, n_phase, phase_type = load_data(file)

    # ✅ Step 2: Outer 5-fold CV on phase diagrams
    uni_p_id = np.unique(phase_idx)
    skf = StratifiedKFold(n_splits=n_folds, random_state=random_seed, shuffle=True)

    fold_found = False  # Track if we found the requested fold

    for fold_temp, (train_idx, test_idx) in enumerate(skf.split(uni_p_id, phase_type)):
        if fold == fold_temp + 1:
            print(f"# Fold: {fold}")
            uni_p_id_train = uni_p_id[train_idx]
            uni_p_id_test = uni_p_id[test_idx]
            phase_type_train = phase_type[train_idx]
            fold_found = True
            break

    # ✅ Handle invalid fold index
    if not fold_found:
        raise ValueError(f"Fold {fold} not found. Valid values are 1 to {n_folds}")

    # ✅ Step 3: Mask data based on test IDs
    mask_test = np.isin(phase_idx, uni_p_id_test)
    test_idx = np.where(mask_test)[0]

    x_test = x[test_idx]
    yr_test = yr[test_idx]
    yc_test = yc[test_idx]
    phase_idx_test = phase_idx[test_idx]

    # ✅ Step 4: Inner CV on training data
    skf2 = StratifiedKFold(n_splits=n_folds, random_state=random_seed, shuffle=True)
    for fold_temp, (train_idx, val_idx) in enumerate(skf2.split(uni_p_id_train, phase_type_train)):
        if sub_fold == fold_temp:
            print(f"# Sub-Fold: {sub_fold}")
            uni_p_id_train_inner = uni_p_id_train[train_idx]
            uni_p_id_val = uni_p_id_train[val_idx]
            break

    # ✅ Final splits for train and val
    mask_train = np.isin(phase_idx, uni_p_id_train_inner)
    mask_val = np.isin(phase_idx, uni_p_id_val)

    train_idx = np.where(mask_train)[0]
    val_idx = np.where(mask_val)[0]

    x_train = x[train_idx]
    yr_train = yr[train_idx]
    yc_train = yc[train_idx]

    x_val = x[val_idx]
    yr_val = yr[val_idx]
    yc_val = yc[val_idx]

    t1 = timer()
    print(f"⏱ Data split done in {t1 - t0:.2f} seconds")

    # ✅ Return all sets as a dictionary of CustomDataset objects
    return {
        "train": CustomDataset(x_train, yc_train, yr_train),
        "val": CustomDataset(x_val, yc_val, yr_val),
        "test": CustomDataset(x_test, yc_test, yr_test),
    }
