# ✅ STEP 1: Update post_stat.py
# File: /content/ml-ternary-phase/mlphase/analysis/post_stat.py

import os
import glob
import pickle
import numpy as np
import torch

# Find all outer result pickles
tuple_files = sorted(glob.glob("/content/ml-ternary-phase/result_pickle/outer_result_fold*.pickle"))

def average_metrics_across_folds(return_raw=False):
    results = []

    for fpath in tuple_files:
        print(f"📂 Loading: {fpath}")
        with open(fpath, "rb") as handle:
            result = pickle.load(handle)
            results.append(result)

    if not results:
        raise ValueError("❌ No result files found to average.")

    scalar_values = [res[0].item() for res in results]                      # overall MAE
    vector_values = [res[1].numpy() for res in results]                    # per-property MAE (9-d vector)

    avg_scalar = np.mean(scalar_values)
    avg_vector = np.mean(np.stack(vector_values), axis=0)

    if not return_raw:
        print("\n📊 Averaged Metrics Across Outer Folds:")
        print(f"• Overall MAE: {avg_scalar:.4f}")
        for i, val in enumerate(avg_vector):
            print(f"  - Property {i+1} MAE: {val:.4f}")

    output = {
        "overall_mae": avg_scalar,
        "per_property_mae": avg_vector
    }

    if return_raw:
        return output, scalar_values, vector_values
    else:
        return output
