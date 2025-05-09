# ✅ STEP 2: Update run_opt.py
# File: /content/ml-ternary-phase/mlphase/run_opt.py

import os
import numpy as np
import matplotlib.pyplot as plt
from mlphase.analysis.post_stat import average_metrics_across_folds

RESULT_PATH = "/content/ml-ternary-phase/result_pickle"

def main():
    print("📊 Optimizing Results from Outer CV...\n")

    # Load average + raw values
    avg_metrics, raw_scalar_values, raw_vector_values = average_metrics_across_folds(return_raw=True)

    # Print with stats
    print("\n✅ Optimization complete.")
    print("📈 Averaged Performance (with ±95% CI):")

    scalar_std = np.std(raw_scalar_values)
    scalar_ci = 1.96 * scalar_std / np.sqrt(len(raw_scalar_values))
    print(f"• overall_mae: {avg_metrics['overall_mae']:.4f} ± {scalar_ci:.4f}")

    print("• per_property_mae:")
    vector_std = np.std(np.stack(raw_vector_values), axis=0)
    vector_ci = 1.96 * vector_std / np.sqrt(len(raw_vector_values))
    for i, (mean, ci) in enumerate(zip(avg_metrics["per_property_mae"], vector_ci)):
        print(f"   - Property {i+1}: {mean:.4f} ± {ci:.4f}")

    # Save to file
    summary_path = os.path.join(RESULT_PATH, "averaged_metrics.txt")
    with open(summary_path, "w") as f:
        f.write(f"overall_mae: {avg_metrics['overall_mae']:.4f} ± {scalar_ci:.4f}\n")
        f.write("per_property_mae:\n")
        for i, (mean, ci) in enumerate(zip(avg_metrics["per_property_mae"], vector_ci)):
            f.write(f"  - Property {i+1}: {mean:.4f} ± {ci:.4f}\n")

    # ✅ Generate bar plot with error bars
    property_ids = [f"P{i+1}" for i in range(len(avg_metrics["per_property_mae"]))]
    means = avg_metrics["per_property_mae"]
    errors = vector_ci

    plt.figure(figsize=(10, 6))
    plt.bar(property_ids, means, yerr=errors, capsize=5, color='skyblue', edgecolor='black')
    plt.ylabel("Mean Absolute Error (MAE)")
    plt.xlabel("Property")
    plt.title("Per-Property MAE with 95% Confidence Intervals")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()

    plot_path = os.path.join(RESULT_PATH, "per_property_mae_plot.png")
    plt.savefig(plot_path)
    print(f"\n📊 Saved plot to: {plot_path}")
    print(f"📁 Saved summary to: {summary_path}")

if __name__ == "__main__":
    main()
