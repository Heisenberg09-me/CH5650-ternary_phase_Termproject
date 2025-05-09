# CH-5650 - Molecular Data Science and Informatics
# ML-Ternary-Phase Term paper -CH24M029



## 📋 Overview

This repository reimplements and enhances the methods from:

***"Accelerating Multicomponent Phase-Coexistence Calculations with Physics-Informed Neural Networks"***

The core purpose is to:
- ✅ Train machine learning models for phase classification and property regression
- ✅ Evaluate them using nested cross-validation (inner + outer)
- ✅ Generate scientific plots for result interpretation
- ✅ Run entirely in Google Colab or locally via Git

---

## 🔬 Scientific Background

This project uses machine learning to accelerate thermodynamic phase coexistence calculations, which are typically computationally expensive. By applying physics-informed neural networks, we can predict:

- Phase classifications (one-phase vs multi-phase regions)
- Thermodynamic properties in multicomponent systems
- Phase boundaries in ternary mixtures

The approach significantly reduces computational time while maintaining high accuracy compared to traditional numerical methods.

---

## 📦 Installation & Setup

### 1. 📥 Clone the Repository

```bash
git clone https://github.com/Heisenberg09-me/ml-ternary-phase.git
cd ml-ternary-phase
```

### 2. ⚙️ Install the Dependencies

**If using locally:**
```bash
pip install -e .
```

**If using Google Colab:**
```python
%cd /content/ml-ternary-phase
!pip install -e .
```

### 3. 📂 Upload the Dataset

Upload your `data_clean.pickle` file to:
```
/content/ml-ternary-phase/data_clean.pickle
```

> ⚠️ **Important**: Make sure this file exists before running any training scripts.

---

## 🚀 Running the Pipeline

### 4. 🧪 Run Model Training and Evaluation

**Inner CV (5-fold):**
```bash
python -m mlphase.run_innercv
```

**Outer CV (5-fold):**
```bash
python -m mlphase.run_outercv
```

**Post-Processing and Metrics Aggregation:**
```bash
python -m mlphase.run_opt
```

### 5. 📊 Generate Plots

Run the script to generate all figures and save them in `/figure`:
```bash
python -m mlphase.generate_all_figures
```

---

## 📁 Project Structure

```
ml-ternary-phase/
├── mlphase/              # Core ML pipeline
│   ├── analysis/         # Analysis utilities
│   ├── data/             # Data handling modules
│   ├── model/            # Model definitions
│   ├── plot/             # Plotting scripts
│   ├── run_innercv.py    # Run inner cross-validation
│   ├── run_outercv.py    # Run outer cross-validation
│   ├── run_opt.py        # Aggregate metrics
│   ├── generate_all_figures.py # Plot generation script
├── result_pickle/        # Output from CV runs
├── figure/               # Generated plots
├── notebook/             # Jupyter notebooks
├── setup.py              # Package installation
└── README.md             # This file
```

---

## 📊 Results

After running the full pipeline, you'll find:

1. **Model Performance Metrics**: Accuracy, F1-score, MSE, etc. in `result_pickle/`
2. **Visualization Plots**: 
   - Ternary phase diagrams
   - Learning curves
   - Feature importance plots
   - Prediction error analysis
3. **Optimized Hyperparameters**: Best configuration for each model type


## 🧠 Tips for Optimal Use

- Always verify `data_clean.pickle` exists before running training
- For Colab: re-upload the file every session or load from Google Drive
- Use `git status` and `git log` to track changes before pushing
- Consider using a virtual environment for local development
- For large datasets, adjust batch sizes in configuration files
- Export your best models using the built-in save functionality

---

