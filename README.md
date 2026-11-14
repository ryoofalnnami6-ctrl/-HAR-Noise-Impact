# Noise Impact on Deep Learning for Activity Recognition

CAI 421 group project: MLP vs Random Forest on the 561 extracted UCI HAR features under test-time Gaussian noise.

## Repository contents
- `HAR_MLP_RF_Complete.ipynb` — main notebook (authoritative version of the code).
- `har_experiment.py` — Python export of the notebook (omits only the Colab ZIP-download cell).
- `requirements.txt` — dependency list.
- `reported_summary.csv` — four-decimal summary as reported in the paper.
- `results/` — original outputs of the executed Colab run (see below).

## Run in Google Colab
Upload `HAR_MLP_RF_Complete.ipynb` to Colab, connect a runtime, and run all cells in order.

## Local run
Use Python 3.11 or 3.12, create a virtual environment, install `pip install -r requirements.txt`, then run `python har_experiment.py`. The script creates `/content/HAR` and `/content/har_comparison`; edit these two paths for another writable directory.

## Data and protocol
UCI HAR dataset 240: https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones
CC BY 4.0. Keep the official 7,352/2,947 subject-separated train/test split. Inputs: 561 extracted features. Internal GroupShuffleSplit test_size=.20, seed 42; tune MLP duration on validation Macro F1. Refit scaler/models on all official training data. MLP: 561/256/128/6 with ReLU, dropout .3 and Softmax. RF: 100 trees. Noise sigma: 0, .05, .10, .20, .40 on standardized test features. Seeds 42–46; same noisy matrices for both models. No test-driven tuning.

## Original results (`results/`)
Outputs of the executed run reported in the paper:
- `all_runs.csv` — accuracy and Macro F1 for every seed, model and noise level; `summary.csv` — mean ± SD across seeds.
- `per_class.csv` — per-class precision, recall, F1 and support.
- `confusion_<model>_seed_<s>_sigma_<σ>.csv` — confusion matrices (rows = true, columns = predicted).
- `training.csv` — selected MLP epochs and training times; `validation_seed_<s>.csv` — validation histories.
- `mlp_seed_<s>.keras` — trained MLPs; `scaler.npz` — scaler statistics; `config.json` — run configuration.
- `noise_comparison.png` — Fig. 2 of the paper.

All tables in the paper (Tables III and V, confusion counts, timing) can be recomputed from these files.

## Reproducibility limits
`requirements.txt` declares compatible ranges, not a lockfile from the original Colab session (TensorFlow 2.21.0, CPU runtime). TensorFlow or device changes can slightly affect a rerun's outputs; the files in `results/` are the reference values.
