# Noise Impact on Deep Learning for Activity Recognition

## Run in Google Colab
Upload `HAR_MLP_RF_Complete.ipynb` to Colab, connect a runtime, and run all cells in order. The original reported experiment has already been executed by the group. This archive contains source and a four-decimal transcription of the supplied summary, not the original trained models or raw predictions.

## Local run
Use Python 3.11 or 3.12, create a virtual environment, install `pip install -r requirements.txt`, then run `python har_experiment.py`. The script creates `/content/HAR` and `/content/har_comparison`; edit these two paths for another writable directory. The Python export omits only the Colab ZIP-download cell. The notebook remains authoritative.

## Data and protocol
UCI HAR dataset 240: https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones
CC BY 4.0. Keep the official 7,352/2,947 subject-separated train/test split. Inputs: 561 extracted features. Internal GroupShuffleSplit test_size=.20, seed42; tune MLP duration on validation Macro F1. Refit scaler/models on all official training data. MLP: 561/256/128/6 with ReLU, dropout.3 and Softmax. RF:100 trees. Noise sigma:0,.05,.10,.20,.40 on standardized test features. Seeds42–46; same noisy matrices for both models. No test-driven tuning.

## Outputs
`summary.csv`, `all_runs.csv`, `per_class.csv`, per-seed confusion CSVs, `training.csv`, validation histories, five saved MLPs, scaler statistics, `config.json`, and `noise_comparison.png`. Results are archived by the final Colab cell. Original raw output ZIP should be retained separately by the group.

## Reproducibility limits
requirements.txt declares compatible ranges, not a recovered lockfile from the original Colab session. TensorFlow/device changes can affect outputs. Exact original environment and timing data must be retained from the executed run. Reported summary values are rounded to four decimals. Do not manufacture per-class scores, hardware details or training durations from this summary.

## Publication
Upload this folder to a group-owned repository and submit its actual URL. No repository has been published by this archive.
