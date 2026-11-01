from pathlib import Path
import io
import json
import time
import zipfile
import urllib.request
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupShuffleSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

SEEDS = [42, 43, 44, 45, 46]
NOISE_LEVELS = [0.0, 0.05, 0.10, 0.20, 0.40]
MAX_EPOCHS = 100
PATIENCE = 10
BATCH_SIZE = 64
OUT = Path('/content/har_comparison')
OUT.mkdir(parents=True, exist_ok=True)
tf.config.experimental.enable_op_determinism()
print('TensorFlow:', tf.__version__)

DATA_ROOT = Path('/content/HAR')
found = list(DATA_ROOT.rglob('X_train.txt')) if DATA_ROOT.exists() else []
if not found:
    url = 'https://archive.ics.uci.edu/static/public/240/human+activity+recognition+using+smartphones.zip'
    print('Downloading UCI HAR...')
    payload = urllib.request.urlopen(url, timeout=180).read()
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        archive.extractall(DATA_ROOT)
    for inner in DATA_ROOT.rglob('*.zip'):
        with zipfile.ZipFile(inner) as archive:
            archive.extractall(inner.parent)
    found = list(DATA_ROOT.rglob('X_train.txt'))
if not found:
    raise FileNotFoundError('Cannot find the extracted UCI HAR training file.')
DATA = found[0].parent.parent

def read_values(relative):
    return pd.read_csv(DATA / relative, sep=r'\s+', header=None).to_numpy()

if not all(name in globals() for name in ['X_train', 'X_test', 'y_train', 'y_test']):
    X_train = read_values('train/X_train.txt')
    X_test = read_values('test/X_test.txt')
    y_train = read_values('train/y_train.txt').ravel()
    y_test = read_values('test/y_test.txt').ravel()

train_x = np.asarray(X_train, dtype=np.float32)
test_x = np.asarray(X_test, dtype=np.float32)
train_y = np.asarray(y_train, dtype=np.int64).ravel() - 1
test_y = np.asarray(y_test, dtype=np.int64).ravel() - 1
subjects = read_values('train/subject_train.txt').ravel().astype(int)
test_subjects = read_values('test/subject_test.txt').ravel().astype(int)
class_names = pd.read_csv(DATA / 'activity_labels.txt', sep=r'\s+', header=None)[1].tolist()
assert train_x.shape == (7352, 561) and test_x.shape == (2947, 561)
assert len(subjects) == len(train_y)
assert set(train_y) == set(range(6)) and set(test_y) == set(range(6))
assert not set(subjects) & set(test_subjects)
assert np.isfinite(train_x).all() and np.isfinite(test_x).all()
print('Training:', train_x.shape, 'Testing:', test_x.shape)
print('Activities:', class_names)

splitter = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
fit_idx, val_idx = next(splitter.split(train_x, train_y, groups=subjects))
assert not set(subjects[fit_idx]) & set(subjects[val_idx])
assert set(train_y[fit_idx]) == set(range(6))
assert set(train_y[val_idx]) == set(range(6))
selection_scaler = StandardScaler().fit(train_x[fit_idx])
fit_x = selection_scaler.transform(train_x[fit_idx]).astype(np.float32)
val_x = selection_scaler.transform(train_x[val_idx]).astype(np.float32)
final_scaler = StandardScaler().fit(train_x)
full_x = final_scaler.transform(train_x).astype(np.float32)
eval_x = final_scaler.transform(test_x).astype(np.float32)
print('Validation subjects:', sorted(set(subjects[val_idx])))
print('Fit windows:', len(fit_idx), 'Validation windows:', len(val_idx))
np.savez(OUT / 'scaler.npz', mean=final_scaler.mean_, scale=final_scaler.scale_)

def build_mlp():
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(561,)),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(6, activation='softmax'),
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001, weight_decay=0.0001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'],
    )
    return model

class ValidationMacroF1(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        predictions = self.model.predict(val_x, batch_size=256, verbose=0).argmax(axis=1)
        score = f1_score(train_y[val_idx], predictions, labels=range(6), average='macro', zero_division=0)
        if logs is not None:
            logs['val_macro_f1'] = score

build_mlp().summary()

rows, training_rows, class_rows = [], [], []
for seed in SEEDS:
    print(f'\nSeed {seed}: selecting epoch count...')
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(seed)
    candidate = build_mlp()
    stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_macro_f1', mode='max', patience=PATIENCE,
        restore_best_weights=True,
    )
    history = candidate.fit(
        fit_x, train_y[fit_idx], validation_data=(val_x, train_y[val_idx]),
        epochs=MAX_EPOCHS, batch_size=BATCH_SIZE,
        callbacks=[ValidationMacroF1(), stop], verbose=0,
    )
    best_epochs = int(np.argmax(history.history['val_macro_f1']) + 1)
    pd.DataFrame(history.history).to_csv(OUT / f'validation_seed_{seed}.csv', index=False)
    del candidate
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(seed)
    mlp = build_mlp()
    started = time.perf_counter()
    mlp.fit(full_x, train_y, epochs=best_epochs, batch_size=BATCH_SIZE, verbose=0)
    mlp_seconds = time.perf_counter() - started
    mlp.save(OUT / f'mlp_seed_{seed}.keras')
    rf = RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=-1)
    started = time.perf_counter()
    rf.fit(full_x, train_y)
    rf_seconds = time.perf_counter() - started
    training_rows.append(dict(seed=seed, selected_epochs=best_epochs,
                              mlp_train_seconds=mlp_seconds, rf_train_seconds=rf_seconds))
    base_noise = np.random.default_rng(seed).normal(size=eval_x.shape).astype(np.float32)
    for sigma in NOISE_LEVELS:
        noisy_x = eval_x + np.float32(sigma) * base_noise
        predictions = {
            'MLP': mlp.predict(noisy_x, batch_size=256, verbose=0).argmax(axis=1),
            'Random Forest': rf.predict(noisy_x),
        }
        for model_name, pred in predictions.items():
            rows.append(dict(seed=seed, model=model_name, sigma=sigma,
                             accuracy=accuracy_score(test_y, pred),
                             macro_f1=f1_score(test_y, pred, labels=range(6), average='macro', zero_division=0)))
            report = classification_report(test_y, pred, labels=range(6),
                                           target_names=class_names, output_dict=True, zero_division=0)
            for activity in class_names:
                class_rows.append(dict(seed=seed, model=model_name, sigma=sigma,
                                       activity=activity, **report[activity]))
            slug = model_name.lower().replace(' ', '_')
            pd.DataFrame(confusion_matrix(test_y, pred, labels=range(6)),
                         index=class_names, columns=class_names).to_csv(
                OUT / f'confusion_{slug}_seed_{seed}_sigma_{sigma:.2f}.csv')
    pd.DataFrame(rows).to_csv(OUT / 'all_runs.csv', index=False)
    pd.DataFrame(training_rows).to_csv(OUT / 'training.csv', index=False)
    pd.DataFrame(class_rows).to_csv(OUT / 'per_class.csv', index=False)
    print(f'Finished seed {seed}; MLP epochs: {best_epochs}')

runs = pd.DataFrame(rows)
summary = runs.groupby(['model', 'sigma'], as_index=False).agg(
    accuracy_mean=('accuracy', 'mean'), accuracy_std=('accuracy', 'std'),
    macro_f1_mean=('macro_f1', 'mean'), macro_f1_std=('macro_f1', 'std'),
)
summary.to_csv(OUT / 'summary.csv', index=False)
print(summary.round(4).to_string(index=False))

fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
colors = {'MLP': '#007580', 'Random Forest': '#75B8BD'}
for ax, metric, title in zip(axes, ['accuracy', 'macro_f1'], ['Accuracy', 'Macro F1']):
    for name, group in summary.groupby('model'):
        group = group.sort_values('sigma')
        ax.errorbar(group['sigma'], group[metric + '_mean'],
                    yerr=group[metric + '_std'], color=colors[name],
                    marker='o' if name == 'MLP' else 's', capsize=4, label=name)
    ax.set(title=title, xlabel='Gaussian noise standard deviation (σ)', ylim=(0, 1))
    ax.set_xticks(NOISE_LEVELS)
    ax.grid(alpha=0.25)
    ax.legend()
axes[0].set_ylabel('Score (mean ± SD across 5 seeds)')
fig.tight_layout()
fig.savefig(OUT / 'noise_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

config = dict(seeds=SEEDS, noise_levels=NOISE_LEVELS,
              validation_subjects=sorted(int(s) for s in set(subjects[val_idx])),
              architecture=[561, 256, 128, 6], dropout=0.3, learning_rate=0.001,
              weight_decay=0.0001, max_epochs=MAX_EPOCHS, patience=PATIENCE,
              batch_size=BATCH_SIZE, tensorflow_version=tf.__version__,
              noise_location='standardized extracted test features')
(OUT / 'config.json').write_text(json.dumps(config, indent=2))
