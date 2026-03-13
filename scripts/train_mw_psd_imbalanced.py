import os
import mne
import numpy as np
import pandas as pd
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    precision_score,
)
from sklearn.base import clone

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

# ==========================
# Config
# ==========================
data_root = "/gpfs1/pi/djangraw/mindless_reading/data"
sfreq = 256
window_seconds = 2
window_size = int(sfreq * window_seconds)

# ==========================
# EEG band definitions (global)
# ==========================
band_names = [
    "theta1", "theta2", "alpha1", "alpha2",
    "beta1", "beta2", "gamma1", "gamma2",
]

band_defs = [
    (4.0, 6.0),    # theta1
    (6.5, 8.0),    # theta2
    (8.5, 10.0),   # alpha1
    (10.5, 13.0),  # alpha2
    (13.5, 18.0),  # beta1
    (18.5, 30.0),  # beta2
    (30.5, 40.0),  # gamma1
    (40.0, 49.5),  # gamma2
]

n_bands = len(band_defs)

# ==========================
# Helper functions
# ==========================
def get_col_array(data, col_names, targets):
    """
    Return array shaped (n_epochs, seq_len) for the first matching column
    in `targets`. If column not found, return None.
    """
    idx = np.where(np.isin(col_names, targets))[0]
    if idx.size == 0:
        return None

    arr = data[:, idx].astype(np.float64)  # (n_epochs, 1, seq_len) typically
    if arr.ndim == 3 and arr.shape[1] == 1:
        arr = arr[:, 0, :]  # (n_epochs, seq_len)
    return arr


def mean_per_epoch(arr):
    """
    Compute per-epoch mean across valid (non-NaN) timepoints.
    Returns list of length n_epochs; NaN if no valid points.
    """
    out = []
    mask = ~np.isnan(arr)
    for row, m in zip(arr, mask):
        if m.any():
            out.append(np.nanmean(row[m]))
        else:
            out.append(np.nan)
    return out


def unique_counts_per_epoch(arr):
    """
    Count unique valid values per epoch (0 if none).
    """
    out = []
    mask = ~np.isnan(arr)
    for row, m in zip(arr, mask):
        if m.any():
            out.append(np.unique(row[m]).size)
        else:
            out.append(0)
    return out


# ==========================
# Collect subjects
# ==========================
all_subjects = sorted(
    d for d in os.listdir(data_root)
    if d.startswith("s") and os.path.isdir(os.path.join(data_root, d))
)

epoch_counts = []
group_df = []

# ==========================
# Main loop over subjects
# ==========================
for subject_id in all_subjects:

    subject_dir = os.path.join(
        data_root,
        subject_id,
        "ml_data",
        f"{window_size}window_datasets",
    )

    data_file = os.path.join(subject_dir, f"{subject_id}_{window_size}windowed_data_imbalanced.npy")
    label_file = os.path.join(subject_dir, f"{subject_id}_{window_size}windowed_labels_imbalanced.npy")
    col_file = os.path.join(subject_dir, f"{subject_id}_col_names.npy")

    # Skip subjects with missing files
    if not (os.path.exists(data_file) and os.path.exists(label_file) and os.path.exists(col_file)):
        print(f"Missing files for {subject_id}, skipping.")
        epoch_counts.append(0)
        continue

    data = np.load(data_file, allow_pickle=True)
    labels = np.load(label_file)
    col_name = np.load(col_file)

    epoch_count = data.shape[0]
    epoch_counts.append(epoch_count)

    # check sample count for every subject
    if epoch_count < 10:
        print(f"Subject {subject_id} has insufficient data samples: {epoch_count} samples.")
        continue

    # ==========================
    # EEG PSD features
    # ==========================
    # get the 64-channel EEG data (first 64 cols)
    eeg_data = data[:, :64, :].astype(np.float64)

    psds, freqs = mne.time_frequency.psd_array_multitaper(
        eeg_data,
        sfreq,
        fmin=4,
        fmax=50,
        output="power",
    )

    n_epochs, n_ch, _ = psds.shape

    psds_band = np.empty((n_epochs, n_ch, n_bands), dtype=psds.dtype)
    for i, (fmin, fmax) in enumerate(band_defs):
        idx = (freqs >= fmin) & (freqs <= fmax)
        psds_band[:, :, i] = psds[:, :, idx].mean(axis=-1)

    # Flatten features for ML: (n_epochs, n_ch * n_bands)
    psds_band_flat = psds_band.reshape(n_epochs, -1)

    # column names: chan_band
    eeg_columns = [
        f"{ch}_{band}"
        for ch in col_name[:n_ch]
        for band in band_names
    ]
    df_eeg = pd.DataFrame(psds_band_flat, columns=eeg_columns)

    # ==========================
    # Eye-tracking features
    # ==========================
    df_eye = pd.DataFrame(index=np.arange(epoch_count))

    # fixation count
    arr = get_col_array(data, col_name, ["fix_R_tStart"])
    if arr is not None:
        df_eye["fix_count"] = unique_counts_per_epoch(arr)

    # average fixation duration
    arr = get_col_array(data, col_name, ["fix_R_duration"])
    if arr is not None:
        df_eye["fix_R_duration"] = mean_per_epoch(arr)

    # fixation pupil average
    arr = get_col_array(data, col_name, ["fix_R_pupilAvg"])
    if arr is not None:
        df_eye["fix_R_pupilAvg"] = mean_per_epoch(arr)

    # saccade count
    arr = get_col_array(data, col_name, ["sacc_R_tStart"])
    if arr is not None:
        df_eye["sacc_count"] = unique_counts_per_epoch(arr)

    # saccade duration
    arr = get_col_array(data, col_name, ["sacc_R_duration"])
    if arr is not None:
        df_eye["sacc_R_duration"] = mean_per_epoch(arr)

    # saccade amplitude
    arr = get_col_array(data, col_name, ["sacc_R_ampDeg"])
    if arr is not None:
        df_eye["sacc_R_ampDeg"] = mean_per_epoch(arr)

    # saccade peak velocity
    arr = get_col_array(data, col_name, ["sacc_R_vPeak"])
    if arr is not None:
        df_eye["sacc_R_vPeak"] = mean_per_epoch(arr)

    # pupil (normalized)
    arr = get_col_array(data, col_name, ["blink_interp_RPupil_norm"])
    if arr is not None:
        df_eye["pupil_avg"] = mean_per_epoch(arr)

    # drop all-NaN columns if any
    df_eye = df_eye.dropna(axis=1, how="all")

    # ==========================
    # Combine EEG + eye features and save
    # ==========================
    df_data = pd.concat([df_eeg, df_eye], axis=1)
    df_data["label"] = labels
    df_data["subject_id"] = subject_id

    out_file = os.path.join(
        subject_dir,
        f"{subject_id}_{window_size}windowed_features_imbalanced.csv",
    )
    df_data.to_csv(out_file, index=False)
    print(f"Saved features for {subject_id} to: {out_file}")
    group_df.append(df_data)

# ==========================
# Save epoch counts summary
# ==========================
df_epoch_counts = pd.DataFrame({
    "subject_id": all_subjects,
    "n_epochs": epoch_counts,
})
summary_file = os.path.join(
    data_root,
    f"all_subjects_{window_size}_window_epoch_counts_imbalanced.csv",
)
df_epoch_counts.to_csv(summary_file, index=False)

# concatenate once all subjects are processed
df = pd.concat(group_df, ignore_index=True)
# save combined dataframe
df.to_csv(os.path.join(data_root, f"all_subjects_{window_size}windowed_features_imbalanced.csv"), index=False)


# ==============================================================
# start the MW classification pipeline with LOSO CV
df = df.dropna(axis=0, how="any").reset_index(drop=True)

pca_variance = 0.95  # keep 95% variance; or set an int for n_components

# Features: everything except label and subject_id
feature_cols = [c for c in df.columns if c not in ["label", "subject_id"]]
X = df[feature_cols].values
y = df["label"].values
groups = df["subject_id"].values  # for LOSO

# Separate EEG and eye-tracking features
# fix_count is the first eye-feature column
idx_fix = np.where(df.columns.str.contains("fix_count"))[0][0]

X_eeg = X[:, :idx_fix]   # EEG-only features
X_eye = X[:, idx_fix:]   # Eye-only features

# ==========================
# Models
# ==========================
base_models = {
    "logreg": LogisticRegression(max_iter=1000, n_jobs=-1),
    "linear_svc": LinearSVC(),  # no probas, but we can use decision_function
    "rbf_svc": SVC(kernel="rbf", probability=True),
    "random_forest": RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        n_jobs=-1,
        random_state=42,
    ),
    "gradient_boosting": GradientBoostingClassifier(random_state=42),
    "knn": KNeighborsClassifier(n_neighbors=5),
    "mlp": MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=42),
}

# ==========================
# LOSO + prediction logging
# ==========================
logo = LeaveOneGroupOut()

all_preds = []  # per-sample predictions across all folds / models / feature sets

for feature_set, X_data in zip(
    ["EEG + Eye", "EEG", "Eye"],
    [X, X_eeg, X_eye],
):
    print(f"\n========== Feature set: {feature_set} ==========")

    for model_name, base_clf in base_models.items():
        print(f"\n=== Model: {model_name} ===")

        acc_list = []
        f1_list = []
        prec_list = []
        auc_list = []

        for train_idx, test_idx in logo.split(X_data, y, groups=groups):
            subj_test = np.unique(groups[test_idx])
            assert len(subj_test) == 1  # LOSO: only one subject held out
            subj_test = subj_test[0]

            X_train, X_test = X_data[train_idx], X_data[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            # fresh clone of classifier
            clf = clone(base_clf)

            # Pipeline: StandardScaler -> PCA -> classifier
            pipe = Pipeline([
                ("scaler", StandardScaler()),
                ("pca", PCA(n_components=pca_variance)),
                ("clf", clf),
            ])

            pipe.fit(X_train, y_train)
            y_pred = pipe.predict(X_test)

            # scores for AUC
            y_scores = None
            if hasattr(pipe, "predict_proba"):
                try:
                    y_scores = pipe.predict_proba(X_test)[:, 1]
                except Exception:
                    y_scores = None
            elif hasattr(pipe, "decision_function"):
                try:
                    y_scores = pipe.decision_function(X_test)
                except Exception:
                    y_scores = None

            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, zero_division=0)

            if (y_scores is not None) and (np.unique(y_test).size == 2):
                auc = roc_auc_score(y_test, y_scores)
            else:
                auc = np.nan

            acc_list.append(acc)
            f1_list.append(f1)
            prec_list.append(prec)
            auc_list.append(auc)

            # log per-subject fold summary
            print(
                f"  Subject {subj_test}: "
                f"acc={acc:.3f}, prec={prec:.3f}, f1={f1:.3f}, "
                f"auc={auc if not np.isnan(auc) else float('nan'):.3f}"
            )

            # store per-sample predictions for this fold
            for i, idx in enumerate(test_idx):
                all_preds.append({
                    "feature_set": feature_set,
                    "model": model_name,
                    "test_subject": subj_test,
                    "sample_idx": int(idx),
                    "y_true": int(y_test[i]),
                    "y_pred": int(y_pred[i]),
                    "y_score": float(y_scores[i]) if y_scores is not None else np.nan,
                })
            
            # break  # TEMP: only do one fold for testing; REMOVE for full LOSO

        # per-model, per-feature-set mean over subjects
        print(
            f"Mean over subjects — acc={np.mean(acc_list):.3f}, "
            f"prec={np.mean(prec_list):.3f}, "
            f"f1={np.mean(f1_list):.3f}, "
            f"auc={np.nanmean(auc_list):.3f}"
        )

# ==========================
# Build DataFrame of predictions
# ==========================
df_preds = pd.DataFrame(all_preds)

results_dir = os.path.join(data_root, "ml_results")
os.makedirs(results_dir, exist_ok=True)

pred_file = os.path.join(
    results_dir,
    f"loso_predictions_{window_size}win_imbalanced.csv",
)
df_preds.to_csv(pred_file, index=False)
print(f"\nSaved per-sample LOSO predictions to {pred_file}")

# ==========================
# Final metrics from saved predictions (overall)
# ==========================
rows = []
for (feature_set, model_name), g in df_preds.groupby(["feature_set", "model"]):
    y_true = g["y_true"].values
    y_pred = g["y_pred"].values
    y_score = g["y_score"].values

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)

    # AUC over all samples for this (feature_set, model)
    if np.unique(y_true).size == 2 and not np.all(np.isnan(y_score)):
        auc = roc_auc_score(y_true, y_score)
    else:
        auc = np.nan

    rows.append({
        "feature_set": feature_set,
        "model": model_name,
        "accuracy": acc,
        "precision": prec,
        "f1": f1,
        "auc": auc,
        "n_samples": len(g),
    })

df_metrics = pd.DataFrame(rows)

metrics_file = os.path.join(
    results_dir,
    f"loso_metrics_{window_size}win_imbalanced.csv",
)
df_metrics.to_csv(metrics_file, index=False)

print(f"\nSaved aggregated metrics to {metrics_file}")
print("\n=== Overall summary ===")
print(df_metrics.sort_values(["feature_set", "accuracy"], ascending=[True, False]))

# ==========================
# Subject-level metrics
# ==========================
rows_subj = []
for (feature_set, model_name, subj), g in df_preds.groupby(
    ["feature_set", "model", "test_subject"]
):
    y_true = g["y_true"].values
    y_pred = g["y_pred"].values
    y_score = g["y_score"].values

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)

    if np.unique(y_true).size == 2 and not np.all(np.isnan(y_score)):
        auc = roc_auc_score(y_true, y_score)
    else:
        auc = np.nan

    rows_subj.append({
        "feature_set": feature_set,
        "model": model_name,
        "subject_id": subj,
        "accuracy": acc,
        "precision": prec,
        "f1": f1,
        "auc": auc,
        "n_samples": len(g),
    })

df_subject_metrics = pd.DataFrame(rows_subj)

subj_metrics_file = os.path.join(
    results_dir,
    f"loso_subject_metrics_{window_size}win_imbalanced.csv",
)
df_subject_metrics.to_csv(subj_metrics_file, index=False)

print(f"\nSaved subject-level metrics to {subj_metrics_file}")