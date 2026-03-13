import os
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
from sklearn.model_selection import StratifiedKFold

# ==========================
# Config & data loading
# ==========================
data_root = "/gpfs1/pi/djangraw/mindless_reading/data"
sfreq = 256
window_seconds = 2
window_size = int(sfreq * window_seconds)
df = pd.read_csv(
    os.path.join(data_root, f"all_subjects_{window_size}windowed_features.csv")
)
df = df.dropna(axis=0, how="any").reset_index(drop=True)

# 10-fold CV strategy for evaluation of overall performance
n_splits = 10
cv_strategy = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
pca_variance = 0.95  # keep 95% variance; or set an int for n_components

# Features: everything except label and subject_id
feature_cols = [c for c in df.columns if c not in ["label", "subject_id"]]
X = df[feature_cols].values
y = df["label"].values

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

        acc_list, f1_list, prec_list, auc_list = [], [], [], []

        # Changed to Stratified 10-Fold
        for fold_idx, (train_idx, test_idx) in enumerate(cv_strategy.split(X_data, y)):
            
            X_train, X_test = X_data[train_idx], X_data[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            clf = clone(base_clf)
            pipe = Pipeline([
                ("scaler", StandardScaler()),
                ("pca", PCA(n_components=pca_variance)),
                ("clf", clf),
            ])

            pipe.fit(X_train, y_train)
            y_pred = pipe.predict(X_test)

            # --- Logic for Scores (AUC) remains the same ---
            y_scores = None
            if hasattr(pipe, "predict_proba"):
                y_scores = pipe.predict_proba(X_test)[:, 1]
            elif hasattr(pipe, "decision_function"):
                y_scores = pipe.decision_function(X_test)

            # Calculate metrics
            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, zero_division=0)
            auc = roc_auc_score(y_test, y_scores) if y_scores is not None else np.nan

            acc_list.append(acc)
            f1_list.append(f1)
            prec_list.append(prec)
            auc_list.append(auc)

            print(f"  Fold {fold_idx + 1}: acc={acc:.3f}, f1={f1:.3f}, auc={auc:.3f}")

            # Store per-sample predictions
            for i, idx in enumerate(test_idx):
                all_preds.append({
                    "feature_set": feature_set,
                    "model": model_name,
                    "fold": fold_idx + 1,
                    "sample_idx": int(idx),
                    "y_true": int(y_test[i]),
                    "y_pred": int(y_pred[i]),
                    "y_score": float(y_scores[i]) if y_scores is not None else np.nan,
                })

        print(f"Mean over folds — acc={np.mean(acc_list):.3f}, f1={np.mean(f1_list):.3f}")

# ==========================
# Build DataFrame of predictions
# ==========================
df_preds = pd.DataFrame(all_preds)

results_dir = os.path.join(data_root, "ml_results")
os.makedirs(results_dir, exist_ok=True)

pred_file = os.path.join(
    results_dir,
    f"loso_predictions_{window_size}win_10cv.csv",
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
    f"loso_metrics_{window_size}win_10cv.csv",
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
    f"loso_subject_metrics_{window_size}win_10cv.csv",
)
df_subject_metrics.to_csv(subj_metrics_file, index=False)

print(f"\nSaved subject-level metrics to {subj_metrics_file}")