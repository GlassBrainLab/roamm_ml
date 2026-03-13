import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
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
# Config
# ==========================
data_root = "/gpfs1/pi/djangraw/mindless_reading/data"
sfreq = 256
window_seconds = 2
window_size = int(sfreq * window_seconds)
dataset_type = "balanced"
pca_variance = 0.95  # keep 95% variance; or set an int for n_components
# 10-fold CV strategy for evaluation of overall performance
n_splits = 10
cv_strategy = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

# ==========================
# All subjects
# ==========================
all_subjects = sorted(
    d for d in os.listdir(data_root)
    if d.startswith("s") and os.path.isdir(os.path.join(data_root, d))
)

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

    # Load the dataset for this subject
    df = pd.read_csv(os.path.join(subject_dir, f"{subject_id}_{window_size}windowed_features.csv"))
    # Drop rows with any NaN values (if any)
    df = df.dropna(axis=0, how="any").reset_index(drop=True)
    # skip if MW samples are too few (e.g., <10) for this subject
    if df["label"].sum() < 10:
        print(f"Skipping {subject_id} due to too few MW samples ({df['label'].sum()})")
        continue
    
    print(f"\n=== Processing subject {subject_id} ===")
    # Features: everything except label and subject_id
    feature_cols = [c for c in df.columns if c not in ["label", "subject_id"]]
    X = df[feature_cols].values
    y = df["label"].values

    # Separate EEG and eye-tracking features
    # fix_count is the first eye-feature column
    idx_fix = np.where(df.columns.str.contains("fix_count"))[0][0]
    X_eeg = X[:, :idx_fix]   # EEG-only features
    X_eye = X[:, idx_fix:]   # Eye-only features

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

            for fold_idx, (train_idx, test_idx) in enumerate(cv_strategy.split(X_data, y)):
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
                    f"acc={acc:.3f}, prec={prec:.3f}, f1={f1:.3f}, "
                    f"auc={auc if not np.isnan(auc) else float('nan'):.3f}"
                )

                # store per-sample predictions for this fold
                for i, idx in enumerate(test_idx):
                    all_preds.append({
                        "feature_set": feature_set,
                        "model": model_name,
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
    pred_file = os.path.join(
        subject_dir,
        f"{subject_id}_{window_size}win_pred_{dataset_type}.csv",
    )
    df_preds.to_csv(pred_file, index=False)

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
        subject_dir,
        f"{subject_id}_{window_size}win_metrics_{dataset_type}.csv",
    )
    df_metrics.to_csv(metrics_file, index=False)

    print(f"\n=== Overall summary {subject_id}===")
    print(df_metrics.sort_values(["feature_set", "accuracy"], ascending=[True, False]))