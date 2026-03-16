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
# Models with Added Penalties
# ==========================
base_models = {
    # C=0.1: Stronger regularization than default (1.0)
    "logreg": LogisticRegression(
        l1_ratio=0, # L2 penalty only
        C=0.1, 
        solver="lbfgs", 
        max_iter=1000
    ),
    
    # LinearSVC: Lower C and dual=False for faster convergence with penalty
    "linear_svc": LinearSVC(
        C=0.01, 
        dual=False, 
        max_iter=2000
    ),
    
    # RBF SVC: Lower C to increase margin; gamma='scale' for auto-regularization
    "rbf_svc": SVC(
        kernel="rbf", 
        C=0.5, 
        gamma="scale", 
        probability=True
    ),
    
    # Random Forest: Constrained depth and bootstrapping (80%)
    "random_forest": RandomForestClassifier(
        n_estimators=200,
        max_depth=10,            # Prevents trees from memorizing training samples
        min_samples_leaf=5,      # Requires more samples to create a leaf
        max_samples=0.8,         # Trains each tree on a random 80% subset
        random_state=42,
    ),
    
    # Gradient Boosting: Lower learning rate + Stochastic subsampling
    "gradient_boosting": GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.05,      # Slows down learning to prevent overshooting
        subsample=0.8,           # Stochastic Gradient Boosting
        max_depth=3,             # Keeps base learners very simple
        random_state=42
    ),
    
    # k-NN: Higher k value results in smoother decision boundaries
    "knn": KNeighborsClassifier(n_neighbors=15),
    
    # MLP: Higher alpha (L2 penalty) and Early Stopping
    "mlp": MLPClassifier(
        hidden_layer_sizes=(50,), # Smaller layer to limit capacity
        alpha=0.1,                # Strong weight penalty
        early_stopping=True,      # Automatically stops if validation performance drops
        validation_fraction=0.1, 
        max_iter=500, 
        random_state=42
    ),
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
    f"loso_predictions_{window_size}win.csv",
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
    f"loso_metrics_{window_size}win_reg.csv",
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
    f"loso_subject_metrics_{window_size}win_reg.csv",
)
df_subject_metrics.to_csv(subj_metrics_file, index=False)

print(f"\nSaved subject-level metrics to {subj_metrics_file}")
