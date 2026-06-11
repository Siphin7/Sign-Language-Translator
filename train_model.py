#!/usr/bin/env python3
# =============================================================
#  train_model.py  ──  STEP 2: Train the sign classifier
# =============================================================
#
#  Prerequisites
#  -------------
#  Run collect_data.py first to build dataset/gesture_data.csv
#
#  What this script does
#  ---------------------
#  1. Loads and analyses your dataset
#  2. Trains two models: Random Forest + MLP Neural Network
#  3. Picks the better one and saves to models/gesture_model.pkl
#  4. Plots a confusion matrix so you can spot weak signs
#  5. Prints per-class accuracy so you know what to improve
#
#  Usage
#  -----
#    python train_model.py
# =============================================================

import os
import sys
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from sklearn.ensemble         import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network   import MLPClassifier
from sklearn.model_selection  import train_test_split, cross_val_score
from sklearn.preprocessing    import LabelEncoder, StandardScaler
from sklearn.pipeline         import Pipeline
from sklearn.metrics          import (classification_report, confusion_matrix,
                                      accuracy_score)

# ── Paths ────────────────────────────────────────────────────────────
DATASET_PATH = os.path.join("dataset", "gesture_data.csv")
MODEL_PATH   = os.path.join("models",  "gesture_model.pkl")
PLOT_DIR     = os.path.join("models",  "plots")
os.makedirs("models", exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)

# ── Reproducibility ──────────────────────────────────────────────────
SEED = 42
np.random.seed(SEED)


# ═════════════════════════════════════════════════════════════════════
def load_dataset(path: str):
    """Load CSV produced by collect_data.py."""
    if not os.path.exists(path):
        print(f"\nERROR: Dataset not found at '{path}'")
        print("Run  python collect_data.py  first.\n")
        sys.exit(1)

    df = pd.read_csv(path, header=None)
    print(f"\nDataset shape: {df.shape}")

    labels   = df.iloc[:, 0].values                 # first column = label
    features = df.iloc[:, 1:].values.astype(float)  # rest = 63 landmark coords

    # Class distribution
    unique, counts = np.unique(labels, return_counts=True)
    print(f"\nClasses found: {len(unique)}")
    print("─" * 50)
    for cls, cnt in sorted(zip(unique, counts), key=lambda x: -x[1]):
        bar = "█" * (cnt // 5)
        print(f"  {cls:<14} {cnt:>5} samples  {bar}")

    low = [(c, n) for c, n in zip(unique, counts) if n < 50]
    if low:
        print(f"\n⚠  Low-sample classes (< 50) — consider collecting more:")
        for c, n in low:
            print(f"   {c}: {n}")

    return features, labels, list(unique)


def build_models():
    """Return a dict of sklearn estimator pipelines."""
    return {
        "RandomForest": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(
                n_estimators   = 200,
                max_depth      = None,
                min_samples_leaf = 2,
                n_jobs         = -1,
                random_state   = SEED,
            ))
        ]),
        "MLP": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", MLPClassifier(
                hidden_layer_sizes = (256, 128, 64),
                activation         = "relu",
                solver             = "adam",
                max_iter           = 500,
                early_stopping     = True,
                n_iter_no_change   = 15,
                random_state       = SEED,
            ))
        ]),
        "GradientBoosting": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", GradientBoostingClassifier(
                n_estimators  = 150,
                learning_rate = 0.1,
                max_depth     = 5,
                random_state  = SEED,
            ))
        ]),
    }


def plot_confusion_matrix(y_true, y_pred, classes, name: str):
    """Save a colour confusion matrix PNG."""
    cm  = confusion_matrix(y_true, y_pred, labels=classes)
    fig_sz = max(10, len(classes) // 2)
    fig, ax = plt.subplots(figsize=(fig_sz, fig_sz))

    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=classes, yticklabels=classes,
        linewidths=0.5, ax=ax,
    )
    ax.set_xlabel("Predicted",  fontsize=12)
    ax.set_ylabel("True Label", fontsize=12)
    ax.set_title(f"Confusion Matrix — {name}", fontsize=14, fontweight="bold")
    plt.tight_layout()

    out = os.path.join(PLOT_DIR, f"confusion_matrix_{name}.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  Confusion matrix saved → {out}")


def plot_feature_importance(model_pipeline, feature_count: int, name: str):
    """Feature importance (RandomForest only)."""
    clf = model_pipeline.named_steps.get("clf")
    if not hasattr(clf, "feature_importances_"):
        return

    imp  = clf.feature_importances_
    # Group by landmark index (each landmark = 3 features: x,y,z)
    lm_importance = np.array([
        imp[i*3 : i*3+3].sum() for i in range(feature_count // 3)
    ])

    landmark_names = [
        "Wrist", "Thumb_CMC", "Thumb_MCP", "Thumb_IP", "Thumb_Tip",
        "Idx_MCP", "Idx_PIP", "Idx_DIP", "Idx_Tip",
        "Mid_MCP", "Mid_PIP", "Mid_DIP", "Mid_Tip",
        "Rng_MCP", "Rng_PIP", "Rng_DIP", "Rng_Tip",
        "Pink_MCP", "Pink_PIP", "Pink_DIP", "Pink_Tip",
    ]

    fig, ax = plt.subplots(figsize=(12, 5))
    colors  = plt.cm.viridis(lm_importance / lm_importance.max())
    ax.bar(landmark_names, lm_importance, color=colors)
    ax.set_xlabel("Landmark")
    ax.set_ylabel("Importance")
    ax.set_title(f"Landmark Importance — {name}")
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.tight_layout()

    out = os.path.join(PLOT_DIR, f"feature_importance_{name}.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  Feature importance saved → {out}")


# ═════════════════════════════════════════════════════════════════════
def main():
    print("""
╔══════════════════════════════════════════════════════════╗
║   SIGN LANGUAGE TRANSLATOR — MODEL TRAINING              ║
╚══════════════════════════════════════════════════════════╝
    """)

    # 1. Load data
    X, y, classes = load_dataset(DATASET_PATH)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )
    print(f"\nTrain size: {len(X_train)}  |  Test size: {len(X_test)}")

    # 2. Train and evaluate each model
    print("\n" + "═" * 60)
    print("  Training models...")
    print("═" * 60)

    models     = build_models()
    results    = {}

    for name, pipeline in models.items():
        print(f"\n  [{name}] training...", end=" ", flush=True)
        pipeline.fit(X_train, y_train)
        y_pred   = pipeline.predict(X_test)
        acc      = accuracy_score(y_test, y_pred)
        results[name] = {"pipeline": pipeline, "acc": acc, "y_pred": y_pred}
        print(f"Test accuracy: {acc:.4f}  ({acc*100:.1f}%)")

        # Cross-validation for robustness estimate
        cv_scores = cross_val_score(pipeline, X, y, cv=5, scoring="accuracy", n_jobs=-1)
        print(f"  [{name}] 5-fold CV: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # 3. Pick best model
    best_name     = max(results, key=lambda k: results[k]["acc"])
    best          = results[best_name]
    best_pipeline = best["pipeline"]
    y_pred_best   = best["y_pred"]

    print(f"\n  ★ Best model: {best_name}  ({best['acc']*100:.1f}% accuracy)")

    # 4. Detailed report
    print("\n" + "═" * 60)
    print("  PER-CLASS ACCURACY (best model)")
    print("═" * 60)
    report = classification_report(y_test, y_pred_best, target_names=sorted(classes))
    print(report)

    # 5. Plots
    plot_confusion_matrix(y_test, y_pred_best, sorted(classes), best_name)
    plot_feature_importance(best_pipeline, X.shape[1], best_name)

    # 6. Save model
    payload = {
        "model"     : best_pipeline,
        "labels"    : sorted(classes),
        "model_name": best_name,
        "accuracy"  : best["acc"],
    }
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(payload, f)

    print(f"\n  ✓ Model saved → {os.path.abspath(MODEL_PATH)}")
    print(f"  ✓ Accuracy   : {best['acc']*100:.1f}%")
    print(f"  ✓ Signs      : {sorted(classes)}")
    print("""
  ─────────────────────────────────────────────
  Next step:  python app.py
  ─────────────────────────────────────────────
    """)


if __name__ == "__main__":
    main()
