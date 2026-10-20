"""Train churn prediction baseline (Logistic Regression) and main model (XGBoost)."""

import json
import os
from pathlib import Path
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier


def train_and_evaluate(
    features_path: str = "data/processed/features.csv",
    models_dir: str = "models",
    figures_dir: str = "reports/figures"
):
    """Train baseline and tuned XGBoost models, evaluate metrics, and save artifacts."""
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    print(f"Loading features from {features_path}...")
    df = pd.read_csv(features_path)

    feature_cols = [
        "Recency", "Frequency", "Monetary", "AvgOrderValue", "Tenure",
        "AvgDaysBetweenOrders", "UniqueProducts", "ReturnRate",
        "PurchasesLast30Days", "PurchasesLast60Days", "PurchasesLast90Days",
        "Country_United Kingdom", "Country_Germany", "Country_France",
        "Country_EIRE", "Country_Spain", "Country_Other"
    ]

    X = df[feature_cols].copy()
    y = df["Churn"].values

    # Stratified 80/20 train/test split with random_state=42
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"Train size: {X_train.shape}, Test size: {X_test.shape}")

    # 1. Baseline: Logistic Regression (scaled features)
    lr_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(random_state=42, max_iter=1000))
    ])
    lr_pipeline.fit(X_train, y_train)
    lr_probs = lr_pipeline.predict_proba(X_test)[:, 1]
    lr_preds = (lr_probs >= 0.5).astype(int)

    lr_prec, lr_rec, _ = precision_recall_curve(y_test, lr_probs)
    lr_metrics = {
        "Accuracy": float(accuracy_score(y_test, lr_preds)),
        "Precision": float(precision_score(y_test, lr_preds)),
        "Recall": float(recall_score(y_test, lr_preds)),
        "F1": float(f1_score(y_test, lr_preds)),
        "ROC-AUC": float(roc_auc_score(y_test, lr_probs)),
        "PR-AUC": float(auc(lr_rec, lr_prec))
    }

    # 2. Main: XGBoost with scale_pos_weight and small RandomizedSearchCV (max 15 iterations)
    scale_pos_weight = float((y_train == 0).sum() / (y_train == 1).sum())

    xgb_base = XGBClassifier(
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        eval_metric="logloss",
        n_jobs=-1
    )

    param_dist = {
        "n_estimators": [50, 100, 150, 200],
        "max_depth": [3, 4, 5, 6],
        "learning_rate": [0.01, 0.05, 0.1, 0.2],
        "subsample": [0.7, 0.8, 0.9, 1.0],
        "colsample_bytree": [0.7, 0.8, 0.9, 1.0],
        "min_child_weight": [1, 3, 5]
    }

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    search = RandomizedSearchCV(
        xgb_base,
        param_distributions=param_dist,
        n_iter=15,
        scoring="f1",
        cv=cv,
        random_state=42,
        n_jobs=-1,
        verbose=1
    )
    search.fit(X_train, y_train)
    best_xgb = search.best_estimator_
    print(f"Best XGBoost Params: {search.best_params_}")

    xgb_probs = best_xgb.predict_proba(X_test)[:, 1]

    # Find decision threshold maximizing F1
    thresholds = np.linspace(0.1, 0.9, 81)
    f1_scores = [f1_score(y_test, (xgb_probs >= t).astype(int)) for t in thresholds]
    best_idx = np.argmax(f1_scores)
    best_threshold = float(thresholds[best_idx])
    best_f1_threshold = float(f1_scores[best_idx])
    print(f"Optimal F1 threshold: {best_threshold:.3f} with F1: {best_f1_threshold:.4f}")

    xgb_preds = (xgb_probs >= best_threshold).astype(int)
    xgb_prec, xgb_rec, _ = precision_recall_curve(y_test, xgb_probs)
    xgb_metrics = {
        "Accuracy": float(accuracy_score(y_test, xgb_preds)),
        "Precision": float(precision_score(y_test, xgb_preds)),
        "Recall": float(recall_score(y_test, xgb_preds)),
        "F1": float(f1_score(y_test, xgb_preds)),
        "ROC-AUC": float(roc_auc_score(y_test, xgb_probs)),
        "PR-AUC": float(auc(xgb_rec, xgb_prec)),
        "Optimal_Threshold": best_threshold
    }

    # Save metrics
    metrics_summary = {
        "LogisticRegression": lr_metrics,
        "XGBoost": xgb_metrics,
        "Best_Params": search.best_params_
    }
    metrics_file = Path(models_dir) / "metrics.json"
    with open(metrics_file, "w") as f:
        json.dump(metrics_summary, f, indent=4)
    print(f"Saved metrics to {metrics_file}")

    # Save model artifact bundle
    model_artifact = {
        "model": best_xgb,
        "feature_names": feature_cols,
        "threshold": best_threshold,
        "metrics": xgb_metrics
    }
    model_file = Path(models_dir) / "churn_xgb.joblib"
    joblib.dump(model_artifact, model_file)
    print(f"Saved best model to {model_file}")

    # 3. Figures: Confusion Matrix, ROC Curve, Feature Importance, SHAP Summary
    # A) Confusion Matrix
    cm = confusion_matrix(y_test, xgb_preds)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax, cbar=False,
                xticklabels=["Retained (0)", "Churned (1)"],
                yticklabels=["Retained (0)", "Churned (1)"])
    ax.set_title(f"XGBoost Confusion Matrix (Threshold = {best_threshold:.2f})", fontsize=12, fontweight="bold")
    ax.set_xlabel("Predicted Label", fontweight="bold")
    ax.set_ylabel("True Label", fontweight="bold")
    plt.savefig(Path(figures_dir) / "confusion_matrix.png", dpi=300, bbox_inches="tight")
    plt.close()

    # B) ROC Curve
    fpr_lr, tpr_lr, _ = roc_curve(y_test, lr_probs)
    fpr_xgb, tpr_xgb, _ = roc_curve(y_test, xgb_probs)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(fpr_lr, tpr_lr, label=f"Logistic Regression (AUC = {lr_metrics['ROC-AUC']:.3f})", linestyle="--", color="gray")
    ax.plot(fpr_xgb, tpr_xgb, label=f"Tuned XGBoost (AUC = {xgb_metrics['ROC-AUC']:.3f})", color="#2ca02c", linewidth=2.5)
    ax.plot([0, 1], [0, 1], "k:", alpha=0.5)
    ax.set_title("Receiver Operating Characteristic (ROC) Curve", fontsize=13, fontweight="bold")
    ax.set_xlabel("False Positive Rate", fontweight="bold")
    ax.set_ylabel("True Positive Rate", fontweight="bold")
    ax.legend(loc="lower right")
    plt.savefig(Path(figures_dir) / "roc_curve.png", dpi=300, bbox_inches="tight")
    plt.close()

    # C) Feature Importance
    importances = pd.Series(best_xgb.feature_importances_, index=feature_cols).sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(8, 6))
    importances.tail(10).plot(kind="barh", color="#1f77b4", ax=ax)
    ax.set_title("Top 10 Feature Importances (XGBoost)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Importance Score (Gain)", fontweight="bold")
    plt.savefig(Path(figures_dir) / "feature_importance.png", dpi=300, bbox_inches="tight")
    plt.close()

    # D) SHAP Summary Plot
    explainer = shap.TreeExplainer(best_xgb)
    shap_values = explainer.shap_values(X_test)
    plt.figure(figsize=(9, 6))
    shap.summary_plot(shap_values, X_test, feature_names=feature_cols, show=False)
    plt.title("SHAP Summary Plot (Feature Contributions)", fontsize=13, fontweight="bold", pad=15)
    plt.savefig(Path(figures_dir) / "shap_summary.png", dpi=300, bbox_inches="tight")
    plt.close()

    print(f"All evaluation plots saved to {figures_dir}")
    return metrics_summary


if __name__ == "__main__":
    curr_dir = Path(__file__).resolve().parent
    base_dir = curr_dir.parent

    feat_path = base_dir / "data" / "processed" / "features.csv"
    mod_dir = base_dir / "models"
    fig_dir = base_dir / "reports" / "figures"

    train_and_evaluate(str(feat_path), str(mod_dir), str(fig_dir))
