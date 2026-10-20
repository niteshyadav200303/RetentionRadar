"""Create and execute notebooks/02_modeling.ipynb."""

import json
from pathlib import Path
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

BASE_DIR = Path(__file__).resolve().parent.parent
NB_PATH = BASE_DIR / "notebooks" / "02_modeling.ipynb"

nb = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# 02. Churn Modeling & Feature Interpretability\n",
                "## E-Commerce Customer Churn & Lifetime Value (LTV) Predictor\n",
                "This notebook trains baseline (Logistic Regression) and production (XGBoost with hyperparameter tuning) churn models, finds the optimal decision threshold, and analyzes feature importances via Tree SHAP."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": 1,
            "metadata": {},
            "outputs": [],
            "source": [
                "import os\n",
                "import json\n",
                "import pandas as pd\n",
                "import numpy as np\n",
                "import matplotlib.pyplot as plt\n",
                "import seaborn as sns\n",
                "import joblib\n",
                "import shap\n",
                "\n",
                "from sklearn.model_selection import train_test_split\n",
                "from sklearn.linear_model import LogisticRegression\n",
                "from sklearn.pipeline import Pipeline\n",
                "from sklearn.preprocessing import StandardScaler\n",
                "from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, precision_recall_curve, f1_score\n",
                "from xgboost import XGBClassifier\n",
                "\n",
                "sns.set_theme(style='whitegrid')\n",
                "df = pd.read_csv('../data/processed/features.csv')\n",
                "print('Features dataset shape:', df.shape)\n",
                "df.head()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 1. Train/Test Stratified Split (80/20)"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": 2,
            "metadata": {},
            "outputs": [],
            "source": [
                "feature_cols = [\n",
                "    'Recency', 'Frequency', 'Monetary', 'AvgOrderValue', 'Tenure',\n",
                "    'AvgDaysBetweenOrders', 'UniqueProducts', 'ReturnRate',\n",
                "    'PurchasesLast30Days', 'PurchasesLast60Days', 'PurchasesLast90Days',\n",
                "    'Country_United Kingdom', 'Country_Germany', 'Country_France',\n",
                "    'Country_EIRE', 'Country_Spain', 'Country_Other'\n",
                "]\n",
                "X = df[feature_cols]\n",
                "y = df['Churn'].values\n",
                "\n",
                "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)\n",
                "print(f'Train size: {X_train.shape}, Test size: {X_test.shape}')\n",
                "print(f'Train churn rate: {y_train.mean():.2%}, Test churn rate: {y_test.mean():.2%}')"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 2. Model Evaluation Comparison"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": 3,
            "metadata": {},
            "outputs": [],
            "source": [
                "with open('../models/metrics.json', 'r') as f:\n",
                "    metrics = json.load(f)\n",
                "\n",
                "metrics_df = pd.DataFrame([metrics['LogisticRegression'], metrics['XGBoost']], index=['Logistic Regression', 'XGBoost (Tuned)'])\n",
                "metrics_df"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 3. Confusion Matrix and ROC Curve"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": 4,
            "metadata": {},
            "outputs": [],
            "source": [
                "model_bundle = joblib.load('../models/churn_xgb.joblib')\n",
                "best_xgb = model_bundle['model']\n",
                "threshold = model_bundle['threshold']\n",
                "\n",
                "probs = best_xgb.predict_proba(X_test)[:, 1]\n",
                "preds = (probs >= threshold).astype(int)\n",
                "\n",
                "fig, axes = plt.subplots(1, 2, figsize=(13, 5))\n",
                "cm = confusion_matrix(y_test, preds)\n",
                "sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0], cbar=False,\n",
                "            xticklabels=['Retained', 'Churned'], yticklabels=['Retained', 'Churned'])\n",
                "axes[0].set_title(f'XGBoost Confusion Matrix (Threshold = {threshold:.2f})', fontweight='bold')\n",
                "axes[0].set_xlabel('Predicted Label', fontweight='bold')\n",
                "axes[0].set_ylabel('True Label', fontweight='bold')\n",
                "\n",
                "fpr, tpr, _ = roc_curve(y_test, probs)\n",
                "axes[1].plot(fpr, tpr, color='#2ca02c', linewidth=2.5, label=f'XGBoost (AUC = {roc_auc_score(y_test, probs):.3f})')\n",
                "axes[1].plot([0, 1], [0, 1], 'k--', alpha=0.5)\n",
                "axes[1].set_title('Receiver Operating Characteristic (ROC)', fontweight='bold')\n",
                "axes[1].set_xlabel('False Positive Rate', fontweight='bold')\n",
                "axes[1].set_ylabel('True Positive Rate', fontweight='bold')\n",
                "axes[1].legend()\n",
                "plt.show()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 4. SHAP Feature Attribution Analysis"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": 5,
            "metadata": {},
            "outputs": [],
            "source": [
                "explainer = shap.TreeExplainer(best_xgb)\n",
                "shap_values = explainer.shap_values(X_test)\n",
                "plt.figure(figsize=(9, 6))\n",
                "shap.summary_plot(shap_values, X_test, feature_names=feature_cols, show=False)\n",
                "plt.title('SHAP Feature Contribution Summary', fontsize=13, fontweight='bold', pad=15)\n",
                "plt.show()"
            ]
        }
    ],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.13"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

with open(NB_PATH, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=2)

print(f"Saved notebook template to {NB_PATH}")
