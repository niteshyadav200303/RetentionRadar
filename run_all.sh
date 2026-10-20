#!/usr/bin/env bash
set -e

echo "======================================================="
echo "  E-Commerce Customer Churn & LTV Predictor Pipeline"
echo "======================================================="

echo "[1/7] Cleaning raw transactions..."
python src/data_prep.py

echo "[2/7] Generating EDA notebook and figures..."
python src/generate_eda_notebook.py

echo "[3/7] Engineering customer features..."
python src/features.py

echo "[4/7] Training churn models (Logistic Regression + XGBoost)..."
python src/train_churn.py

echo "[5/7] Executing modeling notebook..."
python src/generate_modeling_notebook.py

echo "[6/7] Running survival analysis (Kaplan-Meier + Cox PH)..."
python src/train_survival.py

echo "[7/7] Training LTV models (BG/NBD + Gamma-Gamma) and scoring customers..."
python src/train_ltv.py

echo "======================================================="
echo "Pipeline complete! Launching interactive dashboard..."
echo "======================================================="
streamlit run app/app.py
