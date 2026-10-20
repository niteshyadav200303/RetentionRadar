@echo off
echo =======================================================
echo   E-Commerce Customer Churn & LTV Predictor Pipeline
echo =======================================================

echo [1/7] Cleaning raw transactions...
python src/data_prep.py
if %ERRORLEVEL% neq 0 ( echo Error in data_prep.py & exit /b %ERRORLEVEL% )

echo [2/7] Generating EDA notebook and figures...
python src/generate_eda_notebook.py
if %ERRORLEVEL% neq 0 ( echo Error in generate_eda_notebook.py & exit /b %ERRORLEVEL% )

echo [3/7] Engineering customer features...
python src/features.py
if %ERRORLEVEL% neq 0 ( echo Error in features.py & exit /b %ERRORLEVEL% )

echo [4/7] Training churn models (Logistic Regression + XGBoost)...
python src/train_churn.py
if %ERRORLEVEL% neq 0 ( echo Error in train_churn.py & exit /b %ERRORLEVEL% )

echo [5/7] Executing modeling notebook...
python src/generate_modeling_notebook.py
if %ERRORLEVEL% neq 0 ( echo Error in generate_modeling_notebook.py & exit /b %ERRORLEVEL% )

echo [6/7] Running survival analysis (Kaplan-Meier + Cox PH)...
python src/train_survival.py
if %ERRORLEVEL% neq 0 ( echo Error in train_survival.py & exit /b %ERRORLEVEL% )

echo [7/7] Training LTV models (BG/NBD + Gamma-Gamma) and scoring customers...
python src/train_ltv.py
if %ERRORLEVEL% neq 0 ( echo Error in train_ltv.py & exit /b %ERRORLEVEL% )

echo =======================================================
echo Pipeline complete! Launching interactive dashboard...
echo =======================================================
streamlit run app/app.py
