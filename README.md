# RetentionRadar — E-Commerce Customer Intelligence

<p align="center">
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python"/>
  <img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" alt="Streamlit"/>
  <img src="https://img.shields.io/badge/XGBoost-EC6B23?style=for-the-badge&logo=xgboost&logoColor=white" alt="XGBoost"/>
  <img src="https://img.shields.io/badge/Machine_Learning-Models-blueviolet?style=for-the-badge" alt="Machine Learning"/>
</p>

<p align="center">
  An interactive machine learning application that predicts customer churn, estimates customer lifetime value, and helps businesses identify high-risk customers worth retaining.
</p>

<p align="center">
  <a href="https://retentionradar-f23usj4z9jgmvsmmtczzbs.streamlit.app/">
    <img src="https://img.shields.io/badge/🚀_OPEN_LIVE_WEBSITE-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" alt="Open RetentionRadar Live Website"/>
  </a>
  &nbsp;
  <a href="https://github.com/niteshyadav200303/RetentionRadar">
    <img src="https://img.shields.io/badge/VIEW_SOURCE_CODE-181717?style=for-the-badge&logo=github&logoColor=white" alt="View Source Code"/>
  </a>
</p>

<p align="center">
  <a href="https://github.com/niteshyadav200303/RetentionRadar/stargazers">
    <img src="https://img.shields.io/github/stars/niteshyadav200303/RetentionRadar?style=social" alt="GitHub Stars"/>
  </a>
  <a href="https://github.com/niteshyadav200303/RetentionRadar/network/members">
    <img src="https://img.shields.io/github/forks/niteshyadav200303/RetentionRadar?style=social" alt="GitHub Forks"/>
  </a>
  <a href="https://github.com/niteshyadav200303/RetentionRadar">
    <img src="https://img.shields.io/github/last-commit/niteshyadav200303/RetentionRadar?color=blue" alt="Last Commit"/>
  </a>
</p>

---

## Overview

**RetentionRadar** is an end-to-end e-commerce customer analytics and machine learning project designed to turn transaction history into actionable customer retention insights.

Instead of looking at churn as an isolated yes-or-no prediction, the project combines churn probability, customer lifetime value (LTV), and survival analysis to help prioritize retention efforts.

The application includes an interactive Streamlit dashboard for exploring customer risk, estimating future value, and testing predictions with custom inputs.

## Key Features

- **Customer Churn Prediction** — estimates the probability that a customer will become inactive.
- **Customer Lifetime Value Forecasting** — estimates customer value over 6-month and 12-month horizons.
- **Survival Analysis** — models customer retention and expected remaining lifetime.
- **Customer Lookup** — explores individual customer risk and value estimates.
- **Live Prediction Sandbox** — submits custom customer metrics to the trained prediction models.
- **At-Risk Customer Prioritization** — helps identify high-risk customers with potentially high future value.
- **Model Explainability** — uses SHAP-based feature attribution to help interpret churn predictions.
- **Interactive Analytics** — visualizes churn distributions, customer segments, model performance, and supporting metrics.

## Machine Learning Approach

| Component | Technique | Purpose |
|---|---|---|
| Churn prediction | XGBoost | Estimate customer churn probability |
| Model explainability | Tree SHAP | Understand influential prediction features |
| Survival analysis | Cox Proportional Hazards and Kaplan–Meier | Analyze customer retention over time |
| Purchase behavior | BG/NBD | Estimate future transaction frequency |
| Monetary value | Gamma-Gamma | Estimate expected customer spending |
| Business prioritization | Churn risk × predicted LTV | Highlight potentially valuable customers at risk |

The project uses a historical observation window and a subsequent 90-day evaluation window to define churn for the retail dataset.

## Model Performance

The following are the evaluation results documented for the trained models.

| Metric | XGBoost Result |
|---|---:|
| Accuracy | 75.1% |
| Precision | 73.8% |
| Recall | 86.8% |
| F1 Score | 0.798 |
| ROC-AUC | 0.819 |
| PR-AUC | 0.833 |

Additional documented validation results:

- **Cox Concordance Index:** 0.9174
- **Holdout correlation:** 0.7312

These figures describe the recorded evaluation results; actual performance may differ on new customers or different datasets.

## Dashboard Preview

The repository includes model evaluation charts and exploratory data analysis visuals.

<p align="center">
  <img src="https://raw.githubusercontent.com/niteshyadav200303/RetentionRadar/main/churn_rate_overview.png" width="48%" alt="Churn Rate Overview"/>
  <img src="https://raw.githubusercontent.com/niteshyadav200303/RetentionRadar/main/confusion_matrix.png" width="48%" alt="Churn Model Confusion Matrix"/>
</p>

<p align="center">
  <img src="https://raw.githubusercontent.com/niteshyadav200303/RetentionRadar/main/feature_importance.png" width="48%" alt="Feature Importance"/>
  <img src="https://raw.githubusercontent.com/niteshyadav200303/RetentionRadar/main/roc_curve.png" width="48%" alt="ROC Curve"/>
</p>

<p align="center">
  <a href="https://retentionradar-f23usj4z9jgmvsmmtczzbs.streamlit.app/">
    <img src="https://img.shields.io/badge/Try_RetentionRadar_Live-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" alt="Try RetentionRadar Live"/>
  </a>
</p>

## Technology Stack

- **Language:** Python
- **Dashboard:** Streamlit
- **Data processing:** Pandas, NumPy
- **Machine learning:** XGBoost, scikit-learn
- **Model interpretation:** SHAP
- **Survival analysis:** Lifelines
- **Customer lifetime value:** Lifetimes
- **Model persistence:** Joblib and Pickle
- **Visualization:** Matplotlib and Seaborn

## Dataset

The project is based on the **UCI Online Retail II dataset**, containing historical transactions from a UK-based online retailer.

The documented workflow cleans transaction records, engineers customer-level features, defines a churn evaluation window, and trains the predictive models.

The original transaction data is separate from the trained model artifacts and processed customer data used by the application.

## Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/niteshyadav200303/RetentionRadar.git
cd RetentionRadar
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Launch the dashboard

```bash
streamlit run app.py
```

The local dashboard will be available at the URL displayed in your terminal, usually `http://localhost:8501`.

**Note:** The repository must include the required model artifacts and processed CSV files for the existing dashboard and prediction functions to run. Reproducing model training may also require the original transaction dataset.

## Project Structure

```text
RetentionRadar/
├── app.py
├── predict.py
├── data_prep.py
├── features.py
├── make_synthetic.py
├── train_churn.py
├── train_ltv.py
├── train_survival.py
├── generate_eda_notebook.py
├── generate_modeling_notebook.py
├── 01_eda.ipynb
├── 02_modeling.ipynb
├── requirements.txt
├── customer_scores.csv
├── features.csv
├── churn_xgb.joblib
├── cox.joblib
├── bgf.pkl
├── ggf.pkl
├── metrics.json
└── README.md
```

The repository also contains evaluation charts, analysis images, and helper scripts.

## Future Improvements

- Monitor model performance as customer behavior changes.
- Improve model validation on additional time periods.
- Add automated retention campaign recommendations.
- Integrate new transaction data for refreshed customer scores.
- Expand explanations of individual customer predictions.

## License

The project README currently describes the project as MIT-licensed. Before reusing or distributing the project, ensure the repository contains the corresponding `LICENSE` file.

---

<p align="center">
  <b>RetentionRadar — Predict risk. Understand value. Prioritize retention.</b>
</p>

<p align="center">
  <a href="https://retentionradar-f23usj4z9jgmvsmmtczzbs.streamlit.app/">Live Website</a>
  ·
  <a href="https://github.com/niteshyadav200303/RetentionRadar">GitHub Repository</a>
</p>
