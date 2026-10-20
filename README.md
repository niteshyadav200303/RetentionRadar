# E-Commerce Customer Churn & Lifetime Value (LTV) Predictor

An end-to-end, production-grade machine learning system that predicts individual customer churn, models customer retention survival curves, and forecasts future Customer Lifetime Value (LTV) using probabilistic buy-till-you-die models, survival analysis, and XGBoost with Tree SHAP explainability.

---

## 📌 Problem Statement

Customer acquisition in e-commerce is 5x to 25x more expensive than retaining existing customers. However, traditional churn models treat retention as a static classification task, leading to two major pitfalls:
1. **Data Leakage & Arbitrary Churn Definitions:** In non-contractual retail, customers do not explicitly cancel accounts; they silently churn. Arbitrary churn definitions create severe forward-looking leakage.
2. **Missing Customer Value Context:** Preventing a customer with £10 expected spend from churning produces negligible ROI compared to defending a customer with £2,000+ expected annual value.

This project solves both challenges by combining:
- A rigorous, leakage-free 90-day post-cutoff churn evaluation window.
- **XGBoost classification** with Tree SHAP explainability for immediate churn risk.
- **Cox Proportional Hazards & Kaplan-Meier survival analysis** to model remaining customer tenure.
- **Beta-Geometric / Negative Binomial (BG/NBD) + Gamma-Gamma** probabilistic models to estimate future transaction frequency and monetary lifetime value.
- An interactive **Streamlit Executive Dashboard** offering real-time scoring, cohort monitoring, and revenue-at-risk prioritization.

---

## 📊 Dataset

- **Source:** [UCI Machine Learning Repository: Online Retail II Data Set](https://archive.ics.uci.edu/dataset/502/online+retail+ii)
- **Timeframe:** December 2009 – December 2011 (Two full years of actual transactions from a UK-based online retailer).
- **Volume:** 1,067,371 raw transaction rows across two annual sheets (`Year 2009-2010` and `Year 2010-2011`).
- **Cleaned Volume:** 805,549 verified transactions across 5,878 unique customer accounts after filtering missing IDs, cancellations, and negative entries.

---

## 🧠 Methodology & Approach

### 1. Leakage-Free Churn Definition
- **Cutoff Date:** Last transaction timestamp in the dataset minus 90 days (`2011-09-10 12:50:00`).
- **Observation Window:** All transactions occurring strictly **before** the cutoff date are used to engineer customer features.
- **Evaluation Window:** The 90-day window following the cutoff date.
- **Churn Label:**
  $$\text{Churn} = \begin{cases} 1 & \text{if customer made 0 purchases in the 90-day evaluation window} \\ 0 & \text{if customer made } \ge 1 \text{ purchase in the 90-day evaluation window} \end{cases}$$

### 2. Why Survival Analysis?
Traditional binary classification only predicts *if* an event happens within a static window. **Survival analysis (Kaplan-Meier and Cox Proportional Hazards)** models *when* the event will happen and accounts for right-censoring:
- Computes baseline survival probability across customer lifespans.
- Identifies hazard ratios for order frequency, basket value, and repurchase cadence.
- Yields expected remaining lifetime days for personalized retention timing.

### 3. BG/NBD & Gamma-Gamma Probabilistic LTV
In non-contractual commerce, purchasing and churn are latent continuous processes:
- **BetaGeoFitter (BG/NBD):** Decomposes transaction frequency into a Poisson purchase process while active, and an exponential dropout process with Beta-distributed churn propensity.
- **GammaGammaFitter:** Assumes transaction monetary values vary randomly around customer-specific average spends, modeled via a Gamma distribution.
- **CLV Forecast:** Combines both models discounted at 1% weekly rate over 6-month and 12-month forward horizons:
  $$\text{CLV} = \text{Expected Transactions} \times \text{Expected Average Spend} \times \text{Discount Factor}$$

---

## 📈 Model Performance & Results

### Churn Classification Comparison (Test Set Evaluation)

| Metric | Baseline: Logistic Regression (Scaled) | Main: Tuned XGBoost (RandomizedSearchCV) |
|---|:---:|:---:|
| **Accuracy** | 74.6% | **75.1%** |
| **Precision** | **75.4%** | 73.8% |
| **Recall** | 81.9% | **86.8%** |
| **F1 Score** | 0.785 | **0.798** |
| **ROC-AUC** | 0.808 | **0.819** |
| **PR-AUC** | **0.834** | 0.833 |
| **Decision Threshold** | 0.500 | **0.440** (Optimized for F1) |

### Survival & LTV Validation

| Metric | Score | Interpretation |
|---|:---:|---|
| **Cox Concordance Index** | **0.9174** | Exceptional discriminative ranking of customer survival timelines. |
| **Holdout Correlation** | **0.7312** | High correlation between predicted and actual customer purchases in holdout period. |
| **High-Risk High-Value Cohort** | **196 Customers** | High churn risk with top 20% predicted LTV, representing prime targets for VIP retention. |

---

## 🖼️ Application Screenshots & Visuals

*(Visualizations generated in `reports/figures/` and accessible live in the Streamlit app)*

- **Overview Dashboard:** Executive KPIs, churn probability distribution, and segment LTV bar charts.
  `reports/figures/monthly_revenue_and_active_customers.png`
- **Confusion Matrix & ROC:**
  `reports/figures/confusion_matrix.png` | `reports/figures/roc_curve.png`
- **Feature Attribution (Tree SHAP):**
  `reports/figures/shap_summary.png`
- **Survival Curves (Kaplan-Meier):**
  `reports/figures/survival_curves.png`

---

## 📁 Project Structure

```text
churn-ltv-predictor/
├── data/
│   ├── raw/                      # Raw UCI Online Retail II dataset (git-ignored)
│   └── processed/                # clean_transactions.csv, features.csv, customer_scores.csv
├── models/                       # churn_xgb.joblib, cox.joblib, bgf.pkl, ggf.pkl, metrics.json
├── notebooks/
│   ├── 01_eda.ipynb              # Exploratory data analysis with inline visual plots
│   └── 02_modeling.ipynb         # Model training, threshold tuning, and SHAP analysis
├── src/
│   ├── make_synthetic.py         # Realistic synthetic transaction generator fallback
│   ├── data_prep.py              # Data cleaning and standardization pipeline
│   ├── generate_eda_notebook.py  # Generates 01_eda.ipynb and reports/figures
│   ├── features.py               # Feature engineering and cutoff churn labeling
│   ├── train_churn.py            # Logistic Regression and XGBoost training with SHAP
│   ├── generate_modeling_notebook.py # Generates 02_modeling.ipynb
│   ├── train_survival.py         # Lifelines Kaplan-Meier and Cox PH model
│   ├── train_ltv.py              # Lifetimes BG/NBD and Gamma-Gamma pipeline
│   └── predict.py                # Real-time score_customer() inference engine
├── app/
│   └── app.py                    # Multi-page interactive Streamlit dashboard
├── reports/
│   └── figures/                  # Publication-ready charts, ROC curves, SHAP plots
├── requirements.txt              # Production dependency specifications
├── run_all.bat                   # 1-click Windows pipeline script
├── run_all.sh                    # 1-click Unix/macOS pipeline script
└── README.md                     # Comprehensive project documentation
```

---

## 🚀 Installation & Quickstart

### 1. Clone the Repository and Set Up Environment
```bash
git clone https://github.com/your-username/churn-ltv-predictor.git
cd churn-ltv-predictor

# Create and activate virtual environment
python -m venv venv
# Windows:
venv\Scripts\activate
# Unix/macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Execute Full Pipeline in Order
You can run the full automated pipeline with a single command:

**Windows:**
```cmd
run_all.bat
```

**Linux / macOS:**
```bash
chmod +x run_all.sh
./run_all.sh
```

Or execute individual pipeline stages sequentially:
```bash
# 1. Clean transactions
python src/data_prep.py

# 2. Engineer features & churn label
python src/features.py

# 3. Train churn models & SHAP explanations
python src/train_churn.py

# 4. Train Kaplan-Meier and Cox survival model
python src/train_survival.py

# 5. Train BG/NBD and Gamma-Gamma LTV models
python src/train_ltv.py

# 6. Launch interactive Streamlit dashboard
streamlit run app/app.py
```

---

## 🖥️ Streamlit Dashboard Pages

1. **📊 Overview Dashboard:** Executive KPI summary cards (Customer Count, Baseline Churn Rate, Total 12-Month LTV, High-Risk High-Value targets), interactive churn probability histogram, segment LTV breakdown, Kaplan-Meier curves, and expandable technical model diagnostics.
2. **🔍 Customer Lookup:** Search any Customer ID to display an interactive churn risk gauge, 6m/12m LTV forecast, probability alive, personalized projected survival curve, top 3 SHAP drivers, and automated retention action recommendations.
3. **⚡ Live Prediction:** Interactive sandbox form accepting customer transaction metrics to test real-time inference via `score_customer()`, returning churn probability, expected survival days, and plain-English explanations.
4. **⚠️ At-Risk Customers:** Prioritized target table ranked by **Revenue at Risk** ($\text{Churn Probability} \times \text{LTV}_{12\text{m}}$) with multidimensional filters (risk level, segment, country) and instant CSV export.

---

## 🔮 Limitations & Future Work

- **Seasonality & Macro Effects:** Current BG/NBD formulation assumes stationary purchase rates; incorporating holiday and seasonal spikes (e.g. Q4 Black Friday / Christmas) using Hierarchical Bayesian models in PyMC would improve calendar accuracy.
- **Product Recommender Integration:** Combining the churn trigger with collaborative filtering to suggest specific SKU replacements directly inside the automated retention playbook.
- **Dynamic Pricing Elasticity:** Extending the retention recommendation module to estimate individual price elasticity before offering percentage discounts.

---

## 📄 License
MIT License. Open for educational and commercial adaptation.
