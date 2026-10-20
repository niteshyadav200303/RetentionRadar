"""Single-customer inference and explanation engine for churn and lifetime value."""

from pathlib import Path
import joblib
from lifetimes import BetaGeoFitter, GammaGammaFitter
import numpy as np
import pandas as pd
import shap

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR

# Lazy-loaded global model cache
_MODELS = {}


def _get_models():
    """Load model artifacts once and cache in memory."""
    if not _MODELS:
        churn_artifact = joblib.load(MODELS_DIR / "churn_xgb.joblib")
        _MODELS["churn_model"] = churn_artifact["model"]
        _MODELS["feature_names"] = churn_artifact["feature_names"]
        _MODELS["threshold"] = churn_artifact.get("threshold", 0.5)
        _MODELS["explainer"] = shap.TreeExplainer(_MODELS["churn_model"])

        bgf = BetaGeoFitter()
        bgf.load_model(str(MODELS_DIR / "bgf.pkl"))
        _MODELS["bgf"] = bgf

        ggf = GammaGammaFitter()
        ggf.load_model(str(MODELS_DIR / "ggf.pkl"))
        _MODELS["ggf"] = ggf

        cox_artifact = joblib.load(MODELS_DIR / "cox.joblib")
        _MODELS["cox_model"] = cox_artifact.get("cox_model")

    return _MODELS


def _format_shap_reason(feature_name: str, val: float, shap_val: float) -> str:
    """Translate feature SHAP impact into clear, plain-English business explanations."""
    impact = "increases" if shap_val > 0 else "reduces"
    if feature_name == "Recency":
        return f"{val:.0f} days since last purchase {impact} churn probability significantly."
    elif feature_name == "Frequency":
        return f"Historical order count of {val:.0f} {impact} customer churn likelihood."
    elif feature_name == "PurchasesLast30Days":
        return f"{val:.0f} purchases in the last 30 days {impact} immediate churn risk."
    elif feature_name == "PurchasesLast90Days":
        return f"{val:.0f} orders placed over the trailing 90 days {impact} engagement health."
    elif feature_name == "AvgOrderValue":
        return f"Average order value of ₹{val:.2f} {impact} retention and value profile."
    elif feature_name == "Monetary":
        return f"Cumulative spend of ₹{val:.2f} {impact} baseline churn risk."
    elif feature_name == "Tenure":
        return f"Customer relationship tenure of {val:.0f} days {impact} retention stability."
    elif feature_name == "AvgDaysBetweenOrders":
        return f"Cadence of {val:.1f} days between orders {impact} regular re-order propensity."
    elif feature_name == "UniqueProducts":
        return f"Basket variety of {val:.0f} unique items {impact} catalog affinity."
    elif "Country" in feature_name:
        country_name = feature_name.replace("Country_", "")
        return f"Customer location ({country_name}) {impact} geographic churn baseline."
    else:
        return f"{feature_name} (value: {val:.2f}) {impact} churn prediction."


def score_customer(features_dict: dict) -> dict:
    """Predict churn probability, 6m/12m LTV, remaining lifetime, and top 3 SHAP explanations.

    Args:
        features_dict: Dict with keys Recency, Frequency, Monetary, AvgOrderValue,
                       Tenure, UniqueProducts, AvgDaysBetweenOrders, etc.

    Returns:
        dict: Churn probability, LTV estimates, expected remaining lifetime, and explanations.
    """
    models = _get_models()
    churn_model = models["churn_model"]
    feature_names = models["feature_names"]
    threshold = models["threshold"]
    explainer = models["explainer"]
    bgf = models["bgf"]
    ggf = models["ggf"]

    # Fill defaults for missing features
    row = {col: 0.0 for col in feature_names}
    for k, v in features_dict.items():
        if k in row:
            row[k] = float(v)

    # Automatically compute derived fields if not provided
    freq = float(features_dict.get("Frequency", 1.0))
    monetary = float(features_dict.get("Monetary", 100.0))
    if "AvgOrderValue" not in features_dict or features_dict["AvgOrderValue"] == 0:
        row["AvgOrderValue"] = monetary / max(1.0, freq)

    rec = float(features_dict.get("Recency", 30.0))
    tenure = float(features_dict.get("Tenure", max(rec, 90.0)))
    if "AvgDaysBetweenOrders" not in features_dict or features_dict["AvgDaysBetweenOrders"] == 0:
        row["AvgDaysBetweenOrders"] = (tenure - rec) / max(1.0, freq - 1) if freq > 1 else tenure

    # Handle Country one-hot if given as a string
    country = features_dict.get("Country", "United Kingdom")
    for col in feature_names:
        if col.startswith("Country_"):
            country_part = col.replace("Country_", "")
            row[col] = 1.0 if country == country_part else 0.0

    X_df = pd.DataFrame([row])[feature_names]

    # 1. Churn probability
    churn_prob = float(churn_model.predict_proba(X_df)[0, 1])
    churn_flag = int(churn_prob >= threshold)

    # 2. Top 3 SHAP explanations
    shap_vals = explainer.shap_values(X_df)[0]
    top_indices = np.argsort(np.abs(shap_vals))[::-1][:3]
    top_reasons = []
    for idx in top_indices:
        feat = feature_names[idx]
        val = float(X_df.iloc[0, idx])
        s_val = float(shap_vals[idx])
        top_reasons.append(_format_shap_reason(feat, val, s_val))

    # 3. LTV estimation via BG/NBD and Gamma-Gamma
    # Convert days to weekly scale
    freq_repeat = max(0.0, freq - 1.0)
    rec_weeks = rec / 7.0
    t_weeks = max(rec_weeks, tenure / 7.0)
    avg_monetary = max(1.0, monetary / max(1.0, freq))

    p_alive_raw = bgf.conditional_probability_alive(freq_repeat, rec_weeks, t_weeks)
    prob_alive = float(np.asarray(p_alive_raw).item())
    ltv_6m = float(ggf.customer_lifetime_value(
        bgf,
        pd.Series([freq_repeat]),
        pd.Series([rec_weeks]),
        pd.Series([t_weeks]),
        pd.Series([avg_monetary]),
        time=6,
        discount_rate=0.01,
        freq="W"
    ).iloc[0])

    ltv_12m = float(ggf.customer_lifetime_value(
        bgf,
        pd.Series([freq_repeat]),
        pd.Series([rec_weeks]),
        pd.Series([t_weeks]),
        pd.Series([avg_monetary]),
        time=12,
        discount_rate=0.01,
        freq="W"
    ).iloc[0])

    ltv_6m = max(0.0, round(ltv_6m, 2))
    ltv_12m = max(0.0, round(ltv_12m, 2))

    # 4. Expected remaining lifetime in days
    expected_days = max(14, int(round((tenure + 90) * (prob_alive if churn_flag == 0 else (1.0 - churn_prob)))))

    return {
        "churn_probability": round(churn_prob, 4),
        "churn_flag": churn_flag,
        "probability_alive": round(prob_alive, 4),
        "ltv_6m": ltv_6m,
        "ltv_12m": ltv_12m,
        "expected_remaining_lifetime_days": expected_days,
        "top_reasons": top_reasons
    }


if __name__ == "__main__":
    sample_cust = {
        "Recency": 120,
        "Frequency": 2,
        "Monetary": 350.0,
        "AvgOrderValue": 175.0,
        "Tenure": 240,
        "UniqueProducts": 15,
        "AvgDaysBetweenOrders": 60,
        "Country": "United Kingdom"
    }
    result = score_customer(sample_cust)
    print("Scored Sample Customer:")
    for k, v in result.items():
        print(f"  {k}: {v}")
