"""
Agent 2 — Conversion Predictor (FULLY AUTO)
Scores each inbound quote with a bind probability (0–100%) using risk tier,
quote timing, coverage preference, salary range, and re-quote behaviour.

Input: Re_Quote, Q_Valid_DT, Coverage, Agent_Type, Region, Sal_Range, HH_Drivers, risk_tier
Target: Policy_Bind (Yes/No) — NOTE: heavy class imbalance (~22% positive) handled via class_weight
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
import pickle, os

MODEL_PATH = os.path.join(os.path.dirname(__file__), "../models/conversion_predictor.pkl")
META_PATH  = os.path.join(os.path.dirname(__file__), "../models/conversion_meta.pkl")

COVERAGE_ORDER = ["Basic", "Balanced", "Enhanced"]
SAL_ORDER = ["<= $ 25 K", "> $ 25 K <= $ 40 K", "> $ 40 K <= $ 60 K",
             "> $ 60 K <= $ 90 K", "> $ 90 K"]
RISK_ORDER = ["Low", "Medium", "High"]
REGION_LIST = list("ABCDEFGH")


def _engineer(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Coverage_enc"]   = df["Coverage"].map({v: i for i, v in enumerate(COVERAGE_ORDER)}).fillna(1)
    df["Sal_enc"]        = df["Sal_Range"].map({v: i for i, v in enumerate(SAL_ORDER)}).fillna(2)
    df["Risk_enc"]       = df["Risk_Tier"].map({v: i for i, v in enumerate(RISK_ORDER)}).fillna(1)
    df["AgentType_enc"]  = (df["Agent_Type"] == "EA").astype(int)
    df["ReQuote_enc"]    = (df["Re_Quote"] == "Yes").astype(int)
    df["Region_enc"]     = df["Region"].map({r: i for i, r in enumerate(REGION_LIST)}).fillna(0)
    # Days to expiry (quote timing)
    try:
        df["Q_Creation_DT"] = pd.to_datetime(df["Q_Creation_DT"], errors="coerce")
        df["Q_Valid_DT"]    = pd.to_datetime(df["Q_Valid_DT"],    errors="coerce")
        df["Days_Valid"]    = (df["Q_Valid_DT"] - df["Q_Creation_DT"]).dt.days.fillna(60)
    except Exception:
        df["Days_Valid"] = 60
    return df

FEATURES = ["Coverage_enc","Sal_enc","Risk_enc","AgentType_enc",
            "ReQuote_enc","Region_enc","Days_Valid","HH_Drivers","Quoted_Premium"]

def train(df: pd.DataFrame):
    # Add synthetic Risk_Tier if not present
    from agents.agent1_risk_profiler import _risk_label, _encode
    if "Risk_Tier" not in df.columns:
        df = _encode(df)
        df["Risk_Tier"] = df.apply(_risk_label, axis=1)

    df = _engineer(df)
    y  = (df["Policy_Bind"] == "Yes").astype(int)
    X  = df[FEATURES]
    clf = GradientBoostingClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, random_state=42
    )
    # Handle class imbalance with sample weights
    ratio = (y == 0).sum() / (y == 1).sum()
    w = y.map({0: 1.0, 1: ratio})
    clf.fit(X, y, sample_weight=w)
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    pickle.dump(clf, open(MODEL_PATH, "wb"))
    # Store feature importances
    fi = dict(zip(FEATURES, clf.feature_importances_))
    pickle.dump(fi, open(META_PATH, "wb"))
    return clf


def load():
    return pickle.load(open(MODEL_PATH, "rb"))


def predict(record: dict, risk_tier: str = "Medium") -> dict:
    clf = load()
    rec = dict(record)
    rec["Risk_Tier"] = risk_tier
    df  = pd.DataFrame([rec])
    df  = _engineer(df)
    for col in FEATURES:
        if col not in df.columns: df[col] = 0
    X    = df[FEATURES]
    prob = float(clf.predict_proba(X)[0][1])
    bind_pct = round(prob * 100, 2)

    # Driver classification
    if bind_pct >= 70:   category = "High Conversion"
    elif bind_pct >= 45: category = "Moderate Conversion"
    elif bind_pct >= 20: category = "Low Conversion"
    else:                category = "Very Unlikely to Bind"

    # Explanation
    drivers = []
    if risk_tier == "Low":          drivers.append("Low risk profile boosts conversion")
    elif risk_tier == "High":       drivers.append("High risk profile suppresses conversion")
    if rec.get("Re_Quote") == "Yes": drivers.append("Re-quote indicates renewed interest")
    cov = rec.get("Coverage", "Balanced")
    if cov == "Enhanced":           drivers.append("Enhanced coverage preference signals intent")
    if not drivers:                 drivers.append("Moderate baseline conversion signals")

    return {
        "bind_probability": bind_pct,
        "conversion_category": category,
        "top_drivers": drivers,
        "explanation": f"Bind probability {bind_pct}% — {category}. Key factors: {'; '.join(drivers)}."
    }
