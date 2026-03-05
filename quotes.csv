"""
Agent 1 — Risk Profiler (FULLY AUTO)
Ingests raw quote data and computes a real-time risk tier (Low / Medium / High)
for every bound or soon-to-expire policy. Drives all downstream agent decisions.

Input features: Prev_Accidents, Prev_Citations, Driving_Exp, Driver_Age, Veh_Usage, Annual_Miles
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import pickle, os

MODEL_PATH = os.path.join(os.path.dirname(__file__), "../models/risk_profiler.pkl")
ENC_PATH   = os.path.join(os.path.dirname(__file__), "../models/risk_le.pkl")

FEATURES = ["Prev_Accidents", "Prev_Citations", "Driving_Exp",
            "Driver_Age", "Veh_Usage_enc", "Annual_Miles_enc"]

ANNUAL_MILES_ORDER = [
    "<= 7.5 K", "> 7.5 K & <= 15 K", "> 15 K & <= 25 K",
    "> 25 K & <= 35 K", "> 35 K & <= 45 K", "> 45 K & <= 55 K", "> 55 K"
]
VEH_USAGE_ORDER = ["Pleasure", "Commute", "Business"]


def _risk_label(row):
    """Create synthetic risk tiers from domain rules (used for training labels)."""
    score = 0
    score += row["Prev_Accidents"] * 3
    score += row["Prev_Citations"] * 2
    if row["Driver_Age"] < 25:   score += 2
    if row["Driver_Age"] > 70:   score += 1
    if row["Driving_Exp"] < 5:   score += 2
    if row["Veh_Usage"] == "Business": score += 1
    miles_idx = ANNUAL_MILES_ORDER.index(row["Annual_Miles_Range"]) if row["Annual_Miles_Range"] in ANNUAL_MILES_ORDER else 3
    if miles_idx >= 5: score += 1
    if score <= 1:   return "Low"
    elif score <= 4: return "Medium"
    else:            return "High"


def _encode(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Veh_Usage_enc"] = df["Veh_Usage"].map(
        {v: i for i, v in enumerate(VEH_USAGE_ORDER)}).fillna(0).astype(int)
    df["Annual_Miles_enc"] = df["Annual_Miles_Range"].map(
        {v: i for i, v in enumerate(ANNUAL_MILES_ORDER)}).fillna(3).astype(int)
    return df


def train(df: pd.DataFrame):
    df = _encode(df)
    df["Risk_Tier"] = df.apply(_risk_label, axis=1)
    le = LabelEncoder()
    y  = le.fit_transform(df["Risk_Tier"])
    X  = df[FEATURES]
    clf = RandomForestClassifier(n_estimators=200, max_depth=8,
                                  class_weight="balanced", random_state=42, n_jobs=-1)
    clf.fit(X, y)
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    pickle.dump(clf, open(MODEL_PATH, "wb"))
    pickle.dump(le,  open(ENC_PATH,   "wb"))
    return clf, le


def load():
    return pickle.load(open(MODEL_PATH, "rb")), pickle.load(open(ENC_PATH, "rb"))


def predict(record: dict) -> dict:
    """
    record: dict with raw quote fields
    Returns: {"risk_tier": "Low|Medium|High", "risk_score": float, "confidence": float, "explanation": str}
    """
    clf, le = load()
    df  = pd.DataFrame([record])
    df  = _encode(df)
    # fill missing
    for col in ["Prev_Accidents","Prev_Citations","Driving_Exp","Driver_Age"]:
        if col not in df.columns: df[col] = 0
    X   = df[FEATURES]
    prob = clf.predict_proba(X)[0]
    idx  = np.argmax(prob)
    tier = le.inverse_transform([idx])[0]
    conf = float(prob[idx])

    # Risk score 0-100
    low_i  = list(le.classes_).index("Low")  if "Low"  in le.classes_ else -1
    high_i = list(le.classes_).index("High") if "High" in le.classes_ else -1
    risk_score = 0.0
    if low_i  >= 0: risk_score -= prob[low_i] * 30
    if high_i >= 0: risk_score += prob[high_i] * 100
    risk_score = max(0, min(100, 50 + risk_score))

    # Chain-of-thought explanation
    factors = []
    if record.get("Prev_Accidents", 0) > 0:
        factors.append(f"{record['Prev_Accidents']} prior accident(s) detected")
    if record.get("Prev_Citations", 0) > 0:
        factors.append(f"{record['Prev_Citations']} citation(s) on record")
    age = record.get("Driver_Age", 35)
    if age < 25: factors.append("Young driver (age < 25)")
    if age > 70: factors.append("Senior driver (age > 70)")
    exp = record.get("Driving_Exp", 10)
    if exp < 5:  factors.append("Limited driving experience (< 5 yrs)")
    if not factors:
        factors.append("No major risk indicators found")
    explanation = "Risk assessment: " + "; ".join(factors) + f". → Tier assigned: {tier}"

    return {
        "risk_tier": tier,
        "risk_score": round(risk_score, 2),
        "confidence": round(conf * 100, 2),
        "explanation": explanation
    }
