"""
train_models.py — Train all ML models on the provided dataset.
Run this ONCE before starting the web app:
    python train_models.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
import pickle
import time

DATA_PATH = os.path.join(os.path.dirname(__file__), "data/quotes.csv")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")

def main():
    print("="*60)
    print("  AUTONOMOUS QUOTE AGENTS — MODEL TRAINING")
    print("="*60)
    os.makedirs(MODELS_DIR, exist_ok=True)

    print(f"\n📂 Loading data from {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH)
    print(f"   ✓ Loaded {len(df):,} records with {df.shape[1]} columns")
    print(f"   Bind rate: {(df['Policy_Bind']=='Yes').mean()*100:.1f}%")

    # ── Train Agent 1: Risk Profiler ─────────────────────────────────────────
    print("\n🤖 Training Agent 1 — Risk Profiler...")
    t0 = time.time()
    from agents.agent1_risk_profiler import train as train_risk
    clf1, le1 = train_risk(df.copy())
    print(f"   ✓ Trained RandomForest (200 trees) in {time.time()-t0:.1f}s")
    print(f"   ✓ Risk classes: {list(le1.classes_)}")

    # ── Train Agent 2: Conversion Predictor ──────────────────────────────────
    print("\n🤖 Training Agent 2 — Conversion Predictor...")
    t0 = time.time()
    from agents.agent2_conversion_predictor import train as train_conv
    clf2 = train_conv(df.copy())
    print(f"   ✓ Trained GradientBoosting (300 trees, class-weighted) in {time.time()-t0:.1f}s")

    # ── Save training statistics ──────────────────────────────────────────────
    stats = {
        "total_records": len(df),
        "bind_rate": round((df['Policy_Bind']=='Yes').mean()*100, 2),
        "ea_pct": round((df['Agent_Type']=='EA').mean()*100, 2),
        "regions": sorted(df['Region'].unique().tolist()),
        "coverage_dist": df['Coverage'].value_counts().to_dict(),
        "risk_dist": {},
    }
    # Compute risk dist
    from agents.agent1_risk_profiler import _encode, _risk_label
    df_enc = _encode(df.copy())
    df_enc["Risk_Tier"] = df_enc.apply(_risk_label, axis=1)
    stats["risk_dist"] = df_enc["Risk_Tier"].value_counts().to_dict()
    stats["avg_premium"] = round(df["Quoted_Premium"].mean(), 2)
    stats["ea_bind_rate"] = round((df[df["Agent_Type"]=="EA"]["Policy_Bind"]=="Yes").mean()*100, 2)
    stats["ia_bind_rate"] = round((df[df["Agent_Type"]=="IA"]["Policy_Bind"]=="Yes").mean()*100, 2)

    pickle.dump(stats, open(os.path.join(MODELS_DIR, "training_stats.pkl"), "wb"))

    print("\n" + "="*60)
    print("  ✅ ALL MODELS TRAINED SUCCESSFULLY")
    print(f"  Models saved to: {MODELS_DIR}/")
    print("="*60)
    print("\n▶  Now run:  python app.py")


if __name__ == "__main__":
    main()
