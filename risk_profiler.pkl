"""
app.py — Flask Web Application
Autonomous Quote Agents — Operations Dashboard
Run: python app.py  (after training models first)
"""

import sys, os, json, pickle, random
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from flask import Flask, render_template, request, jsonify
import pandas as pd

app = Flask(__name__)
DATA_PATH  = os.path.join(os.path.dirname(__file__), "data/quotes.csv")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")

# ── Load data once at startup ─────────────────────────────────────────────────
_df = None
def get_df():
    global _df
    if _df is None:
        _df = pd.read_csv(DATA_PATH)
    return _df

def get_stats():
    sp = os.path.join(MODELS_DIR, "training_stats.pkl")
    if os.path.exists(sp):
        return pickle.load(open(sp, "rb"))
    return {}


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    stats = get_stats()
    return render_template("index.html", stats=stats)


@app.route("/api/run_pipeline", methods=["POST"])
def run_pipeline_api():
    """Run the 4-agent pipeline on submitted quote data."""
    from pipeline import run_pipeline
    data = request.json
    try:
        result = run_pipeline(data)
        return jsonify({"success": True, "result": result})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})


@app.route("/api/random_quote")
def random_quote():
    """Return a random quote record from the dataset."""
    df = get_df()
    row = df.sample(1).iloc[0].to_dict()
    # Clean NaN
    for k, v in row.items():
        if pd.isna(v): row[k] = ""
    return jsonify(row)


@app.route("/api/batch_sample")
def batch_sample():
    """Run pipeline on 10 random quotes and return aggregated stats."""
    from pipeline import run_pipeline
    df = get_df()
    sample = df.sample(20).to_dict(orient="records")
    results = []
    decisions = {"Auto-Approve": 0, "Agent Follow-Up": 0, "Escalate to Underwriter": 0}
    risk_counts = {"Low": 0, "Medium": 0, "High": 0}
    bind_probs = []

    for rec in sample:
        for k, v in rec.items():
            if pd.isna(v): rec[k] = ""
        try:
            r = run_pipeline(rec)
            d = r["agent4_decision"]["final_decision"]
            decisions[d] = decisions.get(d, 0) + 1
            rt = r["agent1_risk"]["risk_tier"]
            risk_counts[rt] = risk_counts.get(rt, 0) + 1
            bind_probs.append(r["agent2_conversion"]["bind_probability"])
            results.append({
                "quote_num": rec.get("Quote_Num", ""),
                "region": rec.get("Region", ""),
                "agent_type": rec.get("Agent_Type", ""),
                "risk_tier": rt,
                "bind_prob": r["agent2_conversion"]["bind_probability"],
                "decision": d,
                "premium": rec.get("Quoted_Premium", ""),
            })
        except Exception as e:
            pass

    avg_bind = round(sum(bind_probs)/len(bind_probs), 1) if bind_probs else 0
    return jsonify({
        "quotes": results,
        "decisions": decisions,
        "risk_counts": risk_counts,
        "avg_bind_prob": avg_bind,
    })


@app.route("/api/stats")
def api_stats():
    return jsonify(get_stats())


if __name__ == "__main__":
    print("\n" + "="*55)
    print("  🚀 AUTONOMOUS QUOTE AGENTS — DASHBOARD")
    print("  Open: http://127.0.0.1:5000")
    print("="*55 + "\n")
    app.run(debug=True, port=5000)
