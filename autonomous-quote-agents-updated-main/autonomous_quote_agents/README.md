# Autonomous Quote Agents — Hackathon Project

A multi-agent AI system that autonomously processes auto insurance quotes end-to-end:
profiling risk, predicting conversion, negotiating premiums, and routing decisions
**without requiring a human in the loop for every step**.

---

## Project Structure

```
autonomous_quote_agents/
├── data/
│   └── quotes.csv              ← Dataset (146,259 records)
├── models/                     ← Trained model files (auto-generated)
├── agents/
│   ├── agent1_risk_profiler.py        ← Agent 1: RandomForest risk tier
│   ├── agent2_conversion_predictor.py ← Agent 2: GBM bind probability
│   ├── agent3_premium_advisor.py      ← Agent 3: Hybrid premium reasoning
│   └── agent4_decision_router.py      ← Agent 4: Decision routing
├── templates/
│   └── index.html              ← Full web dashboard
├── pipeline.py                 ← Multi-agent orchestration
├── train_models.py             ← Train all ML models
├── app.py                      ← Flask web server
├── requirements.txt
└── README.md
```

---

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the ML models
```bash
python train_models.py
```
This trains Agent 1 (Risk Profiler) and Agent 2 (Conversion Predictor) on the full dataset.
**Run this once before starting the app.**

### 3. Start the web server
```bash
python app.py
```

### 4. Open the dashboard
```
http://127.0.0.1:5000
```

---

## The 4-Agent Pipeline

| Agent | Name | Type | Model |
|-------|------|------|-------|
| 1 | Risk Profiler | FULLY AUTO | RandomForest (200 trees, class_weight=balanced) |
| 2 | Conversion Predictor | FULLY AUTO | GradientBoosting (300 trees, sample_weight for 78/22 imbalance) |
| 3 | Premium Advisor | HYBRID | Rule-based affordability + domain thresholds |
| 4 | Decision Router | ESCALATE-ONLY | Confidence threshold routing |

### Pipeline Flow
```
Quote Record
    ↓
[Agent 1] Risk Profiler → risk_tier, risk_score, confidence
    ↓
[Agent 2] Conversion Predictor → bind_probability, conversion_category
    ↓
[Agent 3] Premium Advisor → adjusted_premium, recommendations
    ↓
[Agent 4] Decision Router → Auto-Approve | Agent Follow-Up | Escalate
```

---

## Decision Thresholds

### Auto-Approve 
- Bind probability ≥ 65%
- Risk score ≤ 40
- Agent 1 confidence ≥ 75%
- No premium blocker detected

### Escalate to Underwriter 
- Risk score ≥ 70
- Agent 1 confidence ≤ 50%
- Bind prob ≤ 20% with High risk tier

### Agent Follow-Up 
- Everything else (moderate signals)

---
## Dataset Statistics
- **146,259** quote records
- **22%** bind rate (severe class imbalance — handled via sample weighting)
- **8 Regions** (A–H)
- **2 Agent Types**: EA (Exclusive) and IA (Independent)
- **25 Features** per record

---

## Explainability

Every decision is explainable via:
1. **Chain-of-Thought Traces** — step-by-step log per agent (visible in dashboard)
2. **Feature Attribution** — plain-English top drivers for bind probability
3. **Domain-Rule Audit Trail** — Agent 3 documents every rule it fires

---

## Dependencies
```
flask>=2.3.0
scikit-learn>=1.3.0
pandas>=2.0.0
numpy>=1.24.0
```
