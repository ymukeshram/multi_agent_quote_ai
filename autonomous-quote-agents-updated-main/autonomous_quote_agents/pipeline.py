"""
pipeline.py — Multi-Agent Orchestration Layer
Runs all 4 agents sequentially on a quote record, passing structured outputs between them.
Risk Profiler → Conversion Predictor → Premium Advisor → Decision Router
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agents.agent1_risk_profiler      import predict as risk_predict
from agents.agent2_conversion_predictor import predict as conv_predict
from agents.agent3_premium_advisor    import analyze  as premium_analyze
from agents.agent4_decision_router    import route


def run_pipeline(record: dict) -> dict:
    """
    Full 4-agent pipeline on a single quote record.
    Returns a serializable dict of all agent outputs + final decision.
    """
    quote_num = record.get("Quote_Num", "UNKNOWN")

    # ── Agent 1: Risk Profiler ────────────────────────────────────────────────
    risk_result = risk_predict(record)

    # ── Agent 2: Conversion Predictor ────────────────────────────────────────
    conv_result = conv_predict(record, risk_tier=risk_result["risk_tier"])

    # ── Agent 3: Premium Advisor ──────────────────────────────────────────────
    prem_result = premium_analyze(
        record,
        risk_tier=risk_result["risk_tier"],
        bind_probability=conv_result["bind_probability"]
    )

    # ── Agent 4: Decision Router ──────────────────────────────────────────────
    decision = route(quote_num, risk_result, conv_result, prem_result, record)

    return {
        "quote_num": quote_num,
        "agent1_risk": risk_result,
        "agent2_conversion": conv_result,
        "agent3_premium": prem_result,
        "agent4_decision": {
            "final_decision":    decision.final_decision,
            "decision_score":    decision.decision_score,
            "escalation_reason": decision.escalation_reason,
            "action_summary":    decision.action_summary,
            "chain_of_thought":  decision.chain_of_thought,
        }
    }
