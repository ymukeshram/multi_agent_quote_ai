"""
Agent 4 — Decision Router (ESCALATE-ONLY)
Combines all upstream agent outputs into one of:
  • Auto-Approve    — High confidence, low risk, high conversion
  • Agent Follow-Up — Moderate signals, premium adjustment recommended
  • Escalate        — Low confidence OR complex risk profile → human underwriter

Only the last bucket ever touches a human.
"""

from dataclasses import dataclass
from typing import Literal

DECISION_LABELS = Literal["Auto-Approve", "Agent Follow-Up", "Escalate to Underwriter"]


@dataclass
class PipelineResult:
    quote_num: str
    risk_tier: str
    risk_score: float
    risk_confidence: float
    bind_probability: float
    conversion_category: str
    is_premium_blocker: bool
    adjusted_premium: float
    recommended_coverage: str
    final_decision: str
    decision_score: float
    escalation_reason: str
    action_summary: str
    chain_of_thought: list[str]


# ── Confidence thresholds ─────────────────────────────────────────────────────
AUTO_APPROVE_BIND_MIN     = 35.0   # bind prob >= 35% (lowered from 45%)
AUTO_APPROVE_RISK_MAX     = 45.0   # risk score <= 45 (raised from 40)
AUTO_APPROVE_CONF_MIN     = 70.0   # agent 1 confidence >= 70% (lowered from 75%)

ESCALATE_RISK_MIN         = 70.0   # risk score >= 70 → always escalate
ESCALATE_BIND_MAX         = 20.0   # bind prob <= 20% AND risk is high
ESCALATE_CONF_MAX         = 50.0   # agent 1 confidence <= 50% → uncertain → escalate


def route(
    quote_num: str,
    risk_result: dict,
    conversion_result: dict,
    premium_result: dict,
    record: dict,
) -> PipelineResult:
    """
    Orchestrates all agent outputs into a final routing decision.
    """
    risk_tier       = risk_result["risk_tier"]
    risk_score      = risk_result["risk_score"]
    risk_conf       = risk_result["confidence"]
    bind_prob       = conversion_result["bind_probability"]
    conv_cat        = conversion_result["conversion_category"]
    is_blocker      = premium_result["is_premium_blocker"]
    adj_premium     = premium_result["adjusted_premium"]
    rec_coverage    = premium_result["recommended_coverage"]
    region          = record.get("Region", "?")
    agent_type      = record.get("Agent_Type", "?")

    cot = []  # chain-of-thought trace

    # ── Step 1: Safety / uncertainty check ───────────────────────────────────
    cot.append(f"[Agent 1] Risk Tier={risk_tier}, Score={risk_score}, Confidence={risk_conf}%")
    cot.append(f"[Agent 2] Bind Probability={bind_prob}%, Category={conv_cat}")
    cot.append(f"[Agent 3] Premium Blocker={is_blocker}, Adjusted=${adj_premium:.0f}, Coverage={rec_coverage}")

    # ── Step 2: Decision logic ────────────────────────────────────────────────
    escalation_reason = ""
    decision_score    = 0.0

    # Escalate conditions (any one sufficient)
    must_escalate = False
    if risk_score >= ESCALATE_RISK_MIN:
        must_escalate = True
        escalation_reason = f"Risk score {risk_score} ≥ {ESCALATE_RISK_MIN} threshold — high-risk profile requires underwriter review"
        cot.append(f"[Router] ⚠️  ESCALATE: {escalation_reason}")
    elif risk_conf <= ESCALATE_CONF_MAX:
        must_escalate = True
        escalation_reason = f"Agent 1 confidence {risk_conf}% ≤ {ESCALATE_CONF_MAX}% — uncertain risk profile, deferring to human"
        cot.append(f"[Router] ⚠️  ESCALATE: {escalation_reason}")
    elif bind_prob <= ESCALATE_BIND_MAX and risk_tier == "High":
        must_escalate = True
        escalation_reason = f"Bind probability {bind_prob}% with High risk — marginal case requiring underwriter judgment"
        cot.append(f"[Router] ⚠️  ESCALATE: {escalation_reason}")

    if must_escalate:
        decision = "Escalate to Underwriter"
        decision_score = round(100 - bind_prob, 2)
        action = (
            f"🔴 Escalate to Underwriter | Region {region} | Agent {agent_type} | "
            f"Risk: {risk_tier} ({risk_score}) | Bind prob: {bind_prob}% | "
            f"Reason: {escalation_reason}"
        )

    # Auto-Approve conditions (all must be met)
    elif (bind_prob >= AUTO_APPROVE_BIND_MIN and
          risk_score <= AUTO_APPROVE_RISK_MAX and
          risk_conf >= AUTO_APPROVE_CONF_MIN and
          not is_blocker):
        decision = "Auto-Approve"
        decision_score = round((bind_prob + (100 - risk_score) + risk_conf) / 3, 2)
        cot.append(f"[Router] ✅ AUTO-APPROVE: High bind prob + Low risk + No premium blocker")
        action = (
            f"✅ Auto-Approve Policy | Quoted Premium: ${record.get('Quoted_Premium', 'N/A')} | "
            f"Coverage: {record.get('Coverage', 'N/A')} | Bind prob: {bind_prob}% | Risk: {risk_tier}"
        )

    # Agent Follow-Up (middle ground)
    else:
        decision = "Agent Follow-Up"
        decision_score = round(bind_prob, 2)
        follow_actions = []
        if is_blocker:
            follow_actions.append(f"Offer adjusted premium ${adj_premium:.0f} (band: {premium_result['recommended_band']})")
        if rec_coverage != record.get("Coverage"):
            follow_actions.append(f"Propose coverage change to {rec_coverage}")
        if not follow_actions:
            follow_actions.append("Reach out to customer with current quote details and address objections")
        cot.append(f"[Router] 🔵 AGENT FOLLOW-UP: Moderate signals — human agent should follow up")
        action = (
            f"🔵 Agent Follow-Up Required | Actions: {' | '.join(follow_actions)} | "
            f"Bind prob: {bind_prob}% | Risk: {risk_tier} | Region {region}"
        )

    cot.append(f"[Router] FINAL DECISION: {decision} (Score={decision_score})")

    return PipelineResult(
        quote_num=quote_num,
        risk_tier=risk_tier,
        risk_score=risk_score,
        risk_confidence=risk_conf,
        bind_probability=bind_prob,
        conversion_category=conv_cat,
        is_premium_blocker=is_blocker,
        adjusted_premium=adj_premium,
        recommended_coverage=rec_coverage,
        final_decision=decision,
        decision_score=decision_score,
        escalation_reason=escalation_reason,
        action_summary=action,
        chain_of_thought=cot,
    )
