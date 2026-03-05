"""
Agent 3 — Premium Advisor (HYBRID)
For high-propensity unconverted quotes, reasons whether Quoted_Premium is the
conversion blocker. Recommends an adjusted premium band or alternative coverage
tier autonomously.

Input: Quoted_Premium, Coverage, Sal_Range, Vehicl_Cost_Range, Re_Quote, risk_tier, bind_probability
"""

import numpy as np
import pandas as pd
import pickle, os

# Salary range midpoints (USD)
SAL_MIDPOINTS = {
    "<= $ 25 K":          22500,
    "> $ 25 K <= $ 40 K": 32500,
    "> $ 40 K <= $ 60 K": 50000,
    "> $ 60 K <= $ 90 K": 75000,
    "> $ 90 K":           110000,
}
COVERAGE_DOWN = {"Enhanced": "Balanced", "Balanced": "Basic", "Basic": "Basic"}
COVERAGE_UP   = {"Basic": "Balanced", "Balanced": "Enhanced", "Enhanced": "Enhanced"}

VEHICLE_MIDPOINTS = {
    "<= $ 10 K":          8000,
    "> $ 10 K <= $ 20 K": 15000,
    "> $ 20 K <= $ 30 K": 25000,
    "> $ 30 K <= $ 40 K": 35000,
    "> $ 40 K":           50000,
}


def _premium_affordability(sal_range: str, quoted_premium: float) -> float:
    """Estimate what % of monthly income the premium represents (annualized)."""
    annual_income = SAL_MIDPOINTS.get(sal_range, 50000)
    return (quoted_premium / annual_income) * 100


def analyze(record: dict, risk_tier: str, bind_probability: float) -> dict:
    """
    Hybrid reasoning: rule-based affordability check + ML-calibrated suggestion.
    Returns recommendation dict.
    """
    quoted_premium = float(record.get("Quoted_Premium", 750))
    coverage       = record.get("Coverage", "Balanced")
    sal_range      = record.get("Sal_Range", "> $ 40 K <= $ 60 K")
    veh_cost_range = record.get("Vehicl_Cost_Range", "> $ 10 K <= $ 20 K")
    re_quote       = record.get("Re_Quote", "No")

    pct_of_income  = _premium_affordability(sal_range, quoted_premium)
    annual_income  = SAL_MIDPOINTS.get(sal_range, 50000)
    veh_cost       = VEHICLE_MIDPOINTS.get(veh_cost_range, 15000)

    # ── Rule-based reasoning ──────────────────────────────────────────────────
    is_blocker  = False
    reason_log  = []
    suggestions = []
    adjusted_premium = quoted_premium
    recommended_coverage = coverage

    # Affordability threshold: >2% of annual income → likely blocker
    if pct_of_income > 2.0:
        is_blocker = True
        reason_log.append(f"Premium ({quoted_premium:.0f}) = {pct_of_income:.1f}% of annual income — exceeds 2% affordability threshold")
        discount_pct   = min(15, (pct_of_income - 2.0) * 5)
        adjusted_premium = round(quoted_premium * (1 - discount_pct / 100), 2)
        suggestions.append(f"Reduce premium by {discount_pct:.0f}% → ${adjusted_premium:.0f}")

    # Coverage downgrade if high-risk + high-premium
    if risk_tier == "High" and coverage == "Enhanced" and pct_of_income > 1.5:
        is_blocker = True
        reason_log.append("Enhanced coverage for High-risk profile inflates premium unnecessarily")
        recommended_coverage = COVERAGE_DOWN[coverage]
        suggestions.append(f"Downgrade coverage to {recommended_coverage} to reduce risk-adjusted premium")

    # Premium vs vehicle cost reasonableness
    if quoted_premium > veh_cost * 0.08:
        is_blocker = True
        reason_log.append(f"Annual premium ({quoted_premium:.0f}) exceeds 8% of vehicle cost ({veh_cost:.0f}) — pricing appears misaligned")
        aligned_premium = round(veh_cost * 0.07, 2)
        adjusted_premium = min(adjusted_premium, aligned_premium)
        suggestions.append(f"Align premium to vehicle value → ${adjusted_premium:.0f}")

    # Re-quote + low bind → premium is almost certainly the blocker
    if re_quote == "Yes" and bind_probability < 45:
        is_blocker = True
        reason_log.append("Re-quote with low conversion probability — customer is price-sensitive and shopping around")
        adjusted_premium = round(adjusted_premium * 0.95, 2)
        suggestions.append("Apply loyalty/re-quote discount of 5%")

    if not is_blocker:
        reason_log.append("Premium appears competitively priced relative to income and vehicle cost")
        suggestions.append("No premium adjustment needed — focus on coverage communication")

    # Final adjusted premium band
    lower_band = round(adjusted_premium * 0.97, 2)
    upper_band = round(adjusted_premium * 1.03, 2)

    return {
        "is_premium_blocker": is_blocker,
        "original_premium": quoted_premium,
        "adjusted_premium": adjusted_premium,
        "recommended_band": f"${lower_band} – ${upper_band}",
        "recommended_coverage": recommended_coverage,
        "premium_pct_of_income": round(pct_of_income, 2),
        "reasoning": reason_log,
        "suggestions": suggestions,
        "explanation": (
            f"Premium Analysis: {'BLOCKER detected' if is_blocker else 'No blocker'} | "
            f"Adjusted premium: ${adjusted_premium:.0f} | "
            f"Coverage: {recommended_coverage}. "
            + " | ".join(reason_log[:2])
        )
    }
