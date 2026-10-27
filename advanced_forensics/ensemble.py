"""
advanced_forensics.ensemble
===========================

Transparent ensemble of forensic signals.

Rules (deliberately conservative):
- Only signals with ``available=True`` participate; weights of missing
  signals are dropped and the remainder renormalised (reported).
- The transformer signal participates only when locally available.
- The experimental rPPG signal is EXCLUDED by default and joins only
  with ``include_rppg=True`` (small weight, capped contribution).
- Verdict bands: score >= 0.60 -> FAKE, score <= 0.40 -> REAL, else
  UNCERTAIN.  These are decision-support bands, not calibrated cutoffs.
- Confidence NEVER exceeds reliability, and reliability combines
  signal coverage with inter-signal agreement.  Nothing here inflates
  confidence: borderline or single-signal cases stay low-confidence
  and/or UNCERTAIN.

Each signal dict: ``{"available": bool, "score": float|None, ...}``
with score in [0, 1] meaning "more suspicious / fake-like".
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np

FAKE_BAND = 0.60
REAL_BAND = 0.40

BASE_WEIGHTS = {
    "xception": 0.45,
    "landmark": 0.10,
    "motion": 0.10,
    "frequency": 0.15,
    "temporal": 0.15,
    "transformer": 0.05,
    "rppg": 0.00,  # experimental: excluded unless explicitly included
}
RPPG_OPT_IN_WEIGHT = 0.05
# Hard ceiling on the experimental signal's renormalised share.
RPPG_MAX_NORM = 0.05


def _num(score) -> Optional[float]:
    try:
        v = float(score)
    except (TypeError, ValueError):
        return None
    return v if np.isfinite(v) else None


def combine(signals: Dict[str, dict],
            include_rppg: bool = False) -> dict:
    """Fuse available signals -> {score, verdict, confidence, ...}."""
    weights = dict(BASE_WEIGHTS)
    if include_rppg:
        weights["rppg"] = RPPG_OPT_IN_WEIGHT

    used, total_w = {}, 0.0
    for name, w in weights.items():
        if float(w) <= 0.0:
            continue  # opted-out/experimental-off: stays excluded
        sig = (signals or {}).get(name) or {}
        s = _num(sig.get("score")) if sig.get("available") else None
        if s is None:
            continue
        s = min(1.0, max(0.0, s))
        used[name] = {"score": s, "weight": float(w)}
        total_w += float(w)

    if not used:
        return {"available": False, "score": None, "verdict": "UNCERTAIN",
                "confidence": 0.0, "reliability": 0.0,
                "reason": "no forensic signals available",
                "used": {}, "excluded": sorted(weights)}

    # Renormalise over participating signals (transparently reported).
    norm = {k: v["weight"] / total_w for k, v in used.items()}
    if "rppg" in norm and norm["rppg"] > RPPG_MAX_NORM:
        # Hard cap on experimental influence regardless of coverage.
        overflow = norm["rppg"] - RPPG_MAX_NORM
        norm["rppg"] = RPPG_MAX_NORM
        rest = sum(w for k, w in norm.items() if k != "rppg")
        if rest > 0:
            for k in norm:
                if k != "rppg":
                    norm[k] += overflow * (norm[k] / rest)
    score = float(sum(used[k]["score"] * norm[k] for k in used))

    # Agreement: 1 - weighted std mapped to [0, 1] (higher = more united).
    vals = np.array([used[k]["score"] for k in used])
    wts = np.array([norm[k] for k in used])
    mean = float(np.average(vals, weights=wts))
    spread = float(np.sqrt(np.average((vals - mean) ** 2, weights=wts)))
    agreement = float(max(0.0, 1.0 - spread * 2.5))

    # Coverage: participating weight vs the full non-experimental budget.
    full = sum(v for k, v in BASE_WEIGHTS.items() if k != "rppg")
    coverage = float(min(1.0, total_w / full))
    reliability = float(coverage * (0.35 + 0.65 * agreement))

    if score >= FAKE_BAND:
        verdict = "FAKE"
    elif score <= REAL_BAND:
        verdict = "REAL"
    else:
        verdict = "UNCERTAIN"

    # Confidence is reliability-scaled distance from the unsure middle,
    # floored so unanimous-but-mild evidence is not overstated, and
    # capped by reliability (never inflated past it).
    distance = abs(score - 0.5) * 2.0
    confidence = float(min(reliability, reliability * (0.25 + 0.75 * distance)))
    confidence = round(max(0.0, confidence), 4)

    excluded = [k for k in weights if k not in used]
    notes = [f"renormalised over {len(used)} signal(s): "
             + ", ".join(f"{k}={norm[k]:.2f}" for k in sorted(norm))]
    if excluded:
        notes.append("excluded (unavailable/opted-out): "
                     + ", ".join(sorted(excluded)))
    if "rppg" in used:
        notes.append(f"experimental rPPG share capped at {RPPG_MAX_NORM:.2f}.")
    if verdict == "UNCERTAIN":
        notes.append("score inside the unsure band: collect more evidence, "
                     "do not decide on this alone.")

    return {"available": True, "score": round(score, 4), "verdict": verdict,
            "confidence": confidence, "reliability": round(reliability, 4),
            "agreement": round(agreement, 4), "coverage": round(coverage, 4),
            "used": {k: {"score": used[k]["score"],
                         "weight": round(norm[k], 4)} for k in used},
            "excluded": sorted(excluded), "notes": notes}
