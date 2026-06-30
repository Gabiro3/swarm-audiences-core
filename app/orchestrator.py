"""One entry point, one unified verdict, for any clip.

Ported from the notebook's Cell 6 `moderate()`: frontrunner triage decides
PUBLISH/BLOCK outright when confident, otherwise escalates to the Qwen3-VL
deep auditor and maps its verdict to the final decision.
"""

from . import auditor, frontrunner

FINAL_FROM_PHASE1 = {"AUTO-PASS": "PUBLISH", "AUTO-BLOCK": "BLOCK"}
FINAL_FROM_AUDIT = {"SAFE": "PUBLISH", "VIOLATION": "BLOCK", "REVIEW": "REVIEW"}


def moderate(video_path: str) -> dict:
    fr = frontrunner.frontrunner_triage(video_path)
    if fr["verdict"] in FINAL_FROM_PHASE1:
        return {
            "final": FINAL_FROM_PHASE1[fr["verdict"]],
            "decided_by": "phase1",
            "frontrunner": fr,
            "audit": None,
        }
    audit = auditor.deep_audit(video_path, fr)  # GRAY_ZONE_ESCALATE
    final = FINAL_FROM_AUDIT.get(audit.get("safety_status", "REVIEW"), "REVIEW")
    return {
        "final": final,
        "decided_by": "phase2",
        "frontrunner": fr,
        "audit": audit,
    }
