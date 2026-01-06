import json
from collections import Counter, defaultdict
from typing import List

# Heuristic categories (not exhaustive ontologies)
ENTITY_LIKE = {"company", "person", "city", "country", "organization", "founder", "owner"}
ABSTRACTION_LIKE = {"plan", "summary", "abstract", "general", "concept", "context", "intermediate"}
FILLER_LIKE = {"the", "a", "an", "of", "in", "to", "and", "or", "was", "were", "is", "are"}


def classify_churn(tokens_added: List[str], tokens_removed: List[str]):
    """Bucket token churn into coarse heuristic categories."""
    buckets = []
    if any(t in ENTITY_LIKE for t in tokens_removed):
        buckets.append("entity_drop")
    if any(t in ABSTRACTION_LIKE for t in tokens_added):
        buckets.append("abstraction_flip")
    if any(t in FILLER_LIKE for t in tokens_added + tokens_removed):
        buckets.append("filler_churn")
    if not buckets:
        buckets.append("other")
    return buckets


def aggregate_reports(report_paths: List[str]):
    summary = {
        "runs": len(report_paths),
        "alerts_total_hops": 0,  # counts hop-level alerts
        "alerts_by_question_hops": defaultdict(int),  # hop-level alerts per question
        "fail_with_alert_questions": 0,  # question-level outcomes when any hop alerted
        "pass_with_alert_questions": 0,
        "churn_patterns": Counter(),
    }

    for path in report_paths:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for trace in data.get("traces", []):
            q = trace.get("question", "")[:120]
            f1 = trace.get("f1", 0.0)
            em = trace.get("em", 0.0)
            passed = (f1 == 1.0 and em == 1.0)  # adjust if partial credit should count as pass
            question_had_alert = False
            for h in trace.get("hops", []):
                dm = h.get("debug_metadata", {}) if isinstance(h, dict) else {}
                vel = dm.get("velocity", {}) if isinstance(dm, dict) else {}
                if vel.get("oscillation_alert"):
                    summary["alerts_total_hops"] += 1
                    summary["alerts_by_question_hops"][q] += 1
                    question_had_alert = True
                    tokens_added = [t.lower() for t in vel.get("oscillation_tokens_added", []) or []]
                    tokens_removed = [t.lower() for t in vel.get("oscillation_tokens_removed", []) or []]
                    for bucket in classify_churn(tokens_added, tokens_removed):
                        summary["churn_patterns"][bucket] += 1
            if question_had_alert:
                if passed:
                    summary["pass_with_alert_questions"] += 1
                else:
                    summary["fail_with_alert_questions"] += 1

    # Convert defaultdicts to normal dicts for clean output
    summary["alerts_by_question_hops"] = dict(summary["alerts_by_question_hops"])
    summary["churn_patterns"] = dict(summary["churn_patterns"])
    return summary


if __name__ == "__main__":
    # Example usage; replace with your actual report paths
    REPORTS = [
        "results/diffusion_debug_report_seed1.json",
        "results/diffusion_debug_report_seed2.json",
        "results/diffusion_debug_report_seed3.json",
    ]
    summary = aggregate_reports(REPORTS)
    print("Alerts total (hops):", summary["alerts_total_hops"])
    print("Alerts by question (hop counts):", summary["alerts_by_question_hops"])
    print(
        "Questions with alerts → fail:",
        summary["fail_with_alert_questions"],
        "pass:",
        summary["pass_with_alert_questions"],
    )
    print("Churn patterns (heuristic buckets):", summary["churn_patterns"])

