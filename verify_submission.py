"""Run this file in PyCharm to validate the submitted results without API calls.

This entry never loads config.py or .env and never constructs an LLM or embedding
model. It checks the saved evidence, then prints a compact presentation summary.
"""
import json
from pathlib import Path

from reproduction.verify_results import verify

RUN_NAME = "locomo_window40_20261108"


def main():
    print("SimpleMem submission evidence check", flush=True)
    print("Offline validation only: no API calls and no key loading.", flush=True)
    result = verify(RUN_NAME)
    directory = Path(__file__).resolve().parent / "reproduction" / "results" / RUN_NAME
    plan = json.loads((directory / "plan.json").read_text(encoding="utf-8"))
    summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    print(json.dumps(result, indent=2))
    print("\nFixed questions (original QA indices):")
    for question in plan["questions"]:
        print(f"  QA {question['qa_index']:>3}, category {question['category']}: {question['question']}")
    print("\nArm             Mean F1 (%)   Trial SD (pp)   Mean input tokens")
    for arm, metrics in summary["arms"].items():
        print(f"{arm:<16}{metrics['mean_f1'] * 100:>10.2f}"
              f"{metrics['std_f1'] * 100:>16.2f}{metrics['mean_input_tokens']:>20,.1f}")
        print("  Trial F1 (%): " + ", ".join(f"{score * 100:.2f}" for score in metrics["trial_f1"]))
    baseline = summary["arms"]["baseline_k25"]
    modified = summary["arms"]["modified_k10"]
    saving = 100 * (1 - modified["mean_input_tokens"] / baseline["mean_input_tokens"])
    delta = 100 * (modified["mean_f1"] - baseline["mean_f1"])
    print(f"\nObserved mean F1 difference: {delta:+.2f} percentage points")
    print(f"Observed mean input-token reduction: {saving:.1f}%")
    print("Scope: 8 distinct questions, 3 query trials, one frozen 441-entry bank.")
    print("Similar observed mean lexical F1; larger observed trial variability for k10.")
    print("This is not proof of preserved factual accuracy or a full paper replication.")


if __name__ == "__main__":
    main()
