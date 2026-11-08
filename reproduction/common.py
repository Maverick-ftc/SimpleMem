"""Deterministic subset selection and the pinned SimpleMem evaluator's F1."""
import hashlib
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "reproduction" / "data" / "locomo10.json"


def load_data():
    manifest = json.loads((DATA_PATH.parent / "provenance.json").read_text(encoding="utf-8"))
    raw = DATA_PATH.read_bytes()
    if hashlib.sha256(raw).hexdigest() != manifest["sha256"]:
        raise ValueError("Dataset checksum mismatch")
    return json.loads(raw), manifest


def select_questions(data, sample_indices, per_category=3, seed=5660):
    """Select without looking at answers or model outputs; exclude category 5."""
    rng = random.Random(seed)
    rows = []
    for sample_index in sample_indices:
        sample = data[sample_index]
        for category in (1, 2, 3, 4):
            eligible = [i for i, qa in enumerate(sample["qa"])
                        if qa["category"] == category and qa.get("answer") is not None]
            for qa_index in sorted(rng.sample(eligible, min(per_category, len(eligible)))):
                qa = sample["qa"][qa_index]
                rows.append({"sample_index": sample_index, "sample_id": sample.get("sample_id"),
                             "qa_index": qa_index, "category": category,
                             "question": qa["question"], "answer": str(qa["answer"]),
                             "evidence": qa.get("evidence", [])})
    return rows


def upstream_f1(prediction, reference):
    """Match test_locomo10.py: lowercase, four punctuation replacements, token sets.

    This is deliberately NOT substituted with the different LoCoMo bag-of-words F1.
    """
    if not prediction or not reference:
        return 0.0

    def tokens(value):
        text = str(value).strip().lower()
        for punctuation in ".,!?":
            text = text.replace(punctuation, " ")
        return set(text.split())

    predicted, expected = tokens(prediction), tokens(reference)
    if not predicted or not expected:
        return 0.0
    common = len(predicted & expected)
    precision, recall = common / len(predicted), common / len(expected)
    return 2 * precision * recall / (precision + recall) if common else 0.0


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)
