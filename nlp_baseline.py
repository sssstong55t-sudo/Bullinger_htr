"""Inspectable gazetteer baseline for Bullinger. Python 3.10+, standard library only."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import re
import unicodedata


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def fold(text):
    # Character-wise normalization preserves a mapping back to ORIGINAL offsets.
    # No fuzzy spelling correction, accent removal or silent OCR repair.
    output, starts, ends = [], [], []
    for i, char in enumerate(text):
        normalized = unicodedata.normalize("NFKC", char).casefold()
        output.append(normalized)
        starts.extend([i] * len(normalized))
        ends.extend([i + 1] * len(normalized))
    return "".join(output), starts, ends


class GazetteerNLP:
    def __init__(self, gazetteer):
        self.version = gazetteer["version"]
        self.places = {}
        self.aliases = defaultdict(set)
        for place in gazetteer["places"]:
            pid = place["id"]
            if pid in self.places:
                raise ValueError(f"Duplicate place id: {pid}")
            if not -90 <= place["latitude"] <= 90 or not -180 <= place["longitude"] <= 180:
                raise ValueError(f"Invalid coordinates: {pid}")
            self.places[pid] = place
            for alias in place["aliases"]:
                key = fold(alias)[0]
                if not key.strip():
                    raise ValueError("Empty alias")
                self.aliases[key].add(pid)
        # Longest matching alias wins at each position; word boundaries avoid
        # matching Bern within Bernhard. This is NOT contextual named-entity recognition.
        terms = sorted(self.aliases, key=lambda s: (-len(s), s))
        self.pattern = re.compile(r"(?<!\w)(?:" + "|".join(re.escape(s) for s in terms) + r")(?!\w)") if terms else None

    def extract(self, row):
        for field in ("document_id", "page_id", "line_id", "text"):
            if not isinstance(row.get(field), str):
                raise ValueError(f"{field} must be a string")
        text = row["text"]
        folded, starts, ends = fold(text)
        mentions = []
        for match in self.pattern.finditer(folded) if self.pattern else []:
            start, end = starts[match.start()], ends[match.end() - 1]
            ids = sorted(self.aliases[match.group()])
            candidates = [self.places[pid] for pid in ids]
            unique = len(ids) == 1
            mentions.append({
                "mention_id": f"{row['document_id']}:{row['page_id']}:{row['line_id']}:{start}:{end}",
                "type": "PLACE", "text": text[start:end],
                "start": start, "end": end,
                "matched_alias": match.group(),
                "method": "gazetteer_exact_casefold",
                "confidence": None,
                "resolution_status": "unique_in_gazetteer" if unique else "ambiguous",
                # A unique lookup is a proposal, NOT a historically verified resolution.
                "place_id": ids[0] if unique else None,
                "needs_review": True,
                "candidates": [{k: p[k] for k in ("id", "canonical_name", "latitude", "longitude", "source")} for p in candidates],
                "evidence": {
                    "document_id": row["document_id"], "page_id": row["page_id"],
                    "line_id": row["line_id"], "image_path": row.get("image_path"),
                    "line_bbox": row.get("line_bbox"),
                    "bbox_scope": "line" if row.get("line_bbox") is not None else None,
                },
            })
        return {**row, "nlp": {"method": "gazetteer-v1", "gazetteer_version": self.version}, "mentions": mentions}


def identity(row):
    return tuple(row[k] for k in ("document_id", "page_id", "line_id"))


def index_rows(rows):
    result = {}
    for row in rows:
        key = identity(row)
        if key in result:
            raise ValueError(f"Duplicate line identity: {key}")
        result[key] = row
    return result


def metrics(tp, fp, fn):
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall,
            "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0}


def evaluate(gold, predictions, mode):
    gi, pi = index_rows(gold), index_rows(predictions)
    if gi.keys() != pi.keys():
        raise ValueError("Gold and predictions must contain exactly the same line identities")
    counts = defaultdict(lambda: [0, 0, 0])
    excluded_unresolved = 0
    for key, row in gi.items():
        pred = pi[key]
        if "gold_mentions" not in row:
            raise ValueError("Annotate gold_mentions explicitly, including [] for negative lines")
        if mode == "span" and row["text"] != pred["text"]:
            raise ValueError("Span evaluation requires identical input text; use places for HTR comparison")
        for mention in row["gold_mentions"]:
            if not 0 <= mention["start"] < mention["end"] <= len(row["text"]):
                raise ValueError(f"Invalid gold offsets: {key}")
        if mode == "span":
            expected = Counter((m["start"], m["end"]) for m in row["gold_mentions"])
            actual = Counter((m["start"], m["end"]) for m in pred["mentions"])
        else:
            # Compare canonical mention counts WITHIN each line, not character offsets.
            # OCR changes offsets; repeated mentions must still count individually.
            if any(m.get("place_id") is None for m in row["gold_mentions"]):
                excluded_unresolved += 1
                continue  # exclude whole line rather than arbitrarily count false positives
            expected = Counter(m["place_id"] for m in row["gold_mentions"])
            actual = Counter(m["place_id"] for m in pred["mentions"] if m.get("place_id") is not None)
        tp = sum((expected & actual).values())
        fp, fn = sum(actual.values()) - tp, sum(expected.values()) - tp
        for group in ("overall", "language:" + row.get("language", "unknown")):
            for i, value in enumerate((tp, fp, fn)):
                counts[group][i] += value
    return {"mode": mode, "input_lines": len(gi), "excluded_unresolved_gold_lines": excluded_unresolved,
            "evaluated_lines": len(gi) - excluded_unresolved,
            "definition": "Exact entity spans" if mode == "span" else "End-to-end canonical place mention counts per line; NOT span NER F1 or resolution accuracy",
            "scores": {group: metrics(*values) for group, values in counts.items()}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    extract = sub.add_parser("extract")
    extract.add_argument("--input", required=True)
    extract.add_argument("--gazetteer", required=True)
    extract.add_argument("--output", required=True)
    ev = sub.add_parser("evaluate")
    ev.add_argument("--gold", required=True)
    ev.add_argument("--predictions", required=True)
    ev.add_argument("--mode", choices=["span", "places"], required=True)
    ev.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.command == "extract":
        model = GazetteerNLP(read_json(args.gazetteer))
        rows = read_jsonl(args.input)
        index_rows(rows)
        with Path(args.output).open("w", encoding="utf-8") as f:
            for row in rows:
                # Gold annotations are deliberately never consumed or copied into predictions.
                clean = {k: v for k, v in row.items() if k != "gold_mentions"}
                f.write(json.dumps(model.extract(clean), ensure_ascii=False) + "\n")
    else:
        write_json(args.output, evaluate(read_jsonl(args.gold), read_jsonl(args.predictions), args.mode))


if __name__ == "__main__":
    main()
