import csv
import json
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
INPUT = ROOT / "nlp_results" / "bullinger_reference.jsonl"
OUTPUT = ROOT / "nlp_results" / (
    f"annotation_{datetime.now():%Y%m%d_%H%M%S_%f}.csv"
)

# Export all records by default.
# Set this to 20 to export only the first 20 records.
LIMIT = None


def main():
    if not INPUT.exists():
        raise FileNotFoundError(f"Reference file not found: {INPUT}")

    rows = [
        json.loads(line)
        for line in INPUT.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    if LIMIT is not None:
        rows = rows[:LIMIT]

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    # UTF-8 with BOM helps Excel display multilingual text correctly.
    with OUTPUT.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([
            "Record Number",
            "Document ID",
            "Page ID",
            "Line ID",
            "Language",
            "Reference Text",
            "Image Path",
            "Review Status",
            "Place Mentions (Original Text)",
            "Resolved Places",
            "Notes",
        ])

        for index, row in enumerate(rows, start=1):
            writer.writerow([
                index,
                row.get("document_id", ""),
                row.get("page_id", ""),
                row.get("line_id", ""),
                row.get("language", ""),
                row.get("text", ""),
                row.get("image_path", ""),
                "Not reviewed",
                "",
                "",
                "",
            ])

    print(f"Exported {len(rows)} records.")
    print(f"CSV file: {OUTPUT}")


if __name__ == "__main__":
    main()