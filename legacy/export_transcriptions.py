"""Export existing medium_split.json TXT references; does not load or change TrOCR."""
import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--split-file", required=True)
    p.add_argument("--split", choices=["train", "validation", "test"], default="validation")
    p.add_argument("--output", required=True)
    args = p.parse_args()
    data = json.loads(Path(args.split_file).read_text(encoding="utf-8"))
    rows = []
    for sample in data[args.split]:
        image = Path(sample["image"])
        parts = image.stem.split("_", 2)
        if len(parts) != 3:
            raise ValueError(f"Unexpected filename: {image.name}")
        letter, page, line = parts
        rows.append({"document_id": letter, "page_id": letter + "_" + page,
                     "line_id": image.stem, "language": sample["language"],
                     "text": Path(sample["text"]).read_text(encoding="utf-8").strip(),
                     "image_path": str(image), "line_bbox": None,
                     "text_source": "dataset_reference"})
    with Path(args.output).open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Exported {len(rows)} reference lines. No coordinates inferred; split unchanged.")


if __name__ == "__main__":
    main()
