"""Run the synthetic NLP demo without command-line parameters."""
from datetime import datetime
from pathlib import Path
import subprocess
import sys


def main():
    root = Path(__file__).resolve().parent
    output = root / "nlp_results" / datetime.now().strftime("demo_%Y%m%d_%H%M%S_%f")
    output.mkdir(parents=True)
    for kind in ("reference", "htr"):
        prediction = output / f"{kind}_predictions.jsonl"
        subprocess.run([
            sys.executable, str(root / "nlp_baseline.py"), "extract",
            "--input", str(root / f"demo_{kind}.jsonl"),
            "--gazetteer", str(root / "gazetteer.example.json"),
            "--output", str(prediction),
        ], check=True)
        subprocess.run([
            sys.executable, str(root / "nlp_baseline.py"), "evaluate",
            "--gold", str(root / "demo_reference.jsonl"),
            "--predictions", str(prediction), "--mode", "places",
            "--output", str(output / f"{kind}_place_scores.json"),
        ], check=True)
    print("NLP demo completed. Synthetic examples only, not actual HTR results.")
    print(f"Results: {output}")


if __name__ == "__main__":
    main()
