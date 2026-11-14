# Bullinger HTR and Place Recognition

Split manuscript pages into line images, transcribe them with TrOCR, and extract place mentions with fine-tuned multilingual BERT and official-source name variants, with GLiNER available as an option.

## Sources and attribution

- **Manuscript images and HTR dataset:** [pstroe/bullinger-htr](https://github.com/pstroe/bullinger-htr). This is the original image/line-transcription dataset repository, not this project's trained checkpoint. See [dataset documentation](docs/DATASET_README.md) and [dataset license](docs/DATASET_LICENSE).
- **Official edited text and place annotations:** [stazh/bullinger-korpus-tei](https://github.com/stazh/bullinger-korpus-tei), fixed commit `dc9fb5290c210023a6a9182e8e3102c1f6dff734`. Attribution: Scheurer, Müller, Schroffenegger, Ströbel and Volk (2025), *Bullinger Digital Briefkorpus*, University of Zurich. Source text license: CC BY-NC-ND 3.0 CH as stated upstream. XML is preserved unchanged; analysis is separate.
- **Recognition models:** TrOCR through Hugging Face Transformers; `urchade/gliner_multi-v2.1`, fixed revision `443d26d654e0324125a96bebd8e796c14ff2efe6`. GLiNER is pretrained, not fine-tuned in this project.

Dataset licenses and model terms apply separately. Do not assume all repository content has a permissive software license.

## Setup and deployment

Use Python 3.10 on macOS or Linux with a virtual environment. CPU execution is supported; a CUDA GPU is optional for faster handwriting recognition. On Windows, use WSL or adapt the Python executable paths to your virtual environment.

Clone the repository and install the dependencies:

```sh
git clone https://github.com/sssstong55t-sudo/Bullinger_htr.git
cd Bullinger_htr
python3 -m venv .venv-nlp
.venv-nlp/bin/python -m pip install -r requirements.txt
.venv-nlp/bin/python scripts/download_public_trocr.py
mkdir -p models
curl --fail --location --output models/bullinger_mbert_20261022.zip https://github.com/sssstong55t-sudo/Bullinger_htr/releases/download/v1.0.0-mbert-20261022/bullinger_mbert_20261022.zip
unzip models/bullinger_mbert_20261022.zip -d models
```

The [October 22 mBERT checkpoint ZIP](https://github.com/sssstong55t-sudo/Bullinger_htr/releases/download/v1.0.0-mbert-20261022/bullinger_mbert_20261022.zip) (about 680 MiB) includes the model and tokenizer used in the whole-page evaluation. Archive SHA-256 is `92cd0da3f00234a9045831ed5ca70c6c7b43d8b310fc681b1df104ed12a88683`.

`download_public_trocr.py` installs the default [public Bullinger TrOCR checkpoint](https://huggingface.co/pstroe/bullinger-general-model/tree/be32c883a4deb812ab6f461ae79f583c3f9edc53) and its tokenizer. Allow about 2.2 GB for the download. The installer verifies the weights against the evaluation's SHA-256. Transcription runs offline after installation.

## Run a whole page with one Python command

After setup, use the standalone script from the repository root:

```sh
.venv-nlp/bin/python scripts/run_page_pipeline.py \
  --image /absolute/path/letter.jpg \
  --output-dir outputs/letter_complete \
  --page-id letter-001 --device auto
```

This runs contour segmentation, the public Bullinger TrOCR checkpoint, mBERT with historical aliases, and modern-place candidate lookup. Use a new output directory each time. `--device auto` chooses CUDA when available, otherwise CPU; use `--device mps` explicitly on a supported Mac. It does not install packages or download models during recognition.

Open `outputs/letter_complete/report.html` to see line images next to the transcription and candidate table. Give `page_response.json` to the web backend, or share `bullinger_page_results.zip`. The ZIP includes the normalized page image, segmentation preview, all line crops, JSON/JSONL predictions, plain transcription, and an HTML report that opens offline. Image paths are relative to the ZIP root after extraction. Candidate IDs, names, nullable coordinates, ambiguous alternatives, and unmatched mentions are preserved.

A fresh 31-line whole-page run and an actual browser ZIP download passed the [artifact checks](docs/REPRODUCTION.md#standalone-python-and-downloaded-zip-check). The script and its messages are in English. See [team integration requirements](docs/TEAM_INTEGRATION.md) before connecting it to the shared website. This command produces results; the website still needs a background worker, result storage, and an API/frontend connection.

## Run the recognition pipeline

All commands below run from the repository root. Use new output paths for each run to preserve existing predictions.

### Step 0: Split a full page into line images (optional)

Give `scripts/page_segmentation_loose.py` a scan of one manuscript page. The splitter runs locally with Pillow, NumPy and SciPy. TrOCR and place recognition weights are only needed for the subsequent recognition steps.

```sh
.venv-nlp/bin/python scripts/page_segmentation_loose.py \
  --image /absolute/path/letter.jpg \
  --output-dir outputs/letter_lines
```

The default mode estimates page tilt, traces each line's upper and lower edges, and fills pixels outside its polygon with white. Use `--no-contours` for rectangular crops. Both modes expect a new output directory.

- `line_0001.png`, `line_0002.png`, … are numbered from top to bottom.
- `manifest.json` records the line order, image filenames and `[left, top, right, bottom]` boxes in the EXIF-oriented page. Right and bottom are exclusive. Contour mode also records each `polygon`.
- `preview.png` marks the detected lines on the page. Check it for split, merged or spurious lines, especially on pages with marginal notes, multiple columns or tightly spaced handwriting.

To transcribe the exported lines in order on macOS or Linux:

```sh
.venv-nlp/bin/python scripts/transcribe_images.py \
  --images outputs/letter_lines/line_*.png \
  --output outputs/letter_htr.jsonl
```

### Step 1: Recognize manuscript lines

Provide cropped **line images**, not complete manuscript pages:

```sh
.venv-nlp/bin/python scripts/transcribe_images.py \
  --images /absolute/path/line1.png /absolute/path/line2.png \
  --output outputs/team_htr.jsonl
```

`transcribe_images.py` loads the installed public Bullinger TrOCR checkpoint by default, processes each image, generates text and records image identity. It automatically uses CUDA if available, otherwise CPU; `--device cpu` is available. The output file must not already exist.

The default wrapper uses raw eight-beam decoding with a 128-token generation limit. To enable local candidate reranking, add `--save-candidates --language de` (or `la`), then run `scripts/rerank_htr.py`. New-image outputs have no measured CER/WER until evaluated against explicit references. See [local reproduction and evaluation](docs/REPRODUCTION.md) for the complete example and provenance limits.

### Step 2: Extract place mentions

```sh
.venv-nlp/bin/python scripts/extract_places.py \
  --input outputs/team_htr.jsonl \
  --config configs/place_inference.json \
  --output outputs/team_places.jsonl
```

`extract_places.py` defaults to the October 22 mBERT checkpoint with both historical alias lexicons and whole-word filtering. The classifier was fine-tuned on official TEI body `placeName` spans. It runs offline on CPU, reads the literal transcribed text line by line, and writes place spans. To use a different local mBERT checkpoint, pass its directory after `--mbert`.

Offsets are Python character offsets in the exact `text` string, with an exclusive end, so `text[start:end]` equals the mention. Each mention includes `official_place_candidates` from the bundled historical alias lexicons. Candidates include the official Bullinger place ID, canonical name, source, and any latitude and longitude available in the source record. A missing name or coordinate is returned as JSON `null`. The coordinates refer to the source record and do not certify the identity of the historical mention.

#### Optional GLiNER place recognition

Run `scripts/download_model.py` once to download the pinned GLiNER weights and mDeBERTa encoder configuration and tokenizer (about 1 GB). Then select GLiNER explicitly:

```sh
.venv-nlp/bin/python scripts/download_model.py
.venv-nlp/bin/python scripts/extract_places.py --gliner \
  --input outputs/letter_htr.jsonl \
  --output outputs/letter_gliner_places.jsonl
```

The fixed official-text evaluation below runs GLiNER and uses this same download.

### Input/output record example

```json
{"line_id":"example-line-1","image":"line1.png","text":"Example literal transcription","text_source":"htr_output"}
```

Input requires `line_id` and string `text`. The handwriting command produces both. Additional provenance fields are retained. Empty text is retained with an empty place list. The [saved example](examples/expected_places.jsonl) includes this candidate for `Basileę`:

```json
{
  "start": 0,
  "end": 7,
  "text": "Basileę",
  "sources": ["mbert", "official_source_variant"],
  "place_link_status": "candidate_for_review",
  "official_place_candidates": [
    {
      "official_place_id": "l28",
      "canonical_name": "Basel",
      "latitude": 47.55839,
      "longitude": 7.57327,
      "source": "https://github.com/stazh/bullinger-korpus-tei"
    }
  ]
}
```

- `candidate_for_review` means the lexicons returned one candidate.
- `ambiguous` means the lexicons returned multiple distinct candidates. Each stays in the array.
- `unmatched` means the mention has no candidate in the bundled lexicons. Its candidate array is empty.

Model scores are uncalibrated. Alias-only mentions have no model score. The same candidate fields are returned by both place extractors.

## Entry points

| Command | Purpose | Input | Output |
|---|---|---|---|
| `scripts/download_model.py` | Cache optional GLiNER and its encoder tokenizer and configuration | Internet access | `.cache/huggingface/` |
| `scripts/download_public_trocr.py` | Cache the fixed public Bullinger TrOCR and tokenizer | Hugging Face | `models/bullinger_public_trocr/` |
| `scripts/run_page_pipeline.py` | Run the complete default pipeline and package results | Page image and installed models | Portable JSON, images, offline report and ZIP |
| `scripts/page_segmentation_loose.py` | Split one manuscript page into ordered lines | Page image | Line PNGs, manifest and preview |
| `scripts/transcribe_images.py` | Recognize manuscript line images | Line images and TrOCR checkpoint | HTR text JSONL |
| `scripts/extract_places.py` | Extract literal place spans | Text JSONL | JSONL with a `places` array |
| Package `scripts/run_official_ner.py` | Reproduce the fixed official-text evaluation | Included reference corpus | Metrics, predictions and protocol |

For command-line options, run an entry point with `--help`. The download commands require network access. Once models are cached, the two inference commands and the official-text evaluation run locally offline. This is a command-line pipeline; no web server or API service is deployed by these commands.

## Configuration and output files

| File | Meaning | Used for |
|---|---|---|
| `configs/htr_inference.json` | Beam count, generation length and cache settings | Default public inference enables the generation cache. An explicit `--config` uses the values in that file. |
| Checkpoint configuration, tokenizer and processor files | Model architecture and preprocessing metadata | Installed with the public checkpoint |
| `configs/place_inference.json` | Shared lexicon paths and optional GLiNER settings | Text-to-place command |
| `outputs/bullinger_htr_delivery/data/historical_aliases.json` | Official-source historical variants | Adds literal name variants missed by GLiNER |
| `outputs/bullinger_htr_delivery/data/source_aliases.json` | Additional disjoint source-derived variants | Second lexicon layer |
| `outputs/team_htr.jsonl` | One recognized line per record | Bridge from handwriting recognition to place extraction |
| `outputs/team_places.jsonl` | Text, provenance and detected place spans | Team integration output |
| Package `results/summary.json` | Saved official annotation-agreement metrics | Reporting only, not an inference configuration |
| Package `results/predictions.jsonl` | Recorded official-paragraph predictions | Experiment evidence, not new-image output |
| Package `data/reference.jsonl` | Fixed official reference paragraphs and spans | Evaluation only; never insert its reference text into HTR inference |
| Package `manifest.json` | SHA256 checksums | Package integrity, not model selection |

JSONL stores one JSON object per line. The two configuration files control inference; output JSONL files carry predictions between stages. Checkpoint metadata and saved evaluation records serve separate purposes.

## Reproduce the official-text evaluation

```sh
.venv-nlp/bin/python outputs/bullinger_htr_delivery/scripts/run_official_ner.py \
  --output-dir outputs/official_ner_run01
```

This separate command evaluates **official reference text**, not images or your team's new HTR outputs. It reports all predefined branches against fixed published place spans. Always use a new output directory.

| Evaluation | CER | WER | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|
| Historical frozen HTR test, 500 lines, LM reranked | 13.3917% | 38.6088% | — | — | — |
| Official annotation agreement, 357 paragraphs, GLiNER + variants | — | — | 82.99% | 61.16% | 70.42% |

These are separate experiments, **not image-to-place end-to-end metrics**. Official annotation completeness is unestablished; unmatched predictions are not automatically true false positives. This official sample is already exposed; further tuning on it is development work. See [fixed comparisons and limitations](outputs/bullinger_htr_delivery/RESULTS.md).

## Whole-page evaluation

The evaluation used 10 original manuscript page images from five letters, previously used for the contour comparison. Each contour crop was transcribed with public Bullinger TrOCR at revision `be32c883a4deb812ab6f461ae79f583c3f9edc53`. The two place extractors processed the same 300 transcriptions. The [evaluation package](evaluation/whole_page/) contains the frozen inputs, predictions, and per-page counts. See [reproduction commands](docs/REPRODUCTION.md#reproduce-the-10-page-evaluation) to recompute the scores.

| Stage | Comparison | Result |
| --- | --- | --- |
| Segmentation | One-to-one polygon IoU ≥ 0.50 against 290 TEI reference regions | Contour crops matched 251 of 300 outputs (83.67% precision, 86.55% recall, 85.08% F1). The original rectangle and deskewed rectangle methods matched 214 and 219 regions, respectively. |
| Public TrOCR | Matched lines | 18.94% CER, 35.03% WER |
| Public TrOCR | Whole pages, charging missing lines as deletions and extra lines as insertions | 36.28% CER, 53.33% WER |
| GLiNER + historical aliases | Exact place span and LOC category | 70.21% precision, 58.93% recall, 64.08% F1 (33 TP, 14 FP, 23 FN) |
| mBERT + historical aliases | Exact place span and LOC category | 72.73% precision, 85.71% recall, 78.69% F1 (48 TP, 18 FP, 8 FN) |

On a fixed subset of 18 byte-identical contour crops with usable references, seed44 DE20 scored 19.19% CER and 45.95% WER. Public TrOCR scored 14.34% CER and 36.49% WER. The seed44 comparison is limited to that subset.

The references come from the official edited TEI. Transcription scoring excluded 11 regions without reliable corresponding text. Place scoring also excluded three lines with cross-line locations and used 56 eligible place annotations. Reference spans were automatically projected onto the HTR text, with no manual verification of the projected boundaries. These figures measure annotation agreement for the released mBERT checkpoint with historical aliases on this five-letter sample.

## Reproduce HTR experiments in Colab

Open [Bullinger_Clean_Colab.ipynb](outputs/bullinger_htr_delivery/Bullinger_Clean_Colab.ipynb) and read [TRAINING.md](outputs/bullinger_htr_delivery/TRAINING.md) first. The notebook defaults to `TASK=None`; explicitly select one task. Existing Drive datasets, checkpoints and saved candidates are required. It is not a self-contained from-scratch run.

| Worker under package `training/` | Purpose |
|---|---|
| `train_10k_sampling.py` | Historical natural/balanced 10k training and recovery |
| `train_20k.py` | Historical 20k training and recovery |
| `adapt_validation.py` | Natural/DE20 matched 500-step adaptation, seeds 43–45 |
| `rerank_validation.py` | Candidate reranking using train-text-only language statistics |
| `evaluate_frozen_test.py` | Frozen beam8/8-candidate test, character/word weights 0.03/0.03 |
| `context_validation.py` | Saved-candidate contextual reranking ablation |

Use validation for choices. Do not tune on the test set or rerun a final test just to find a favorable score. Release only your own completed Colab runtime.

## Repository scope and verification

Current delivery contains English documentation, inference/experiment code, fixed official data, actual results and provenance. Original German/Latin sources are preserved. Personal review tables, raw image archives, caches and virtual environments are excluded. The default TrOCR weights are downloaded from Hugging Face. Obsolete legacy entry points have been removed from the current branch; their earlier versions remain in Git history.

The original packaged official evaluation was fully reproduced offline. Both new team wrappers passed syntax and CLI checks. The text-to-place wrapper completed an offline two-record smoke run including an empty line. The default public checkpoint was verified with an offline CPU inference smoke check. No new independent accuracy result is claimed. Dependency versions are not a full cross-platform lockfile. A fresh computer must install dependencies and obtain the models selected for inference. Raw beam inference is distinct from the historical LM-reranked test.

## Local reranking, example and HTR evaluation

[REPRODUCTION.md](docs/REPRODUCTION.md) provides a runnable public-line example, the optional candidate reranker, corpus-level overall/de/la CER/WER evaluation, frozen split manifests, training-only statistics rebuilding and checks. The original Colab input image bundles are still required for historical training reproduction; they are not embedded in this repository.
