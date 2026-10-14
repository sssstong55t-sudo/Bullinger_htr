# Bullinger HTR and Place Recognition

Recognize cropped manuscript line images with TrOCR, then extract place mentions from the recognized text with pretrained GLiNER and official-source historical name variants.

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
.venv-nlp/bin/python scripts/download_model.py
```

`download_model.py` downloads the pinned GLiNER snapshot once, requiring network access and approximately 1 GB of cache space. Subsequent place inference runs offline on CPU. A trained **TrOCR checkpoint must be supplied separately**: weights, model config, generation config, tokenizer and image-processor files must be together in the complete directory saved by `save_pretrained`. Keep the model files together; the downloader below installs the complete checkpoint.

Download the complete **seed44 DE20 inference checkpoint** from the [model release](https://github.com/sssstong55t-sudo/Bullinger_htr/releases/tag/v1.0.0-seed44-de20):

```sh
.venv-nlp/bin/python scripts/download_htr_checkpoint.py
```

The downloader fetches 20 Release parts, combines them, verifies the fixed archive SHA256 and every checkpoint file, and extracts `models/bullinger_checkpoint/`. Allow approximately 4 GB of free disk space for parts, archive and extracted weights. Verified temporary downloads remain under `models/checkpoint_download/` and can be deleted afterward. Do not delete the extracted checkpoint. The model archive is approximately 1.34 GB and excludes optimizer state. An actual single-image CPU smoke run passed. GLiNER is downloaded separately by `scripts/download_model.py`.


## Run the recognition pipeline

All commands below run from the repository root. Create cropped line images first; page segmentation is outside this pipeline. Use new output filenames for each run to preserve existing predictions.

### Step 1: Recognize manuscript lines

Provide cropped **line images**, not complete manuscript pages:

```sh
.venv-nlp/bin/python scripts/transcribe_images.py \
  --model-dir models/bullinger_checkpoint \
  --config configs/htr_inference.json \
  --images /absolute/path/line1.png /absolute/path/line2.png \
  --output outputs/team_htr.jsonl
```

`transcribe_images.py` loads your local TrOCR checkpoint, processes each image, generates text and records image identity. It automatically uses CUDA if available, otherwise CPU; `--device cpu` is available. The output file must not already exist.

The default wrapper uses raw eight-beam decoding. To enable local candidate reranking, add `--save-candidates --language de` (or `la`), then run `scripts/rerank_htr.py`. New-image outputs have no measured CER/WER until evaluated against explicit references. See [local reproduction and evaluation](docs/REPRODUCTION.md) for the complete example and provenance limits.

### Step 2: Extract place mentions

```sh
.venv-nlp/bin/python scripts/extract_places.py \
  --input outputs/team_htr.jsonl \
  --config configs/place_inference.json \
  --output outputs/team_places.jsonl
```

`extract_places.py` reads literal recognized text, runs GLiNER plus the two source-backed variant lexicons, and writes place spans. It emits only places; `person` and `organization` labels compete internally to reduce ambiguity. Text is not rewritten or modernized. Each line is processed independently; this wrapper does not reconstruct paragraph context across line boundaries. Historical variants and official-body results may not transfer to noisy HTR text.

Offsets are Python character offsets in the exact `text` string, with an exclusive end: `text[start:end]` equals the mention. No geographic coordinates or verified entity identities are generated. Overlapping candidates may remain; these are extraction candidates, not confirmed places.

### Input/output record example

```json
{"line_id":"example-line-1","image":"line1.png","text":"Example literal transcription","text_source":"htr_output"}
```

Input requires `line_id` and string `text`. The handwriting command produces both. Additional provenance fields are retained. Empty text is retained with an empty place list. For illustration, a mention object looks like:

```json
{"start":0,"end":6,"text":"Zurich","sources":["gliner"],"score":0.9}
```

This object is a schema example, not an actual scored prediction. A GLiNER score is not a calibrated probability of correctness. Alias-only matches have no GLiNER score.

## Entry points

| Command | Purpose | Input | Output |
|---|---|---|---|
| `scripts/download_model.py` | Cache the fixed pretrained GLiNER snapshot | Internet access | `.cache/huggingface/` |
| `scripts/download_htr_checkpoint.py` | Download, assemble and verify the trained TrOCR checkpoint | GitHub Release assets | `models/bullinger_checkpoint/` |
| `scripts/transcribe_images.py` | Recognize manuscript line images | Line images and TrOCR checkpoint | HTR text JSONL |
| `scripts/extract_places.py` | Extract literal place spans | Text JSONL | JSONL with a `places` array |
| Package `scripts/run_official_ner.py` | Reproduce the fixed official-text evaluation | Included reference corpus | Metrics, predictions and protocol |

For command-line options, run an entry point with `--help`. The download commands require network access. Once models are cached, the two inference commands and the official-text evaluation run locally offline. This is a command-line pipeline; no web server or API service is deployed by these commands.

## Configuration and output files

| File | Meaning | Used for |
|---|---|---|
| `configs/htr_inference.json` | Beam count, generation length and cache settings | New-image TrOCR command; does not contain weights |
| Checkpoint `config.json`, `generation_config.json`, tokenizer and processor JSON files | Model architecture and preprocessing metadata | Must accompany real weights inside `--model-dir` |
| `configs/place_inference.json` | Pinned GLiNER revision, labels, threshold, windowing and lexicon paths | Text-to-place command |
| `outputs/bullinger_htr_delivery_EN/data/historical_aliases.json` | Official-source historical variants | Adds literal name variants missed by GLiNER |
| `outputs/bullinger_htr_delivery_EN/data/source_aliases.json` | Additional disjoint source-derived variants | Second lexicon layer |
| `outputs/team_htr.jsonl` | One recognized line per record | Bridge from handwriting recognition to place extraction |
| `outputs/team_places.jsonl` | Text, provenance and detected place spans | Team integration output |
| Package `results/summary.json` | Saved official annotation-agreement metrics | Reporting only, not an inference configuration |
| Package `results/predictions.jsonl` | Recorded official-paragraph predictions | Experiment evidence, not new-image output |
| Package `data/reference.jsonl` | Fixed official reference paragraphs and spans | Evaluation only; never insert its reference text into HTR inference |
| Package `manifest.json` | SHA256 checksums | Package integrity, not model selection |

JSONL stores one JSON object per line. The two configuration files control inference; output JSONL files carry predictions between stages. Checkpoint metadata and saved evaluation records serve separate purposes.

## Reproduce the official-text evaluation

```sh
.venv-nlp/bin/python outputs/bullinger_htr_delivery_EN/scripts/run_official_ner.py \
  --output-dir outputs/official_ner_run01
```

This separate command evaluates **official reference text**, not images or your team's new HTR outputs. It reports all predefined branches against fixed published place spans. Always use a new output directory.

| Evaluation | CER | WER | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|
| Historical frozen HTR test, 500 lines, LM reranked | 13.3917% | 38.6088% | — | — | — |
| Official annotation agreement, 357 paragraphs, GLiNER + variants | — | — | 82.99% | 61.16% | 70.42% |

These are separate experiments, **not image-to-place end-to-end metrics**. Official annotation completeness is unestablished; unmatched predictions are not automatically true false positives. This official sample is already exposed; further tuning on it is development work. See [fixed comparisons and limitations](outputs/bullinger_htr_delivery_EN/RESULTS.md).

## Reproduce HTR experiments in Colab

Open [Bullinger_Clean_Colab.ipynb](outputs/bullinger_htr_delivery_EN/Bullinger_Clean_Colab.ipynb) and read [TRAINING.md](outputs/bullinger_htr_delivery_EN/TRAINING.md) first. The notebook defaults to `TASK=None`; explicitly select one task. Existing Drive datasets, checkpoints and saved candidates are required. It is not a self-contained from-scratch run.

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

Current delivery contains English documentation, inference/experiment code, fixed official data, actual results and provenance. Original German/Latin sources are preserved. Personal review tables, raw image archives, caches and virtual environments are excluded. TrOCR inference weights are distributed as a Release asset rather than in the Git code tree. Obsolete legacy entry points have been removed from the current branch; their earlier versions remain in Git history.

The original packaged official evaluation was fully reproduced offline. Both new team wrappers passed syntax and CLI checks. The text-to-place wrapper completed an offline two-record smoke run including an empty line. The released seed44 DE20 checkpoint was loaded successfully and used for a single-image CPU inference smoke check. No new independent accuracy result is claimed. Dependency versions are not a full cross-platform lockfile. A fresh computer must install dependencies, download the HTR release asset and cache GLiNER. Raw beam inference is distinct from the historical LM-reranked test.

## Local reranking, example and HTR evaluation

[REPRODUCTION.md](docs/REPRODUCTION.md) provides a runnable public-line example, the optional candidate reranker, corpus-level overall/de/la CER/WER evaluation, frozen split manifests, training-only statistics rebuilding and checks. The original Colab input image bundles are still required for historical training reproduction; they are not embedded in this repository.
