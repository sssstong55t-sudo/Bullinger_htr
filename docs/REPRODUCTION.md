# Local reproduction and evaluation

## Model inference and example

Install the root requirements and the default public TrOCR and mBERT checkpoints as documented in the [README setup](../README.md#setup-and-deployment). Then run from the repository root:

```sh
.venv-nlp/bin/python scripts/transcribe_images.py --images examples/611_01_r1l31.png --save-candidates --language la --output outputs/example_candidates.jsonl --device cpu
.venv-nlp/bin/python scripts/rerank_htr.py --input outputs/example_candidates.jsonl --output outputs/example_reranked.jsonl
.venv-nlp/bin/python scripts/extract_places.py --input outputs/example_reranked.jsonl --output outputs/example_places.jsonl
.venv-nlp/bin/python scripts/evaluate_htr.py --predictions outputs/example_reranked.jsonl --references examples/references.jsonl --output outputs/example_metrics.json
```

Fresh output paths are required. The sample is an already exposed public dataset line, not a new test. The edition/provisional transcription is illustrative, not independently certified manuscript gold. Expected artifacts under examples/ are actual local CPU outputs; model/library/device differences can affect decoding. Raw and reranked text may coincide for this short line. This is not a claim of improved corpus accuracy or verified geolocation.

The reranker uses eight returned beams, their sequence scores, char3 and word1 language statistics, smoothing 0.1 and fixed weights 0.03/0.03, matching the historical scoring formula. Input language must be known de/la; no test-reference text enters candidate selection. For mixed-language or unknown lines, use raw decoding rather than inventing a language. The evaluator compares literal strings without lowercasing, punctuation removal or modern spelling normalization. Historical worker reference loading uses strip(); supply consistently prepared reference strings.

## Whole-page inference with public Bullinger TrOCR

Run from the repository root after installing the root requirements. Set the input to a single manuscript page and choose fresh output paths.

```sh
.venv-nlp/bin/python scripts/download_public_trocr.py
.venv-nlp/bin/python scripts/page_segmentation_loose.py --image /path/to/page.jpg --output-dir outputs/page_lines
.venv-nlp/bin/python scripts/transcribe_images.py --images outputs/page_lines/line_*.png --output outputs/page_text.jsonl
.venv-nlp/bin/python scripts/extract_places.py --input outputs/page_text.jsonl --output outputs/page_places_mbert.jsonl
```

The splitter exports polygon-masked crops by default. Check `outputs/page_lines/preview.png` for missing or merged lines before transcribing. Install the October 22 checkpoint into `models/` using the [README setup instructions](../README.md#setup-and-deployment). For the GLiNER comparison, run `scripts/download_model.py` once to cache the weights, encoder configuration, and tokenizer, then run `scripts/extract_places.py` with `--gliner`, using a fresh output filename.

## Reproduce the 10-page evaluation

The [whole-page evaluation](../README.md#whole-page-evaluation) used pages `11511-1/2`, `12111-1/2`, `12476-1/2`, `12982-1/2`, and `13141-1/2`. The [evaluation package](../evaluation/whole_page_20261111/) contains the frozen page and crop manifest, reference text and place spans, all 300 TrOCR predictions, both sets of place predictions, and per-page and aggregate scores. The five source TEI letters are in [official_letters](../outputs/bullinger_htr_delivery_EN/data/official_letters/).

From the repository root, recompute CER, WER, place F1, and the 18-line seed44 comparison from the saved predictions. This step uses only the Python standard library and checks the results against the archived score reports.

```sh
python3 evaluation/whole_page_20261111/score.py
```

With the root dependencies installed, recompute the one-to-one contour matches at polygon IoU ≥ 0.50 from the frozen crop polygons and TEI reference regions. The historical rectangle comparison counts are recorded in `segmentation.json`.

```sh
.venv-nlp/bin/python evaluation/whole_page_20261111/verify_geometry.py
```

To regenerate the contours and line images, use the original scans from the manifest. The optional download retrieves missing scans from their source URLs. The script checks the scan hashes and compares the resulting polygons and crop hashes with the archived run.

```sh
.venv-nlp/bin/python evaluation/whole_page_20261111/verify_geometry.py \
  --images-dir /path/to/whole_page_scans --download --crop-dir /path/to/new_crops
```

Remove `--download` if the ten scans are already saved as `LETTER-PAGE.jpg` in the image directory. The metric replay uses the archived predictions. A fresh neural inference run uses the model setup and inference commands above. The replay retains the original exclusions and the provisional TEI-to-HTR span projection.

## Fixed manifests and training-only statistics

- data/splits/n20000_natural_s42.json: original portable 20k training selection.
- data/splits/evaluation.json: fixed 246 validation and 500 test line identities.
- data/reranker_char3_word1.json: statistics rebuilt only from those 20k training text files, using the original packaged training data; no validation/test text was read to build the LM.

The statistics record a training-text SHA256 and evaluation fingerprint. The original frozen Colab protocol digest has not been retrieved for an independent identity comparison. Therefore these artifacts restore the fixed recipe and selection, but do not certify bitwise equivalence to the prior GPU test or reproduce its published score by themselves. No full test was rerun during this update.

For local data preparation, obtain the existing dataset-baseline.zip and dataset-additional.zip input bundles from the project owner. Their source is the original TRAIN 0–2000 and TRAIN 2001–4000 blocks of pstroe/bullinger-htr. They are not included in GitHub and raw upstream split ZIPs have different internal paths. This is an explicit remaining dependency for historical training reproduction.

```sh
python3 scripts/prepare_htr_data.py --archives /path/dataset-baseline.zip /path/dataset-additional.zip --output-dir work/prepared_data --split train
python3 scripts/build_reranker.py --data-root work/prepared_data --output outputs/rebuilt_reranker.json
```

The preparation tool checks the frozen evaluation fingerprint and letter separation, extracts selected files without changing labels, and defaults to training data. Request validation or test explicitly only when needed. It does not train a model. The Colab workers still require their documented Drive folder structure, historical initialization weights and experiment records. Complete from-scratch reproduction on an arbitrary machine is not promised.

## Evaluation interface

Prediction JSONL requires line_id and text. Reference JSONL requires line_id, reference and language (de/la). IDs must match exactly; duplicates and missing predictions are errors. CER and WER are computed from corpus-level edit counts, overall and per language, and returned as ratios. Zero reference denominators return null. These are transcription metrics, not place extraction metrics.

## Checks and environment

```sh
python3 -m unittest discover -s tests -v
```

GitHub Actions runs metric/reranker, contour/rectangle segmentation, and place-candidate tests, plus Python syntax checks. `docs/TESTED_ENVIRONMENT.json` records the earlier CPU environment; it is not a complete dependency lockfile.

On November 14, 2026, the upload notebook was exercised in a Colab T4 session with Python 3.13, PyTorch 2.6.0+cu124, Transformers 4.51.0, and Accelerate 1.15.0 in an isolated environment. A missing Accelerate dependency was found and added to notebook setup. Subprocess logs now appear in the notebook while installation and inference run. The uploaded public line `01_611_01_r1l31.png` produced `Basileę ultimo iulii 1549.` and the Basel candidate (`l28`, latitude 47.55839, longitude 7.57327). The JSON downloaded through Chrome was opened and checked. A local whole-page run had also verified all 31 crops, their transcriptions, four place mentions, portable image paths, and ZIP contents on page `12982-1`.

The Colab trial used a line image, so it does not establish whole-page segmentation quality on Colab. The ZIP browser download was not independently verified. Browser output widgets can also be affected by expired Google login access or blocked third-party output scripts; these are separate from model inference. No new accuracy benchmark or universal browser/deployment compatibility is claimed.
