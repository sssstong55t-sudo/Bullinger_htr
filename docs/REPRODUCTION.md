# Local reproduction and evaluation

## Model inference and example

Install root requirements, run scripts/download_model.py and scripts/download_htr_checkpoint.py as documented in README.md. Then run from the repository root:

```sh
.venv-nlp/bin/python scripts/transcribe_images.py --model-dir models/bullinger_checkpoint --images examples/611_01_r1l31.png --save-candidates --language la --output outputs/example_candidates.jsonl --device cpu
.venv-nlp/bin/python scripts/rerank_htr.py --input outputs/example_candidates.jsonl --output outputs/example_reranked.jsonl
.venv-nlp/bin/python scripts/extract_places.py --input outputs/example_reranked.jsonl --output outputs/example_places.jsonl
.venv-nlp/bin/python scripts/evaluate_htr.py --predictions outputs/example_reranked.jsonl --references examples/references.jsonl --output outputs/example_metrics.json
```

Fresh output paths are required. The sample is an already exposed public dataset line, not a new test. The edition/provisional transcription is illustrative, not independently certified manuscript gold. Expected artifacts under examples/ are actual local CPU outputs; model/library/device differences can affect decoding. Raw and reranked text may coincide for this short line. This is not a claim of improved corpus accuracy or verified geolocation.

The reranker uses eight returned beams, their sequence scores, char3 and word1 language statistics, smoothing 0.1 and fixed weights 0.03/0.03, matching the historical scoring formula. Input language must be known de/la; no test-reference text enters candidate selection. For mixed-language or unknown lines, use raw decoding rather than inventing a language. The evaluator compares literal strings without lowercasing, punctuation removal or modern spelling normalization. Historical worker reference loading uses strip(); supply consistently prepared reference strings.

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

GitHub Actions runs standard-library metric/reranker tests and Python syntax checks. docs/TESTED_ENVIRONMENT.json records the actual CPU environment used; it is not a complete dependency lockfile. A clean, isolated fresh-install end-to-end run has not been performed. Existing-environment image inference, eight-candidate generation, fixed reranking, evaluation and place extraction were exercised. No fresh accuracy benchmark is claimed.
