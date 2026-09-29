# Bullinger HTR: ML & NLP

TrOCR fine-tuning and evaluation on German and Latin Bullinger manuscript lines, with an initial gazetteer-based NLP pipeline.

## Project contents

- `train_small.py`, `train_medium.py`, `overfit_test.py`: training experiments.
- `compare_small_model.py`, `run_test_set.py`: OCR evaluation.
- `results/`: saved configurations/tokenizers, medium training history and final evaluation results. Model weights are excluded.
- `medium_split.json`: existing experiment split for traceability. Check local paths before reuse.
- `nlp_baseline.py`, `run_nlp_demo.py`, `test_nlp.py`: NLP baseline, synthetic demo and tests.
- `export_transcriptions.py`, `export_annotation_csv.py`: preparation for data review.
- `nlp_results/`: reference exports, predictions, annotation worksheet and synthetic demo results. These do not constitute a completed manually verified evaluation set.

## Progress and next steps

TrOCR training is completed. Evaluation compares pretrained and fine-tuned recognition on unseen German/Latin samples using CER. Generalization remains inconsistent. Some ground-truth transcriptions may contain alignment or annotation errors, so high CER can reflect both OCR and reference quality.

Next: inspect representative/high-CER image–transcription pairs and build a manually verified evaluation subset; reviewing the entire dataset is unnecessary. NLP validation depends on transcription quality and should continue using the verified subset. The gazetteer baseline is not a trained historical NER model. Synthetic demo scores must not be presented as real model performance.

## Run the NLP demo

Python 3.10+; no third-party dependencies:

```sh
python3 run_nlp_demo.py
python3 -m unittest -v test_nlp.py
```

See `README_NLP.md` for formats and evaluation limitations.

## OCR setup

The experiment scripts are preserved unchanged. Several contain an absolute local `ROOT` path: adjust it to your dataset directory before running. Scripts expect the original extracted dataset layout and evaluation may require local checkpoints. Training scripts execute at top level; do not import them for configuration inspection.

`requirements-ml.txt` lists inferred direct dependencies, not a tested or pinned environment. Full training and OCR evaluation have not been rerun for this repository preparation.

## Data, weights and attribution

Original dataset: https://github.com/pstroe/bullinger-htr

Follow upstream documentation to obtain and extract the data. The original dataset README and license are retained in `docs/` for attribution; the dataset license is not a newly assigned license for the team's code.

Full datasets, archives, model weights, original Git/LFS history, caches and IDE settings are excluded. Checkpoints remain in the original local project; no shared checkpoint download location has been configured yet.
