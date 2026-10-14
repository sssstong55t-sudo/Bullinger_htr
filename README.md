# Bullinger HTR and Official Place Recognition

Current delivery: 14 October 2026. [中文使用说明](README_ZH.md).

This repository delivers a fixed official-text place recognition evaluation using pretrained GLiNER plus official-source historical name variants, and preserved Colab HTR training/reranking workers. No GLiNER fine-tuning is claimed.

| Evaluation | CER | WER | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|
| Historical frozen HTR test (500 lines) | 13.3917% | 38.6088% | — | — | — |
| Official place annotation agreement (357 paragraphs) | — | — | 82.99% | 61.16% | 70.42% |

These are separate experiments, not image-to-place end-to-end metrics. Official annotation completeness is not established; unmatched predictions are not automatically true false positives. Strict WER<37% and precision>=85% targets have not been met. The official sample is now exposed; further tuning on it is development work.

## Install and run

Use Python 3.10 or a compatible environment. From this repository root:

```sh
python3 -m venv .venv-nlp
.venv-nlp/bin/python -m pip install -r requirements.txt
.venv-nlp/bin/python scripts/download_model.py
.venv-nlp/bin/python outputs/bullinger_htr_delivery_EN/scripts/run_official_ner.py --output-dir outputs/official_ner_run01
```

The download requires network access and about 1 GB of cache space. Evaluation uses the pinned local snapshot on CPU; no GPU purchase is necessary. Always choose a fresh output directory. New environments may introduce numerical differences; recorded results were reproduced in the original environment. Dependencies are not a lockfile for every platform.

## Delivery contents

- [Package instructions](outputs/bullinger_htr_delivery_EN/README.md) and [Chinese package guide](outputs/bullinger_htr_delivery_EN/README_ZH.md).
- [Result definitions and fixed comparisons](outputs/bullinger_htr_delivery_EN/RESULTS.md).
- `outputs/bullinger_htr_delivery_EN/data/`: official reference paragraphs, unchanged XML, selection/audit and disjoint source-derived variant lexicons.
- `outputs/bullinger_htr_delivery_EN/results/`: saved predictions, protocol, metrics and miss diagnosis.
- [Colab training instructions](outputs/bullinger_htr_delivery_EN/TRAINING.md) and [clean notebook](outputs/bullinger_htr_delivery_EN/Bullinger_Clean_Colab.ipynb). Existing Drive data and checkpoints are required; TASK defaults to None.
- `legacy/`: previous repository files retained as historical records. Old small/medium model directories contain metadata, not runnable weights. Legacy metrics must not replace current results.
- `docs/DATASET_LICENSE` and `docs/DATASET_README.md`: existing dataset documentation retained.

No dataset image archives, model weights, virtual environments, personal review tables or local Git history are uploaded. The best Colab checkpoint remains in the existing Drive experiment; this repository does not contain it. The package is not a turnkey image-to-place service.

## Provenance and licensing

Official TEI source: [stazh/bullinger-korpus-tei](https://github.com/stazh/bullinger-korpus-tei), commit dc9fb5290c210023a6a9182e8e3102c1f6dff734. Attribution: Scheurer, Müller, Schroffenegger, Ströbel and Volk (2025), Bullinger Digital Briefkorpus, University of Zurich. Source text license: CC BY-NC-ND 3.0 CH, as stated upstream. XML is unchanged; analysis is separate. See original dataset license for manuscript data terms. Do not infer a permissive license for all repository content.
