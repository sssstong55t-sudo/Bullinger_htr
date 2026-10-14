# Bullinger HTR and Official Place Recognition — English Delivery

Updated: 14 October 2026. This directory is the current submission package. It contains executable code, fixed official reference data, actual results, and provenance. Original historical German/Latin text is preserved; translating it would invalidate recognition evaluation.

## Current evaluation policy

Place evaluation uses **official TEI body annotations only**. Personal review tables and AI preannotations do not supply the current reference labels. Historical manual-evaluation reports remain elsewhere in the project as archival records. The official-only runner does not read those tables, the old priority profile, the manual-derived boundary rule, or the Bullinger person-name guard.

The official source is [stazh/bullinger-korpus-tei](https://github.com/stazh/bullinger-korpus-tei), commit `dc9fb5290c210023a6a9182e8e3102c1f6dff734`. Forty candidate letters were fixed before inference; 39 were eligible, providing 357 body paragraphs and 327 annotated geographic mentions. Source-learning letters are excluded. The letters have now been inspected and evaluated, so subsequent improvements using their errors are development work, not unseen confirmation.

## Main official annotation-agreement result

| Measure | Result |
|---|---:|
| Matched official spans | 200 |
| Predictions without an exact official match | 41 |
| Official spans not retrieved | 127 |
| Official positive coverage / agreement recall | 61.16% |
| Official annotation-agreement precision | 82.99% |
| Official annotation-agreement F1 | 70.42% |

The main branch was predefined: competing labels `location/person/organization`, threshold 0.8, plus the existing official-source variant layers. The runner also reports all fixed 0.4/0.8 and location-only controls. It does not select a best threshold after seeing these results.

**Interpretation:** annotation-agreement metrics compare predictions with the published span set. Annotation completeness is not established, so an unmatched prediction is not automatically a genuine false positive. Do not present these values as absolute true precision, full recall, geographic-coordinate accuracy, or HTR end-to-end accuracy. The strict P>=85% goal is not met by this official-only result.

## Run official NER

From the repository root, using the existing offline model cache:

```sh
.venv-nlp/bin/python outputs/bullinger_htr_delivery_EN/scripts/run_official_ner.py --output-dir outputs/official_ner_new_run
```

Use a fresh output directory. The runner preserves the supplied results and refuses to overwrite an existing run. It uses the project's `.cache/huggingface` and pins GLiNER to revision `443d26d654e0324125a96bebd8e796c14ff2efe6`. It requires the existing project Python environment; model weights are not duplicated into this package. Copying the package alone to another computer requires installing dependencies and acquiring that pinned model.

## Package layout

- `scripts/`: official-only inference runner and its two local helper modules.
- `data/`: fixed reference paragraphs, untouched official letter XML, source selection/audit, and source-backed variant lexicons.
- `results/`: actual predictions, metrics, parameters, and 127 documented misses.
- `training/`: preserved Colab training/adaptation/reranking workers, separate from current official NER evaluation.
- `TRAINING.md`: Colab execution and checkpoint dependencies.
- `RESULTS.md`: result definitions, controls, and limitations.
- `manifest.json`: file hashes.
- `CLEANUP.md`: recorded removal of redundant generated artifacts.

## HTR results remain separate

The previous frozen 500-line HTR evaluation is CER **13.3917%**, WER **38.6088%** with seed44 DE20, beam8, and fixed character/word reranking weights 0.03/0.03. No HTR training or new HTR test was performed for this packaging task. The official paragraph NER evaluation does not replace that image-to-text evaluation.

## Official attribution

Scheurer, Müller, Schroffenegger, Ströbel and Volk (2025), *Bullinger Digital Briefkorpus*, University of Zurich. Official text is published under CC BY-NC-ND 3.0 CH as stated by the source repository. Original XML is preserved unchanged; extracted evaluation records and analysis are separate. Published entity identifiers and coordinates are source evidence, not independent historical-resolution certification.

## Fresh GitHub checkout

Follow the repository root README to install dependencies and run scripts/download_model.py. The GitHub runner reads the pinned snapshot directory directly rather than relying on a mutable cache main ref. Recorded metrics are unchanged.

## Manuscript image source

The original Bullinger HTR image dataset is distributed by [pstroe/bullinger-htr](https://github.com/pstroe/bullinger-htr). This is distinct from the official TEI text/entity corpus. See the repository root README for the new-image inference commands and JSON configuration map.
