# Preserved Colab Training Workflow

These workers preserve previously executed configurations. They are not automatically run by the official NER command. Some training workers resume completed epochs from existing Drive checkpoints; do not describe every invocation as training from scratch.

| Worker | Purpose | Resources |
|---|---|---|
| train_10k_sampling.py | Natural versus balanced 10k training, including historical recovery logic | GPU; includes evaluation after training |
| train_20k.py | 20k natural-proportion training/recovery | GPU; includes evaluation after training |
| adapt_validation.py | Matched natural/DE20 500-step adaptation for seeds 43, 44, 45 | GPU; validation only |
| rerank_validation.py | Train-text-only LM and saved beam candidates | GPU; validation only |
| evaluate_frozen_test.py | Frozen 0.03/0.03 test evaluation | GPU; do not tune parameters on this test |
| context_validation.py | Saved-candidate contextual LM ablation | CPU; validation only |

## Drive dependencies

Mount the existing Drive under `/content/drive`. The workers use `/content/drive/MyDrive/BullingerAblation` and its existing experiment directories:

- `input/dataset-baseline.zip`, `dataset-additional.zip`, `experiment-code.zip`.
- `runs_scale_weights_restart_v1/n10000_natural_s42_20261011T212741Z` and saved 10k recovery checkpoints.
- `runs_scale20k_v1/20261012T195316Z/n20000_natural_s42`: configuration, fixed evaluation manifest, epoch5 checkpoint and beam-validation selection.
- `adapt20k_replication_v1/seed44_20261013T172351Z/de20`: reranking source weights.
- `wer_rescoring_v1/20261013T185538Z`: candidates, protocol and selection.

Keep fixed validation/test manifests and letter-disjointness audits. The workers reference actual existing run dates; moving these data requires updating their paths consistently. The 246-line validation and 500-line test are fixed; no annotation labels are changed by packaging. Raw CER/WER retain case, punctuation and historical spelling and use corpus-level error sums. Known language metadata is used by the LM reranker.

For a copied worker in Colab:

```python
from google.colab import drive
import subprocess, sys

drive.mount('/content/drive')
subprocess.run([sys.executable, '-u', '/content/adapt_validation.py', '--seed', '44'], check=True)
```

Select the appropriate CPU/GPU before running a task. To release that task's runtime after success or failure, wrap execution in `try/finally` and use `google.colab.runtime.unassign()`. Do not release another active experiment. The new clean notebook defaults to no selected task, preventing accidental paid GPU work.

Training dependencies are installed by the original workers (Transformers 4.51.0, Pillow 12.2.0, NumPy 2.2.6). Historical Colab runs recorded Python 3.13/PyTorch 2.11; cross-environment bitwise equality is not guaranteed. Image archives and optimizer state are not included. Complete inference weights are distributed through the GitHub Release. Fixed portable manifests are in data/splits/; see docs/REPRODUCTION.md.
