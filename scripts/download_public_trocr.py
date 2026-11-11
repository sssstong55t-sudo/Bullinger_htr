"""Download the pinned public Bullinger TrOCR and verify the model weights."""
import hashlib
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / 'models/bullinger_public_trocr'
TOKENIZER_DIR = MODEL_DIR / 'tokenizer'
MODEL_ID = 'pstroe/bullinger-general-model'
MODEL_REVISION = 'be32c883a4deb812ab6f461ae79f583c3f9edc53'
WEIGHT_SHA256 = 'dbab3d95b50acf4d2596d70c7691f0d53494b2671f9145f957b3af02315bf475'
TOKENIZER_REVISION = 'eaacaf452b06415df8f10bb6fad3a4c11e609406'


def main():
    os.environ.setdefault('HF_HUB_DISABLE_XET', '1')
    from huggingface_hub import snapshot_download

    snapshot_download(
        MODEL_ID, revision=MODEL_REVISION, local_dir=MODEL_DIR,
        allow_patterns=['config.json', 'preprocessor_config.json', 'pytorch_model.bin'],
    )
    digest = hashlib.sha256()
    with (MODEL_DIR / 'pytorch_model.bin').open('rb') as weights:
        for chunk in iter(lambda: weights.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    if digest.hexdigest() != WEIGHT_SHA256:
        raise ValueError('Public TrOCR weight SHA256 mismatch')
    snapshot_download(
        'microsoft/trocr-base-handwritten', revision=TOKENIZER_REVISION,
        local_dir=TOKENIZER_DIR,
        allow_patterns=['vocab.json', 'merges.txt', 'tokenizer_config.json', 'special_tokens_map.json'],
    )
    print(f'Public Bullinger TrOCR ready at {MODEL_DIR}')


if __name__ == '__main__':
    main()
