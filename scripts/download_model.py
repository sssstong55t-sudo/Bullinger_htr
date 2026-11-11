"""Cache GLiNER and its multilingual DeBERTa configuration and tokenizer."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / '.cache/huggingface/hub'
GLINER_REVISION = '443d26d654e0324125a96bebd8e796c14ff2efe6'
ENCODER_REVISION = 'a0484667b22365f84929a935b5e50a51f71f159d'


def main():
    os.environ.setdefault('HF_HUB_DISABLE_XET', '1')
    from huggingface_hub import snapshot_download

    encoder = Path(snapshot_download(
        'microsoft/mdeberta-v3-base', revision=ENCODER_REVISION, cache_dir=CACHE,
        allow_patterns=['config.json', 'spm.model', 'tokenizer_config.json'],
    ))
    # GLiNER resolves its encoder by repository name without a revision.
    refs = encoder.parent.parent / 'refs'
    refs.mkdir(parents=True, exist_ok=True)
    (refs / 'main').write_text(ENCODER_REVISION)
    snapshot_download(
        'urchade/gliner_multi-v2.1', revision=GLINER_REVISION, cache_dir=CACHE,
        allow_patterns=['gliner_config.json', 'model.safetensors'],
    )
    print('GLiNER and its encoder tokenizer are ready for offline inference.')


if __name__ == '__main__':
    main()
