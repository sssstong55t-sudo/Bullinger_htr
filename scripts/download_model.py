"""Download the pinned GLiNER snapshot once; evaluation remains offline."""
from pathlib import Path
import os
ROOT=Path(__file__).resolve().parents[1]
os.environ['HF_HOME']=str(ROOT/'.cache/huggingface')
from huggingface_hub import snapshot_download
snapshot_download(repo_id='urchade/gliner_multi-v2.1', revision='443d26d654e0324125a96bebd8e796c14ff2efe6', cache_dir=str(ROOT/'.cache/huggingface/hub'))
print('Pinned model cached. Official evaluation can now run offline.')
