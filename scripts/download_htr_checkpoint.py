"""Download the released checkpoint parts, verify, and extract the complete model."""
import argparse,hashlib,json,shutil,urllib.request,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
BASE='https://github.com/sssstong55t-sudo/Bullinger_htr/releases/download/v1.0.0-seed44-de20/'
NAME='bullinger-trocr-seed44-de20.zip'
EXPECTED='e701d1ac50d8dd3a40fc101b1ce6f7694d15b709653dbe1c86317105e4a96823'

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir',type=Path,default=ROOT/'models')
    a=p.parse_args();a.output_dir.mkdir(exist_ok=True,parents=True)
    dest=a.output_dir/'bullinger_checkpoint'
    if dest.exists():p.error('Checkpoint directory exists; choose another output-dir')
    cache=a.output_dir/'checkpoint_download';cache.mkdir(exist_ok=True)
    parts=[]
    for i in range(1,21):
        name=NAME+f'.part{i:02d}';local=cache/name
        if not local.exists():
            temp=cache/(name+'.download')
            print(f'Downloading part {i}/20',flush=True)
            with urllib.request.urlopen(BASE+name,timeout=120) as response,temp.open('wb') as out:shutil.copyfileobj(response,out)
            temp.rename(local)
        parts.append(local)
    archive=cache/NAME
    with archive.open('wb') as out:
        for part in parts:
            with part.open('rb') as f:shutil.copyfileobj(f,out)
    if sha(archive)!=EXPECTED:raise ValueError('Archive SHA256 mismatch. Remove checkpoint_download and retry; no model was extracted.')
    with zipfile.ZipFile(archive) as z:
        for m in z.infolist():
            target=(a.output_dir/m.filename).resolve()
            if not target.is_relative_to(a.output_dir.resolve()):raise ValueError('Unsafe archive path')
        if z.testzip() is not None:raise ValueError('ZIP integrity check failed')
        z.extractall(a.output_dir)
    manifest=json.loads((dest/'checkpoint_manifest.json').read_text())
    for name,record in manifest['files'].items():
        if sha(dest/name)!=record['sha256']:raise ValueError('File hash mismatch: '+name)
    print('Complete inference model ready:',dest,flush=True)
    print('Verified download parts remain in checkpoint_download; they may be removed after use.')

if __name__=='__main__':main()
