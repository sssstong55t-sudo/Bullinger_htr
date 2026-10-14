"""Extract only fixed-manifest members from the existing two Colab data archives."""
import argparse,hashlib,json,zipfile
from pathlib import Path

def main():
    root=Path(__file__).resolve().parents[1];p=argparse.ArgumentParser(description=__doc__);p.add_argument('--archives',type=Path,nargs='+',required=True);p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--manifest',type=Path,default=root/'data/splits/n20000_natural_s42.json');p.add_argument('--split',choices=['train','validation','test','all'],default='train');a=p.parse_args()
    m=json.loads(a.manifest.read_text());ev=json.loads((a.manifest.parent/m['evaluation_file']).read_text());assert hashlib.sha256(json.dumps(ev,sort_keys=True).encode()).hexdigest()==m['evaluation_sha256'];groups={'train':m['train'],**ev}
    docsets={k:{Path(r['image']).stem.split('_')[0] for r in v} for k,v in groups.items()}
    for k in docsets:
        for other in docsets:
            if k!=other and docsets[k]&docsets[other]:raise ValueError('Letter leakage')
    needed={r[f] for k,rs in groups.items() if a.split=='all' or k==a.split for r in rs for f in ['image','text']};a.output_dir.mkdir(parents=True,exist_ok=True);missing={n for n in needed if not (a.output_dir/n).is_file()}
    for archive in a.archives:
        with zipfile.ZipFile(archive) as z:
            for info in z.infolist():
                if info.filename not in missing:continue
                target=(a.output_dir/info.filename).resolve()
                if not target.is_relative_to(a.output_dir.resolve()):raise ValueError('Unsafe archive member')
                target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(z.read(info));missing.remove(info.filename)
    if missing:raise ValueError(f'Missing {len(missing)} fixed-manifest files')
    print(f'Ready: {len(needed)} files for {a.split}. No training or inference performed.')
if __name__=='__main__':main()
