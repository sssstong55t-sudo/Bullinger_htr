"""Build historical char3/word1 statistics from the fixed training split only."""
import argparse,collections,hashlib,json
from pathlib import Path

def build(rows,data_root):
    words={l:collections.Counter() for l in ['de','la']};grams={l:collections.Counter() for l in words};contexts={l:collections.Counter() for l in words};alphabet={l:set() for l in words};digest=hashlib.sha256()
    for r in rows:
        text=(data_root/r['text']).read_text(encoding='utf-8').strip();l=r['language']
        digest.update(json.dumps([r['image'],l,text],ensure_ascii=False).encode());words[l].update(text.split());alphabet[l].update(text+'$');s='^^'+text+'$'
        for i in range(2,len(s)):grams[l][s[i-2:i+1]]+=1;contexts[l][s[i-2:i]]+=1
    return {'version':'char3-word1-v1','training_text_sha256':digest.hexdigest(),'training_rows':len(rows),'char_weight':.03,'word_weight':.03,'smoothing':.1,'languages':{l:{'words':dict(words[l]),'grams':dict(grams[l]),'contexts':dict(contexts[l]),'alphabet':sorted(alphabet[l])} for l in words}}

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--manifest',type=Path,default=Path(__file__).resolve().parents[1]/'data/splits/n20000_natural_s42.json');p.add_argument('--data-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():p.error('Output exists')
    m=json.loads(a.manifest.read_text());e=json.loads((a.manifest.parent/m['evaluation_file']).read_text());assert hashlib.sha256(json.dumps(e,sort_keys=True).encode()).hexdigest()==m['evaluation_sha256']
    ids=lambda rows:{Path(r['image']).stem.split('_')[0] for r in rows}
    assert not ids(m['train'])&(ids(e['validation'])|ids(e['test']))
    lm=build(m['train'],a.data_root);lm['evaluation_sha256']=m['evaluation_sha256'];a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(lm,ensure_ascii=False)+'\n');print(lm['training_text_sha256'])
if __name__=='__main__':main()
