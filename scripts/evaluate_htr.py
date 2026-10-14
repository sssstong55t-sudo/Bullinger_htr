"""Compute corpus-level CER/WER against an explicitly supplied reference JSONL."""
import argparse,json
from pathlib import Path

def distance(a,b):
    prev=list(range(len(b)+1))
    for i,x in enumerate(a,1):
        cur=[i]
        for j,y in enumerate(b,1):cur.append(min(cur[-1]+1,prev[j]+1,prev[j-1]+(x!=y)))
        prev=cur
    return prev[-1]

def metrics(rows):
    out={}
    for l in ['overall','de','la']:
        rs=[r for r in rows if l=='overall' or r['language']==l];ch=sum(len(r['reference']) for r in rs);nw=sum(len(r['reference'].split()) for r in rs);ce=sum(distance(r['reference'],r['prediction']) for r in rs);we=sum(distance(r['reference'].split(),r['prediction'].split()) for r in rs)
        out[l]={'n':len(rs),'cer':ce/ch if ch else None,'wer':we/nw if nw else None,'character_errors':ce,'reference_chars':ch,'word_errors':we,'reference_words':nw}
    return out

def indexed(path):
    rows=[json.loads(x) for x in path.read_text().splitlines() if x.strip()];d={r['line_id']:r for r in rows}
    if len(rows)!=len(d):raise ValueError('Duplicate line_id')
    return d

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--predictions',type=Path,required=True);p.add_argument('--references',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():p.error('Output exists')
    pred=indexed(a.predictions);ref=indexed(a.references)
    if set(pred)!=set(ref):raise ValueError('Prediction/reference IDs differ; missing lines cannot be silently ignored')
    rows=[]
    for key,r in ref.items():
        if r['language'] not in ['de','la']:raise ValueError('Reference language must be de/la')
        rows.append({'reference':r['reference'],'prediction':pred[key]['text'],'language':r['language']})
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps({'normalization':'None; case, accents and punctuation retained. WER uses whitespace split. Ratios, not percentages.','metrics':metrics(rows)},indent=2)+'\n')
if __name__=='__main__':main()
