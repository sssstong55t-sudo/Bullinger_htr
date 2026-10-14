"""Choose among saved beam candidates using fixed train-text-only statistics."""
import argparse,json,math
from pathlib import Path

def lm_scores(text,language,stats):
    r=stats['languages'][language];s='^^'+text+'$';v=len(r['alphabet'])+1
    c=sum(math.log((r['grams'].get(s[i-2:i+1],0)+.1)/(r['contexts'].get(s[i-2:i],0)+.1*v)) for i in range(2,len(s)))/max(1,len(s)-2)
    tokens=text.split();n=sum(r['words'].values());vocab=len(r['words'])+1
    w=sum(math.log((r['words'].get(t,0)+.1)/(n+.1*vocab)) for t in tokens)/max(1,len(tokens))
    return c,w

def choose(candidates,language,stats):
    if language not in ['de','la'] or not candidates:raise ValueError('Known de/la language and non-empty candidates are required')
    scored=[]
    for item in candidates:
        c,w=lm_scores(item['text'],language,stats);score=float(item['model_score'])+.03*c+.03*w
        if not math.isfinite(score):raise ValueError('Non-finite candidate score')
        scored.append({**item,'char_score':c,'word_score':w,'reranked_score':score})
    return max(scored,key=lambda x:x['reranked_score']),scored

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--lm',type=Path,default=Path(__file__).resolve().parents[1]/'data/reranker_char3_word1.json');a=p.parse_args()
    if a.output.exists():p.error('Output exists')
    stats=json.loads(a.lm.read_text());a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('x') as out:
        for line in a.input.read_text().splitlines():
            if not line.strip():continue
            r=json.loads(line);winner,candidates=choose(r['candidates'],r['language'],stats)
            out.write(json.dumps({**r,'raw_text':r['text'],'text':winner['text'],'candidates':candidates,'reranking':{'char_weight':.03,'word_weight':.03,'training_text_sha256':stats['training_text_sha256']}},ensure_ascii=False)+'\n')
if __name__=='__main__':main()
