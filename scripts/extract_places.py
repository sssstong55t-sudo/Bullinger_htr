"""Extract literal place spans from HTR JSONL, without evaluating or reading reference labels."""
import argparse,hashlib,json,os,re,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def windows(text,size,overlap):
    words=list(re.finditer(r'\S+',text))
    if len(words)<=size:return [(0,len(text))]
    out=[]
    for i in range(0,len(words),size-overlap):
        out.append((0 if i==0 else words[i].start(),len(text) if i+size>=len(words) else words[i+size-1].end()))
        if i+size>=len(words):break
    return out

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,required=True,help='JSONL records with literal text and line_id')
    p.add_argument('--output',type=Path,required=True,help='New JSONL output file')
    p.add_argument('--config',type=Path,default=ROOT/'configs/place_inference.json')
    a=p.parse_args()
    if a.output.exists():p.error('Output already exists; choose a new path')
    cfg=json.loads(a.config.read_text());size,overlap=cfg['window_words'],cfg['overlap_words']
    if not 0<=overlap<size:p.error('Invalid window overlap')
    rows=[json.loads(line) for line in a.input.read_text().splitlines() if line.strip()]
    if any(not isinstance(r.get('text'),str) or not r.get('line_id') for r in rows):p.error('Each record needs text and line_id')
    os.environ['HF_HOME']=str(ROOT/'.cache/huggingface');os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
    snapshot=ROOT/'.cache/huggingface/hub'/('models--'+cfg['model_id'].replace('/','--'))/'snapshots'/cfg['revision']
    if not snapshot.is_dir():p.error('Run scripts/download_model.py first')
    sys.path.insert(0,str(ROOT/'outputs/bullinger_htr_delivery_EN/scripts'))
    from historical_aliases import HistoricalAliases
    import torch
    from gliner import GLiNER
    torch.set_num_threads(4);torch.manual_seed(42)
    model=GLiNER.from_pretrained(str(snapshot)).to('cpu').eval()
    matchers=[HistoricalAliases(json.loads((ROOT/path).read_text())) for path in cfg['lexicons']]
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('x',encoding='utf-8') as out,torch.inference_mode():
        for i,row in enumerate(rows,1):
            text=row['text'];mentions={}
            if text.strip():
                for start,end in windows(text,size,overlap):
                    chunk=text[start:end]
                    if len(list(model.data_processor.words_splitter(chunk)))>model.config.max_len:raise ValueError('Window exceeds model token limit; reduce window_words')
                    for ent in model.predict_entities(chunk,cfg['labels'],threshold=cfg['threshold']):
                        if ent['label']!='location':continue
                        x,y=ent['start']+start,ent['end']+start
                        mentions[x,y]={'start':x,'end':y,'text':text[x:y],'sources':['gliner'],'score':ent['score']}
                for matcher in matchers:
                    for alias in matcher.extract(text,row['line_id']):
                        for seg in alias['segments']:
                            x,y=seg['start'],seg['end'];mention=mentions.setdefault((x,y),{'start':x,'end':y,'text':text[x:y],'sources':[]})
                            if 'official_source_variant' not in mention['sources']:mention['sources'].append('official_source_variant')
            result={**row,'places':sorted(mentions.values(),key=lambda x:(x['start'],x['end'])),'place_config_sha256':hashlib.sha256(a.config.read_bytes()).hexdigest(),'geographic_identity_status':'Not resolved or verified'}
            out.write(json.dumps(result,ensure_ascii=False)+'\n');out.flush();print(f'{i}/{len(rows)}: {len(mentions)} place candidates',flush=True)

if __name__=='__main__':main()
