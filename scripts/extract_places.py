"""Extract literal place spans from HTR JSONL, without evaluating or reading reference labels."""
import argparse,hashlib,json,os,re,sys
import unicodedata
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def whole_word(text, start, end):
    neighbors = text[start - 1:start] if start else ''
    neighbors += text[end:end + 1]
    return all(unicodedata.category(char)[0] not in 'LM' for char in neighbors)


def mbert_mentions(checkpoint, rows):
    from transformers import AutoConfig, AutoModelForTokenClassification, AutoTokenizer
    from mbert_inference import LABELS, predict

    config = AutoConfig.from_pretrained(checkpoint, local_files_only=True)
    if config.model_type != 'bert' or config.id2label != dict(enumerate(LABELS)):
        raise ValueError('Expected a BERT token classifier with O, B-LOC, and I-LOC labels')
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True, use_fast=True)
    if not tokenizer.is_fast:
        raise ValueError('mBERT requires a fast tokenizer for character offsets')
    model = AutoModelForTokenClassification.from_pretrained(checkpoint, local_files_only=True)
    mentions = []
    for start in range(0, len(rows), 32):
        mentions.extend(row['mentions'] for row in predict(model, tokenizer, rows[start:start + 32]))
    return mentions


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
    default_checkpoint = ROOT/'models/bullinger_mbert_20261022'
    backend = p.add_mutually_exclusive_group()
    backend.add_argument('--mbert', type=Path, nargs='?', const=default_checkpoint,
                         help='Use mBERT with historical aliases (default); optionally give a checkpoint path')
    backend.add_argument('--gliner', action='store_true', help='Use GLiNER with historical aliases')
    a=p.parse_args()
    checkpoint = a.mbert or default_checkpoint
    if not a.gliner and not (checkpoint/'model.safetensors').is_file():
        p.error(f'mBERT weights missing in {checkpoint}. See the README checkpoint download instructions.')
    if a.output.exists():p.error('Output already exists; choose a new path')
    cfg=json.loads(a.config.read_text());size,overlap=cfg['window_words'],cfg['overlap_words']
    if not 0<=overlap<size:p.error('Invalid window overlap')
    rows=[json.loads(line) for line in a.input.read_text().splitlines() if line.strip()]
    if any(not isinstance(r.get('text'),str) or not r.get('line_id') for r in rows):p.error('Each record needs text and line_id')
    os.environ['HF_HOME']=str(ROOT/'.cache/huggingface');os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
    snapshot=ROOT/'.cache/huggingface/hub'/('models--'+cfg['model_id'].replace('/','--'))/'snapshots'/cfg['revision']
    if a.gliner and not snapshot.is_dir():p.error('Run scripts/download_model.py first')
    sys.path.insert(0,str(ROOT/'outputs/bullinger_htr_delivery_EN/scripts'))
    from historical_aliases import HistoricalAliases
    import torch
    torch.set_num_threads(4);torch.manual_seed(42)
    model=None
    neural_mentions=None
    if not a.gliner:
        neural_mentions=mbert_mentions(checkpoint, rows)
    else:
        from gliner import GLiNER
        model=GLiNER.from_pretrained(str(snapshot)).to('cpu').eval()
    matchers=[HistoricalAliases(json.loads((ROOT/path).read_text())) for path in cfg['lexicons']]
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('x',encoding='utf-8') as out,torch.inference_mode():
        for i,row in enumerate(rows,1):
            text=row['text'];mentions={}
            if neural_mentions is not None:
                for span in neural_mentions[i - 1]:
                    x, y = span['start'], span['end']
                    mentions[x,y] = dict(start=x, end=y, text=span['text'], sources=['mbert'], score=span['score'])
            if text.strip():
                for start,end in windows(text,size,overlap) if model is not None else []:
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
            if not a.gliner:
                mentions = {(x,y): mention for (x,y), mention in mentions.items() if whole_word(text,x,y)}
            result={**row,'places':sorted(mentions.values(),key=lambda x:(x['start'],x['end'])),'place_config_sha256':hashlib.sha256(a.config.read_bytes()).hexdigest(),'geographic_identity_status':'Not resolved or verified'}
            if not a.gliner:
                result['place_extraction_mode']='mbert_with_aliases'
            out.write(json.dumps(result,ensure_ascii=False)+'\n');out.flush();print(f'{i}/{len(rows)}: {len(mentions)} place candidates',flush=True)

if __name__=='__main__':main()
