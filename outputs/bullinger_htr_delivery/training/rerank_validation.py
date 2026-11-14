"""Train-text-only character/word LM reranking of beam8 validation candidates."""
from pathlib import Path
import json,time,hashlib,zipfile,collections,math,subprocess,sys,shutil,os
ROOT=Path('/content/drive/MyDrive/BullingerAblation');WORK=Path('/content/wer-rescoring');WORK.mkdir(exist_ok=True)
BASE=ROOT/'runs_scale20k_v1/20261012T195316Z/n20000_natural_s42'
SOURCE=ROOT/'adapt20k_replication_v1/seed44_20261013T172351Z'
cfg=json.loads((BASE/'config.json').read_text());train=cfg['manifest']['train'];ev=json.loads((BASE/'evaluation.json').read_text())
assert len(train)==20000 and len(ev['validation'])==246
assert hashlib.sha256(json.dumps(ev,sort_keys=True).encode()).hexdigest()==cfg['evaluation_sha256']
letter=lambda r:Path(r['image']).stem.split('_')[0]
assert not ({letter(r) for r in train}&{letter(r) for r in ev['validation']})
# Only training TXT and validation images are restored; no test rows or test references.
needed={str(Path('dataset')/r['language']/Path(r['text']).name) for r in train}
needed.update(str(Path('dataset')/r['language']/Path(r['image']).name) for r in ev['validation'])
for name in ['dataset-baseline.zip','dataset-additional.zip']:
 with zipfile.ZipFile(ROOT/'input'/name) as z:
  for info in z.infolist():
   if info.filename in needed:
    assert (WORK/info.filename).resolve().is_relative_to(WORK.resolve())
    z.extract(info,WORK);needed.remove(info.filename)
assert not needed,'Missing data'
OUT=ROOT/'wer_rescoring_v1'/time.strftime('%Y%m%dT%H%M%SZ',time.gmtime());OUT.mkdir(parents=True)
path=lambda r,k:WORK/'dataset'/r['language']/Path(r[k]).name
words={l:collections.Counter() for l in ['de','la']};grams={l:collections.Counter() for l in ['de','la']};contexts={l:collections.Counter() for l in ['de','la']};alphabet={l:set() for l in ['de','la']};digest=hashlib.sha256()
for r in train:
 text=path(r,'text').read_text(encoding='utf-8').strip();l=r['language'];digest.update(json.dumps([r['image'],l,text],ensure_ascii=False).encode());words[l].update(text.split());alphabet[l].update(text+'$')
 s='^^'+text+'$'
 for i in range(2,len(s)):grams[l][s[i-2:i+1]]+=1;contexts[l][s[i-2:i]]+=1
# LM is fitted once, solely on the frozen 20k training references. No modern dictionaries.
def lm(text,l):
 s='^^'+text+'$';v=len(alphabet[l])+1
 c=sum(math.log((grams[l][s[i-2:i+1]]+.1)/(contexts[l][s[i-2:i]]+.1*v)) for i in range(2,len(s)))/max(1,len(s)-2)
 w=text.split();n=sum(words[l].values());vocab=len(words[l])+1
 u=sum(math.log((words[l][token]+.1)/(n+.1*vocab)) for token in w)/max(1,len(w))
 return c,u
weights=[(a,b) for a in [0.,.01,.03,.1] for b in [0.,.01,.03]]
protocol=dict(source=str(SOURCE/'de20'),seed=44,validation_lines=246,evaluation_sha256=cfg['evaluation_sha256'],training_text_sha256=digest.hexdigest(),lm='case-sensitive character trigram + word unigram, add0.1 smoothing, average log probabilities',candidate_generation='beam8, return8, original max128, no sample',weights=weights,selection='overall raw CER/WER must both not worsen; minimize max(CER/.15,WER/.37); retain baseline on ties',language_routing='manifest DE/LA; requires known language at inference',test_evaluated=False,labels_modified=False,training_started=False)
(OUT/'protocol.json').write_text(json.dumps(protocol,indent=2));print('RESCORING OUTPUT',OUT,flush=True)
subprocess.check_call([sys.executable,'-m','pip','install','-q','transformers==4.51.0','Pillow==12.2.0','numpy==2.2.6'])
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import torch
from PIL import Image
from transformers import TrOCRProcessor,VisionEncoderDecoderModel
torch.manual_seed(44);torch.use_deterministic_algorithms(True);assert torch.cuda.is_available();print('GPU',torch.cuda.get_device_name(0),flush=True)
local=WORK/'model';shutil.copytree(SOURCE/'de20',local,ignore=shutil.ignore_patterns('training_state.pt'),dirs_exist_ok=False)
processor=TrOCRProcessor.from_pretrained(local,use_fast=False);model=VisionEncoderDecoderModel.from_pretrained(local).to('cuda').eval();tok=processor.tokenizer
for c in [model.config,model.generation_config]:c.decoder_start_token_id=tok.cls_token_id;c.pad_token_id=tok.pad_token_id;c.eos_token_id=tok.sep_token_id
saved=[json.loads(s) for s in (SOURCE/'de20_validation.jsonl').read_text().splitlines()];refs={r['line_id']:r for r in saved};assert len(refs)==246
rows=[]
with torch.inference_mode():
 for i,r in enumerate(ev['validation'],1):
  line=Path(r['image']).stem;ref=refs[line];assert ref['language']==r['language']
  with Image.open(path(r,'image')) as im:x=processor(images=im.convert('RGB'),return_tensors='pt').pixel_values.to('cuda')
  generated=model.generate(x,num_beams=8,num_return_sequences=8,max_new_tokens=128,do_sample=False,use_cache=False,return_dict_in_generate=True,output_scores=True)
  texts=[s.strip() for s in processor.batch_decode(generated.sequences,skip_special_tokens=True)]
  assert texts[0]==ref['prediction'],'Top beam differs from frozen baseline; stop comparison'
  candidates=[]
  for text,score in zip(texts,generated.sequences_scores.cpu().tolist()):
   char,word=lm(text,r['language']);candidates.append(dict(text=text,model_score=score,char_score=char,word_score=word))
  rows.append(dict(**ref,candidates=candidates))
  if i%25==0:print('CANDIDATES',i,'/246',flush=True)
(OUT/'candidates.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
def distance(a,b):
 prev=list(range(len(b)+1))
 for i,x in enumerate(a,1):
  cur=[i]
  for j,y in enumerate(b,1):cur.append(min(cur[-1]+1,prev[j]+1,prev[j-1]+(x!=y)))
  prev=cur
 return prev[-1]
def metrics(rs):
 result={}
 for l in ['overall','de','la']:
  sub=[r for r in rs if l=='overall' or r['language']==l];chars=sum(len(r['reference']) for r in sub);nw=sum(len(r['reference'].split()) for r in sub);ce=sum(distance(r['reference'],r['prediction']) for r in sub);we=sum(distance(r['reference'].split(),r['prediction'].split()) for r in sub)
  result[l]=dict(n=len(sub),cer=ce/chars,wer=we/nw,character_errors=ce,reference_chars=chars,word_errors=we,reference_words=nw)
 return result
baseline=metrics(rows);assert baseline==json.loads((SOURCE/'de20/complete.json').read_text())['validation']
records=[];predictions={}
for a,b in weights:
 rs=[];changed=0
 for r in rows:
  c=max(r['candidates'],key=lambda c:c['model_score']+a*c['char_score']+b*c['word_score']);changed+=c['text']!=r['prediction'];rs.append(dict(line_id=r['line_id'],language=r['language'],reference=r['reference'],prediction=c['text']))
 score=metrics(rs);v=score['overall'];base=baseline['overall'];eligible=v['cer']<=base['cer'] and v['wer']<=base['wer'];key=f'char{a}_word{b}'
 record=dict(key=key,char_weight=a,word_weight=b,changed_lines=changed,validation=score,eligible=eligible);records.append(record);predictions[key]=rs;print('RESCORE RESULT',json.dumps(record),flush=True)
choices=[r for r in records if r['eligible']];best=min(choices,key=lambda r:max(r['validation']['overall']['cer']/.15,r['validation']['overall']['wer']/.37))
(OUT/'summary.json').write_text(json.dumps(records,indent=2));(OUT/'selection.json').write_text(json.dumps(best,indent=2));(OUT/'selected_validation.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in predictions[best['key']]))
# Oracle uses validation refs only as a diagnostic ceiling, never a deployed candidate selector.
oracle_words=sum(min(distance(r['reference'].split(),c['text'].split()) for c in r['candidates']) for r in rows);oracle=dict(word_errors=oracle_words,reference_words=2049,wer=oracle_words/2049,diagnostic_only=True)
(OUT/'oracle.json').write_text(json.dumps(oracle));print('CANDIDATE ORACLE',json.dumps(oracle),flush=True);print('RESCORING ALL COMPLETE; SELECTED',json.dumps(best),'OUTPUT',OUT,flush=True)
