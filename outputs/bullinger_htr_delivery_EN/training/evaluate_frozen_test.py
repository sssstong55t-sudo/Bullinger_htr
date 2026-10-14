"""Train-text-only character/word LM reranking of beam8 validation candidates."""
from pathlib import Path
import json,time,hashlib,zipfile,collections,math,subprocess,sys,shutil,os
ROOT=Path('/content/drive/MyDrive/BullingerAblation');WORK=Path('/content/wer-rescoring-test');WORK.mkdir(exist_ok=True)
BASE=ROOT/'runs_scale20k_v1/20261012T195316Z/n20000_natural_s42'
SOURCE=ROOT/'adapt20k_replication_v1/seed44_20261013T172351Z'
cfg=json.loads((BASE/'config.json').read_text());train=cfg['manifest']['train'];ev=json.loads((BASE/'evaluation.json').read_text())
assert len(train)==20000 and len(ev['test'])==500
assert hashlib.sha256(json.dumps(ev,sort_keys=True).encode()).hexdigest()==cfg['evaluation_sha256']
letter=lambda r:Path(r['image']).stem.split('_')[0]
assert not ({letter(r) for r in train}&{letter(r) for r in ev['test']})
# Restore only training TXT and frozen test images/TXT for this authorized final test.
needed={str(Path('dataset')/r['language']/Path(r['text']).name) for r in train}
needed.update(str(Path('dataset')/r['language']/Path(r[k]).name) for r in ev['test'] for k in ['image','text'])
for name in ['dataset-baseline.zip','dataset-additional.zip']:
 with zipfile.ZipFile(ROOT/'input'/name) as z:
  for info in z.infolist():
   if info.filename in needed:
    assert (WORK/info.filename).resolve().is_relative_to(WORK.resolve())
    z.extract(info,WORK);needed.remove(info.filename)
assert not needed,'Missing data'
OUT=ROOT/'wer_rescoring_test_v1'/time.strftime('%Y%m%dT%H%M%SZ',time.gmtime());OUT.mkdir(parents=True)
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
selected=json.loads((ROOT/'wer_rescoring_v1/20261013T185538Z/selection.json').read_text())
assert selected['char_weight']==selected['word_weight']==.03
parent_protocol=json.loads((ROOT/'wer_rescoring_v1/20261013T185538Z/protocol.json').read_text())
assert digest.hexdigest()==parent_protocol['training_text_sha256']
protocol=dict(parent_protocol,weights=[[.03,.03]],selection='frozen validation choice; no test-based tuning',test_evaluated=True,test_lines=500)
(OUT/'protocol.json').write_text(json.dumps(protocol,indent=2));print('FROZEN TEST OUTPUT',OUT,flush=True)
subprocess.check_call([sys.executable,'-m','pip','install','-q','transformers==4.51.0','Pillow==12.2.0','numpy==2.2.6'])
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import torch
from PIL import Image
from transformers import TrOCRProcessor,VisionEncoderDecoderModel
torch.manual_seed(44);torch.use_deterministic_algorithms(True);assert torch.cuda.is_available();print('GPU',torch.cuda.get_device_name(0),flush=True)
local=WORK/'model';shutil.copytree(SOURCE/'de20',local,ignore=shutil.ignore_patterns('training_state.pt'),dirs_exist_ok=False)
processor=TrOCRProcessor.from_pretrained(local,use_fast=False);model=VisionEncoderDecoderModel.from_pretrained(local).to('cuda').eval();tok=processor.tokenizer
for c in [model.config,model.generation_config]:c.decoder_start_token_id=tok.cls_token_id;c.pad_token_id=tok.pad_token_id;c.eos_token_id=tok.sep_token_id
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

control=[];rescored=[]
with (OUT/'predictions.jsonl').open('w',encoding='utf-8') as f,torch.inference_mode():
 for i,r in enumerate(ev['test'],1):
  with Image.open(path(r,'image')) as im:x=processor(images=im.convert('RGB'),return_tensors='pt').pixel_values.to('cuda')
  g=model.generate(x,num_beams=8,num_return_sequences=8,max_new_tokens=128,do_sample=False,use_cache=False,return_dict_in_generate=True,output_scores=True)
  texts=[t.strip() for t in processor.batch_decode(g.sequences,skip_special_tokens=True)]
  candidates=[]
  for t,score in zip(texts,g.sequences_scores.cpu().tolist()):
   c,w=lm(t,r['language']);candidates.append(dict(text=t,model_score=score,char_score=c,word_score=w))
  winner=max(candidates,key=lambda c:c['model_score']+.03*c['char_score']+.03*c['word_score'])
  # Candidate choice is complete before reading this test reference.
  reference=path(r,'text').read_text(encoding='utf-8').strip()
  row=dict(line_id=Path(r['image']).stem,language=r['language'],reference=reference,prediction=texts[0])
  control.append(row);rescored.append(dict(row,prediction=winner['text']))
  f.write(json.dumps(dict(row,rescored_prediction=winner['text'],candidates=candidates),ensure_ascii=False)+'\n');f.flush()
  if i%25==0:print('FROZEN TEST',i,'/500',flush=True)
result=dict(control=metrics(control),rescored=metrics(rescored),changed_lines=sum(a['prediction']!=b['prediction'] for a,b in zip(control,rescored)),char_weight=.03,word_weight=.03,test_based_tuning=False)
(OUT/'results.json').write_text(json.dumps(result,indent=2));print('FROZEN TEST ALL COMPLETE',json.dumps(result),'OUTPUT',OUT,flush=True)
