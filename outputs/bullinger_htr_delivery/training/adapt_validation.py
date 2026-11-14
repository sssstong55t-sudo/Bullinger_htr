"""Three-arm 500-step validation-only adaptation; original references retained."""
from pathlib import Path
import json,sys,subprocess,time,zipfile,hashlib,random,shutil
ROOT=Path('/content/drive/MyDrive/BullingerAblation');WORK=Path('/content/adapt20k');WORK.mkdir(exist_ok=True)
subprocess.check_call([sys.executable,'-m','pip','install','-q','transformers==4.51.0','Pillow==12.2.0','numpy==2.2.6'])
BASE=ROOT/'runs_scale20k_v1/20261012T195316Z/n20000_natural_s42'
cfg=json.loads((BASE/'config.json').read_text());m=cfg['manifest'];ev=json.loads((BASE/'evaluation.json').read_text())
assert len(m['train'])==20000
assert hashlib.sha256(json.dumps(ev,sort_keys=True).encode()).hexdigest()==cfg['evaluation_sha256']
assert json.loads((BASE/'results.json').read_text())['best_epoch']==5
selections=sorted(BASE.parent.glob('beam_validation_*/selection.json'));selection=json.loads(selections[-1].read_text())
assert selection['selected_beam']==8 and selection['parent']==str(BASE)
letter=lambda r:Path(r['image']).stem.split('_')[0]
assert not ({letter(r) for r in m['train']}&{letter(r) for rows in ev.values() for r in rows})
assert not ({letter(r) for r in ev['validation']}&{letter(r) for r in ev['test']})
import argparse
p=argparse.ArgumentParser();p.add_argument('--seed',type=int,required=True);args=p.parse_args()
seed=args.seed;budget=1000
natural=list(range(20000));random.Random(seed+1).shuffle(natural);natural=natural[:budget]
groups={l:[i for i,r in enumerate(m['train']) if r['language']==l] for l in ['de','la']}
for l in groups:random.Random(seed+1).shuffle(groups[l])
de20=groups['de'][:200]+groups['la'][:800];random.Random(seed+1).shuffle(de20)
de30=groups['de'][:300]+groups['la'][:700];random.Random(seed+1).shuffle(de30)
arms={'natural':natural,'de20':de20};assert all(len(set(x))==budget for x in arms.values())
out=ROOT/'adapt20k_replication_v1'/('seed'+str(seed)+'_'+time.strftime('%Y%m%dT%H%M%SZ',time.gmtime()));out.mkdir(parents=True)
(out/'protocol.json').write_text(json.dumps(dict(parent=str(BASE),source_checkpoint=5,seed=seed,steps=500,batch_size=2,learning_rate=2e-6,decode_beams=8,max_length=128,evaluation_sha256=cfg['evaluation_sha256'],sampling='natural permutation vs exact 200DE/800LA without replacement from same 20k pool',selection='retain baseline unless candidate improves at least one metric without worsening the other; compare target-normalized worst ratio among eligible candidates',test_evaluated=False,labels_modified=False,worker_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),indent=2))
for name,indices in arms.items():(out/(name+'_indices.json')).write_text(json.dumps(indices))
print('PAIRED PROTOCOL SAVED',out,flush=True)
needed_rows=[m['train'][i] for i in set(natural+de20)]+ev['validation']
needed={str(Path('dataset')/r['language']/Path(r[k]).name) for r in needed_rows for k in ['image','text']}
for archive in ['dataset-baseline.zip','dataset-additional.zip']:
 with zipfile.ZipFile(ROOT/'input'/archive) as z:
  chosen=[i for i in z.infolist() if i.filename in needed and not (WORK/i.filename).exists()]
  assert shutil.disk_usage(WORK).free>sum(i.file_size for i in chosen)+3*1024**3
  for j,i in enumerate(chosen,1):
   assert (WORK/i.filename).resolve().is_relative_to(WORK.resolve());z.extract(i,WORK)
   if j%1000==0:print('RESTORED',archive,j,'/',len(chosen),flush=True)
assert all((WORK/n).is_file() for n in needed)
import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import numpy as np,torch
from PIL import Image
from transformers import TrOCRProcessor,VisionEncoderDecoderModel
assert torch.cuda.is_available();print('GPU',torch.cuda.get_device_name(0),flush=True)
torch.use_deterministic_algorithms(True)
parent_adapt=ROOT/'adapt20k_paired_v1/20261013T101050Z'
assert json.loads((parent_adapt/'selection.json').read_text())['selected_arm']=='de20'
source=parent_adapt/'de20'
(out/'parent.json').write_text(json.dumps(dict(weights=str(source),optimizer='fresh AdamW in each arm',seed=seed),indent=2));local_model=WORK/'initial_model'
if not local_model.exists():shutil.copytree(source,local_model,ignore=shutil.ignore_patterns('training_state.pt'),dirs_exist_ok=False)
processor=TrOCRProcessor.from_pretrained(local_model,use_fast=False);tok=processor.tokenizer
path=lambda r,k:WORK/'dataset'/r['language']/Path(r[k]).name
labels={}
for r in needed_rows:
 text=path(r,'text').read_text(encoding='utf-8').strip();ids=tok(text).input_ids
 assert text and len(ids)<=128
 labels[r['image']]=ids

def distance(a,b):
 prev=list(range(len(b)+1))
 for i,x in enumerate(a,1):
  cur=[i]
  for j,y in enumerate(b,1):cur.append(min(cur[-1]+1,prev[j]+1,prev[j-1]+(x!=y)))
  prev=cur
 return prev[-1]
def metrics(rows):
 result={}
 for group in ['overall','de','la']:
  rs=[r for r in rows if group=='overall' or r['language']==group];chars=sum(len(r['reference']) for r in rs);words=sum(len(r['reference'].split()) for r in rs)
  ce=sum(distance(r['reference'],r['prediction']) for r in rs);we=sum(distance(r['reference'].split(),r['prediction'].split()) for r in rs)
  result[group]=dict(n=len(rs),character_errors=ce,reference_chars=chars,word_errors=we,reference_words=words,cer=ce/chars if chars else None,wer=we/words if words else None)
 return result
@torch.inference_mode()
def evaluate(model,name):
 model.eval();rows=[]
 for i,r in enumerate(ev['validation'],1):
  with Image.open(path(r,'image')) as im:x=processor(images=im.convert('RGB'),return_tensors='pt').pixel_values.to('cuda')
  ids=model.generate(x,max_new_tokens=128,num_beams=8,do_sample=False,use_cache=False)
  rows.append(dict(line_id=Path(r['image']).stem,language=r['language'],reference=path(r,'text').read_text(encoding='utf-8').strip(),prediction=processor.batch_decode(ids,skip_special_tokens=True)[0].strip()))
  if i%50==0:print(name,'VALIDATION',i,'/246',flush=True)
 (out/(name+'_validation.jsonl')).write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
 return metrics(rows)
def load_model():
 random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
 model=VisionEncoderDecoderModel.from_pretrained(local_model).to('cuda')
 for c in [model.config,model.generation_config]:c.decoder_start_token_id=tok.cls_token_id;c.pad_token_id=tok.pad_token_id;c.eos_token_id=tok.sep_token_id
 return model
baseline=None;records=[]
for name,indices in arms.items():
 model=load_model()
 if baseline is None:
  baseline=evaluate(model,'baseline')
  expected=json.loads((parent_adapt/'de20/complete.json').read_text())['validation']
  assert baseline==expected,'Baseline does not reproduce frozen validation'
  print('BASELINE VERIFIED',json.dumps(baseline),flush=True)
 optimizer=torch.optim.AdamW(model.parameters(),lr=2e-6);model.train()
 random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
 total=0.;started=time.time()
 for step in range(500):
  rs=[m['train'][i] for i in indices[step*2:step*2+2]];images=[];ys=[]
  for r in rs:
   with Image.open(path(r,'image')) as im:images.append(im.convert('RGB'))
   ids=labels[r['image']];ys.append(ids+[tok.pad_token_id]*(128-len(ids)))
  x=processor(images=images,return_tensors='pt').pixel_values.to('cuda');y=torch.tensor(ys,device='cuda');y[y==tok.pad_token_id]=-100
  optimizer.zero_grad();outputs=model(pixel_values=x,labels=y)
  loss=torch.nn.functional.cross_entropy(outputs.logits.float().reshape(-1,outputs.logits.shape[-1]),y.reshape(-1),ignore_index=-100)
  assert torch.isfinite(loss);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step();total+=loss.item()
  if (step+1)%100==0:print('ADAPT',name,'STEP',step+1,'/500 LOSS',total/(step+1),flush=True)
 val=evaluate(model,name);d=out/name;d.mkdir();model.save_pretrained(d);processor.save_pretrained(d)
 torch.save(dict(step=500,optimizer=optimizer.state_dict(),torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),python_rng=random.getstate(),numpy_rng=np.random.get_state()),d/'training_state.pt')
 b=baseline['overall'];v=val['overall'];eligible=v['cer']<=b['cer'] and v['wer']<=b['wer'] and (v['cer']<b['cer'] or v['wer']<b['wer'])
 record=dict(arm=name,validation=val,baseline=baseline,eligible=eligible,meets_targets=v['cer']<=.15 and v['wer']<.37,mean_loss=total/500,seconds=time.time()-started,de_draws=sum(m['train'][i]['language']=='de' for i in indices),unique_draws=len(set(indices)),test_evaluated=False)
 (d/'complete.json').write_text(json.dumps(record,indent=2));records.append(record);(out/'summary.json').write_text(json.dumps(records,indent=2));print('ADAPT RESULT',json.dumps(record),flush=True)
 del model,optimizer,outputs,loss;torch.cuda.empty_cache()
choices=[dict(arm='baseline',validation=baseline)]+[r for r in records if r['eligible']]
best=min(choices,key=lambda r:max(r['validation']['overall']['cer']/.15,r['validation']['overall']['wer']/.37))
(out/'selection.json').write_text(json.dumps(dict(selected_arm=best['arm'],criterion='baseline or joint non-worsening candidate; target-normalized worst ratio',test_evaluated=False),indent=2))
print('PAIRED ADAPTATION COMPLETE; SELECTED',best['arm'],'OUTPUT',out,flush=True)
