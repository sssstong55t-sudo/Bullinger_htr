"""CPU validation-only context LM ablation on frozen saved candidates."""
from pathlib import Path
import json,zipfile,collections,math,time,hashlib
ROOT=Path('/content/drive/MyDrive/BullingerAblation');BASE=ROOT/'runs_scale20k_v1/20261012T195316Z/n20000_natural_s42';PARENT=ROOT/'wer_rescoring_v1/20261013T185538Z'
train=json.loads((BASE/'config.json').read_text())['manifest']['train'];rows=[json.loads(s) for s in (PARENT/'candidates.jsonl').read_text().splitlines()]
assert len(train)==20000 and len(rows)==246
assert not ({Path(r['image']).stem.split('_')[0] for r in train}&{r['line_id'].split('_')[0] for r in rows})
needed={str(Path('dataset')/r['language']/Path(r['text']).name) for r in train};texts={}
print('READING TRAINING TEXT ONLY',len(needed),flush=True)
for archive in ['dataset-baseline.zip','dataset-additional.zip']:
 with zipfile.ZipFile(ROOT/'input'/archive) as z:
  for name in z.namelist():
   if name in needed:texts[name]=z.read(name).decode('utf-8').strip()
assert len(texts)==len(needed)
g={l:collections.Counter() for l in ['de','la']};ctx={l:collections.Counter() for l in g};alphabet={l:set() for l in g};big={l:collections.Counter() for l in g};wc={l:collections.Counter() for l in g};digest=hashlib.sha256()
for r in train:
 t=texts[str(Path('dataset')/r['language']/Path(r['text']).name)];l=r['language'];digest.update(json.dumps([r['image'],l,t],ensure_ascii=False).encode());alphabet[l].update(t+'$');s='^^^^'+t+'$'
 for i in range(4,len(s)):g[l][s[i-4:i+1]]+=1;ctx[l][s[i-4:i]]+=1
 tokens=['<BOS>']+t.split()+['<EOS>']
 for a,b in zip(tokens,tokens[1:]):big[l][(a,b)]+=1;wc[l][a]+=1
expected=json.loads((PARENT/'protocol.json').read_text())['training_text_sha256'];assert digest.hexdigest()==expected
vocab={l:len({b for a,b in big[l]})+1 for l in g}
def score(t,l):
 s='^^^^'+t+'$';c=sum(math.log((g[l][s[i-4:i+1]]+.1)/(ctx[l][s[i-4:i]]+.1*(len(alphabet[l])+1))) for i in range(4,len(s)))/max(1,len(s)-4)
 ts=['<BOS>']+t.split()+['<EOS>'];w=sum(math.log((big[l][(a,b)]+.1)/(wc[l][a]+.1*vocab[l])) for a,b in zip(ts,ts[1:]))/max(1,len(ts)-1)
 return c,w
for r in rows:
 for candidate in r['candidates']:candidate['char5'],candidate['word2']=score(candidate['text'],r['language'])
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
  sub=[r for r in rs if l=='overall' or r['language']==l];ch=sum(len(r['reference']) for r in sub);nw=sum(len(r['reference'].split()) for r in sub);ce=sum(distance(r['reference'],r['prediction']) for r in sub);we=sum(distance(r['reference'].split(),r['prediction'].split()) for r in sub)
  result[l]=dict(n=len(sub),cer=ce/ch,wer=we/nw,character_errors=ce,reference_chars=ch,word_errors=we,reference_words=nw)
 return result
out=ROOT/'context_rescoring_val_v1'/time.strftime('%Y%m%dT%H%M%SZ',time.gmtime());out.mkdir(parents=True)
settings=[('current',.03,.03),('beam8',0.,0.)]+[(family,w,0.) for family in ['char5','word2'] for w in [.01,.03,.1]]+[('context_both',.03,.03)]
(out/'protocol.json').write_text(json.dumps(dict(parent=str(PARENT),training_sha256=digest.hexdigest(),settings=settings,selection='both overall CER/WER no worse than current; minimum target worst ratio, retain current on ties',scope='246 validation only; original references; no test read',known_language_required=True),indent=2))
records=[];predictions={}
for family,a,b in settings:
 rs=[]
 for r in rows:
  def total(c):
   if family=='current':return c['model_score']+a*c['char_score']+b*c['word_score']
   if family=='char5':return c['model_score']+a*c['char5']
   if family=='word2':return c['model_score']+a*c['word2']
   if family=='context_both':return c['model_score']+a*c['char5']+b*c['word2']
   return c['model_score']
  winner=max(r['candidates'],key=total);rs.append(dict(line_id=r['line_id'],language=r['language'],reference=r['reference'],prediction=winner['text']))
 key=f'{family}_{a}_{b}';record=dict(key=key,validation=metrics(rs),changed_lines=sum(r['prediction']!=n['prediction'] for r,n in zip(rows,rs)));records.append(record);predictions[key]=rs
 if family=='current':assert record['validation']==json.loads((PARENT/'selection.json').read_text())['validation']
 print('CONTEXT RESULT',json.dumps(record),flush=True)
base=records[0]['validation']['overall'];eligible=[r for r in records if r['validation']['overall']['cer']<=base['cer'] and r['validation']['overall']['wer']<=base['wer']];best=min(eligible,key=lambda r:max(r['validation']['overall']['cer']/.15,r['validation']['overall']['wer']/.37))
(out/'summary.json').write_text(json.dumps(records,indent=2));(out/'selection.json').write_text(json.dumps(best,indent=2));(out/'selected_validation.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in predictions[best['key']]))
print('CONTEXT ALL COMPLETE',json.dumps(best),'OUTPUT',out,flush=True)
