"""Official-only reference evaluation; no local human gold/profile/guard reads."""
from pathlib import Path
import json,os,sys,re,hashlib,collections,argparse
PACKAGE=Path(__file__).resolve().parents[1]
ROOT=PACKAGE.parents[1]
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output-dir',type=Path,required=True,help='Fresh output directory; existing results are preserved')
args=parser.parse_args()
P=args.output_dir.resolve()
if P.exists():parser.error('Output directory already exists; choose a fresh path')
P.mkdir(parents=True)
os.environ['HF_HOME']=str(ROOT/'.cache/huggingface');os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
sys.path.insert(0,str(PACKAGE/'scripts'));from historical_aliases import HistoricalAliases
# Self-contained windows avoid importing any human-review evaluators.
def windows(text,size=160,overlap=40):
 words=list(re.finditer(r'\S+',text))
 if len(words)<=size:return [(0,len(text))]
 out=[]
 for i in range(0,len(words),size-overlap):
  out.append((0 if i==0 else words[i].start(),len(text) if i+size>=len(words) else words[i+size-1].end()))
  if i+size>=len(words):break
 return out
import torch
from gliner import GLiNER
torch.set_num_threads(4);torch.manual_seed(42)
revision='443d26d654e0324125a96bebd8e796c14ff2efe6';cache=ROOT/'.cache/huggingface/hub/models--urchade--gliner_multi-v2.1';assert (cache/'snapshots'/revision).is_dir(), 'Run scripts/download_model.py first'
model=GLiNER.from_pretrained(str(cache/'snapshots'/revision)).to('cpu').eval()
reference=PACKAGE/'data/reference.jsonl';rows=[json.loads(s) for s in reference.read_text().splitlines()]
lexpaths=[PACKAGE/'data/historical_aliases.json',PACKAGE/'data/source_aliases.json'];lexicons=[json.loads(p.read_text()) for p in lexpaths];matchers=[HistoricalAliases(l) for l in lexicons]
docs={r['document_id'] for r in rows};assert not any(docs&set(l['learning_documents']) for l in lexicons)
protocol=dict(reference_source='official TEI body placeName only',source_commit='dc9fb5290c210023a6a9182e8e3102c1f6dff734',reference_sha256=hashlib.sha256(reference.read_bytes()).hexdigest(),model_revision=revision,labels_single=['location'],labels_competing=['location','person','organization'],thresholds=[.4,.8],selection='Report all predefined comparisons; do not tune on official sample',primary_report_branch='competition_08_official_aliases',lexicons_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in lexpaths},local_human_annotations_read=False,legacy_priority_profile_read=False,manual_audit_derived_boundary_or_person_rules=False,annotation_completeness='not established; report positive coverage and annotation-agreement metrics separately')
(P/'protocol.json').write_text(json.dumps(protocol,indent=2));names=['single_04','competition_04','competition_08','competition_04_official_aliases','competition_08_official_aliases'];stats={k:dict(matched=0,unmatched_predictions=0,unretrieved_annotations=0,official_annotations=0,predictions=0) for k in names}
with (P/'predictions.jsonl').open('w') as f,torch.inference_mode():
 for i,row in enumerate(rows,1):
  text=row['text'];single={};comp={};high={}
  for s,e in windows(text):
   chunk=text[s:e];assert len(list(model.data_processor.words_splitter(chunk)))<=model.config.max_len
   for labels,out in [(['location'],single),(['location','person','organization'],comp)]:
    for ent in model.predict_entities(chunk,labels,threshold=.4):
     if ent['label']!='location':continue
     a,b=ent['start']+s,ent['end']+s;assert 0<=a<b<=len(text);out[a,b]=dict(start=a,end=b,text=text[a:b],score=ent['score'])
     if out is comp and ent['score']>=.8:high[a,b]=out[a,b]
  aliases={}
  lid=f"official:{row['document_id']}:p{row['paragraph_number']}"
  for matcher in matchers:
   for m in matcher.extract(text,lid):
    for seg in m['segments']:a,b=seg['start'],seg['end'];aliases[a,b]=dict(start=a,end=b,text=text[a:b],source='official_source_variant')
  branches=dict(single_04=single,competition_04=comp,competition_08=high,competition_04_official_aliases={**comp,**aliases},competition_08_official_aliases={**high,**aliases});gold={(m['start'],m['end']) for m in row['mentions']}
  for name,preds in branches.items():
   actual=set(preds);v=stats[name];v['matched']+=len(actual&gold);v['unmatched_predictions']+=len(actual-gold);v['unretrieved_annotations']+=len(gold-actual);v['official_annotations']+=len(gold);v['predictions']+=len(actual)
  f.write(json.dumps(dict(document_id=row['document_id'],paragraph_number=row['paragraph_number'],text=text,predictions={n:list(ms.values()) for n,ms in branches.items()},source_url=row['source_url']),ensure_ascii=False)+'\n');f.flush()
  if i%30==0:print('OFFICIAL-ONLY',i,'/',len(rows),flush=True)
for v in stats.values():
 tp,up,ua=v['matched'],v['unmatched_predictions'],v['unretrieved_annotations'];v['documented_positive_coverage']=tp/v['official_annotations'];v['annotation_agreement_precision']=tp/(tp+up) if tp+up else None;v['annotation_agreement_recall']=tp/(tp+ua);v['annotation_agreement_f1']=2*tp/(2*tp+up+ua)
(P/'summary.json').write_text(json.dumps(stats,indent=2));print('OFFICIAL-ONLY ALL COMPLETE',json.dumps(stats),flush=True)
