"""LOC span/type scoring, independent of entity spelling correctness.

Projection certifies agreement between optimal edit alignments only; it is not
a certificate that the projected OCR-side annotations have been human checked.
"""
from collections import Counter
from projection_stability import certificate,distances

def typed_counts(projected_gold,predicted):
    """Unprojectable gold stays in FN; require exact bounds AND exact type."""
    gs=Counter((m['start'],m['end'],m['type']) for m in projected_gold if m['projection_unique'])
    ps=Counter((m['start'],m['end'],m['type']) for m in predicted)
    tp=sum((gs&ps).values())
    return dict(tp=tp,fp=len(predicted)-tp,fn=len(projected_gold)-tp)

def score_row(gold,prediction):
    assert gold['line_id']==prediction['line_id']
    text=prediction['text'];projected=[]
    tables=distances(gold['text'],text) if gold['mentions'] else None
    for m in gold['mentions']:
        assert m['type']=='LOC'
        c=certificate(gold['text'],text,m['start'],m['end'],tables)
        a,b=c['boundary'] if c['unique_boundary'] else (None,None)
        projected.append(dict(source_start=m['start'],source_end=m['end'],start=a,end=b,type=m['type'],projection_unique=c['unique_boundary']))
    for m in prediction['mentions']:
        assert 0<=m['start']<m['end']<=len(text) and m['type']=='LOC'
    assert len({(m['start'],m['end'],m['type']) for m in prediction['mentions']})==len(prediction['mentions'])
    return typed_counts(projected,prediction['mentions']),projected

def metric(tp,fp,fn):
    return dict(tp=tp,fp=fp,fn=fn,precision=tp/(tp+fp) if tp+fp else 0.,recall=tp/(tp+fn) if tp+fn else 0.,f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.)

def distance(a,b):
    if len(a)>len(b):a,b=b,a
    previous=list(range(len(a)+1))
    for i,y in enumerate(b,1):
        current=[i]
        for j,x in enumerate(a,1):current.append(min(previous[j]+1,current[j-1]+1,previous[j-1]+(x!=y)))
        previous=current
    return previous[-1]

def sanity_checks():
    # A real OCR spelling difference does not invalidate an exact LOC boundary.
    g=dict(line_id='synthetic',text='Zürich.',mentions=[dict(start=0,end=6,type='LOC')])
    p=dict(line_id='synthetic',text='Zurich.',mentions=[dict(start=0,end=6,type='LOC')])
    assert score_row(g,p)[0]==dict(tp=1,fp=0,fn=0)
    gold=[dict(start=0,end=6,type='LOC',projection_unique=True)]
    assert typed_counts(gold,[dict(start=0,end=6,type='PER')])==dict(tp=0,fp=1,fn=1)
    assert typed_counts(gold,[dict(start=0,end=7,type='LOC')])==dict(tp=0,fp=1,fn=1)
    ambiguous=[dict(start=None,end=None,type='LOC',projection_unique=False)]
    assert typed_counts(ambiguous,[])==dict(tp=0,fp=0,fn=1)
    assert typed_counts(gold,[dict(start=0,end=6,type='LOC')]*2)==dict(tp=1,fp=1,fn=0)
    return dict(status='PASS',spelling_separate_from_NER=True,type_mismatch_is_error=True,boundary_mismatch_is_error=True,unprojectable_gold_kept=True,duplicate_predictions_cannot_inflate_TP=True)
