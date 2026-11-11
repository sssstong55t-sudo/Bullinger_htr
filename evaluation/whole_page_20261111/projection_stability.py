"""Certify projected entity boundaries across ALL optimal edit alignments.

The certificate concerns string alignment only. It does not replace human
annotation of OCR text and does not certify a correct geographic identity.
"""
def distances(ref,hyp):
    n,m=len(ref),len(hyp);f=[list(range(m+1))]
    for i,x in enumerate(ref):
        row=[i+1]
        for j,y in enumerate(hyp):row.append(min(f[-1][j]+(x!=y),f[-1][j+1]+1,row[-1]+1))
        f.append(row)
    b=[[0]*(m+1) for _ in range(n+1)]
    for i in range(n,-1,-1):
        for j in range(m,-1,-1):
            if i==n and j==m:continue
            choices=[]
            if i<n:choices.append(1+b[i+1][j])
            if j<m:choices.append(1+b[i][j+1])
            if i<n and j<m:choices.append((ref[i]!=hyp[j])+b[i+1][j+1])
            b[i][j]=min(choices)
    return f,b

def certificate(ref,hyp,a,z,tables=None):
    n,m=len(ref),len(hyp);f,b=tables if tables is not None else distances(ref,hyp);best=f[n][m]
    empty=[[False]*(m+1) for _ in range(n+1)];ranges=[[None]*(m+1) for _ in range(n+1)];empty[0][0]=True
    def merge(i,j,x):
        old=ranges[i][j]
        ranges[i][j]=x if old is None else (min(old[0],x[0]),max(old[1],x[1]),min(old[2],x[2]),max(old[3],x[3]))
    for i in range(n+1):
        for j in range(m+1):
            e=empty[i][j];margins=ranges[i][j]
            if not e and margins is None:continue
            edges=[]
            if i<n:edges.append((i+1,j,1,False))
            if j<m:edges.append((i,j+1,1,False))
            if i<n and j<m:edges.append((i+1,j+1,int(ref[i]!=hyp[j]),a<=i<z))
            for u,v,c,inside in edges:
                if f[i][j]+c+b[u][v]!=best:continue
                if inside:
                    starts=[]
                    if e:starts.append(j)
                    if margins is not None:starts.extend(margins[:2])
                    merge(u,v,(min(starts),max(starts),j+1,j+1))
                else:
                    empty[u][v] |= e
                    if margins is not None:merge(u,v,margins)
    x=ranges[n][m];unique=not empty[n][m] and x is not None and x[0]==x[1] and x[2]==x[3]
    return dict(unique_boundary=unique,boundary=[x[0],x[2]] if unique else None,
        some_optimal_path_deletes_whole_entity=empty[n][m],extrema=x)

if __name__=='__main__':
    import json,random
    rng=random.Random(20261109);tested=0
    for _ in range(600):
        ref=''.join(rng.choice('ab ') for _ in range(rng.randrange(1,5)));hyp=''.join(rng.choice('ab ') for _ in range(rng.randrange(5)));a=rng.randrange(len(ref));z=rng.randrange(a+1,len(ref)+1);f,b=distances(ref,hyp);n,m=len(ref),len(hyp);out=[]
        def enumerate_paths(i,j,positions):
            if i==n and j==m:out.append((min(positions),max(positions)+1) if positions else None);return
            edges=[]
            if i<n:edges.append((i+1,j,1,False))
            if j<m:edges.append((i,j+1,1,False))
            if i<n and j<m:edges.append((i+1,j+1,int(ref[i]!=hyp[j]),a<=i<z))
            for u,v,c,inside in edges:
                if f[i][j]+c+b[u][v]==f[n][m]:enumerate_paths(u,v,positions+([j] if inside else []))
        enumerate_paths(0,0,[]);poss=set(out);result=certificate(ref,hyp,a,z,(f,b));unique=len(poss)==1 and None not in poss
        assert result['unique_boundary']==unique,(ref,hyp,a,z,poss,result)
        if unique:assert tuple(result['boundary'])==next(iter(poss))
        tested+=1
    print(json.dumps(dict(status='PASS',exhaustive_alignment_crosschecks=tested,no_models_loaded=True)))
