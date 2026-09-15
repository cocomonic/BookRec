import gzip
import hashlib
import json
import urllib.request
from pathlib import Path
import numpy as np
import pandas as pd

BASE = 'https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/'


def download(root):
    raw = root / 'data/raw'
    raw.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for name in ['goodreads_reviews_poetry.json.gz', 'goodreads_books_poetry.json.gz']:
        p = raw / name
        if not p.exists():
            tmp = p.with_suffix('.partial')
            with urllib.request.urlopen(BASE + name, timeout=120) as src, tmp.open('wb') as dst:
                while chunk := src.read(1024*1024):
                    dst.write(chunk)
            tmp.replace(p)
        manifest[name] = {'url': BASE + name, 'bytes': p.stat().st_size, 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}
    return manifest


def kcore(df, k):
    while True:
        out = df[df.groupby('user_id').user_id.transform('size').ge(k) & df.groupby('book_id').book_id.transform('size').ge(k)]
        if len(out) == len(df):
            return out.copy()
        df = out


def prepare(root, c):
    manifest = download(root)
    rows = []
    with gzip.open(root/'data/raw/goodreads_reviews_poetry.json.gz', 'rt', encoding='utf-8') as f:
        for line in f:
            r = json.loads(line)
            rows.append({k:r.get(k) for k in ['user_id','book_id','rating','date_added']})
    df = pd.DataFrame(rows)
    raw_n = len(df)
    ratings = df.rating.value_counts().sort_index().to_dict()
    df['time'] = pd.to_datetime(df.date_added, format='%a %b %d %H:%M:%S %z %Y', errors='coerce', utc=True)
    invalid = int(df.time.isna().sum())
    df = df.dropna(subset=['user_id','book_id','time','rating']).sort_values(['time','user_id','book_id'])
    df = df.drop_duplicates(['user_id','book_id'], keep='first')
    dedup_n = len(df)
    df = df[df.rating >= c['positive_rating']].copy()
    t1, t2 = df.time.quantile([c['train_quantile'],c['val_quantile']])
    train = kcore(df[df.time < t1], c['k_core'])
    users = {x:i for i,x in enumerate(sorted(train.user_id.unique()))}
    items = {x:i for i,x in enumerate(sorted(train.book_id.unique()))}
    parts = [train,df[(df.time>=t1)&(df.time<t2)],df[df.time>=t2]]
    stats = {'raw_reviews':raw_n,'rating_distribution':ratings,'invalid_dates':invalid,'deduplicated_reviews':dedup_n,'positive_reviews':len(df),'train_cutoff':str(t1),'test_cutoff':str(t2),'users':len(users),'items':len(items),'split':{}}
    out = root/'data/processed'; out.mkdir(parents=True,exist_ok=True)
    mapped=[]
    for name,p in zip(['train','val','test'],parts):
        eligible=p.user_id.isin(users)&p.book_id.isin(items)
        stats['split'][name]={'before_warm_filter':len(p),'warm_interactions':int(eligible.sum()),'unknown_users':int((~p.user_id.isin(users)).sum()),'unknown_items':int((~p.book_id.isin(items)).sum())}
        p=p[eligible].copy();p['u']=p.user_id.map(users);p['i']=p.book_id.map(items)
        p[['u','i','time','rating']].to_csv(out/f'{name}.csv',index=False)
        mapped.append(p)
    books={}
    with gzip.open(root/'data/raw/goodreads_books_poetry.json.gz','rt',encoding='utf-8') as f:
        for line in f:
            b=json.loads(line)
            if b['book_id'] in items:
                books[items[b['book_id']]]={'book_id':b['book_id'],'title':b.get('title',''),'authors':b.get('authors',[])}
    (out/'mapping.json').write_text(json.dumps({'users':users,'items':items,'books':books},ensure_ascii=False),encoding='utf-8')
    counts=train.book_id.value_counts()
    stats.update({'density':len(train)/len(users)/len(items),'train_interactions':len(train),'train_top_1pct_share':float(counts.head(max(1,len(items)//100)).sum()/len(train))})
    return mapped, stats, manifest


def sample_pairs(df, seen, nitems, negatives, rng):
    rows=[]; labels=[]
    for u,i in df[['u','i']].itertuples(index=False,name=None):
        rows.append((u,i));labels.append(1)
        pool=np.setdiff1d(np.arange(nitems),list(seen.get(u,set()) | {i}),assume_unique=False)
        if not len(pool):
            continue
        for j in rng.choice(pool,size=min(negatives,len(pool)),replace=False):
            rows.append((u,int(j)));labels.append(0)
    return np.asarray(rows,dtype=np.int64),np.asarray(labels,dtype=np.float32)
