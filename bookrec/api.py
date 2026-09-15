"""Run with: python -m uvicorn bookrec.api:app --host 127.0.0.1 --port 8000."""
from contextlib import asynccontextmanager
from pathlib import Path
import json
import time
import faiss
import numpy as np
import torch
from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse
from .models import DeepFM

ROOT=Path(__file__).resolve().parents[1]


class Recommender:
    def __init__(self,root=ROOT):
        a=root/'artifacts';self.c=json.loads((a/'config.json').read_text());torch.set_num_threads(self.c['threads']);faiss.omp_set_num_threads(self.c['threads'])
        self.s=json.loads((a/'serving.json').read_text());self.books=json.loads((root/'data/processed/mapping.json').read_text(encoding='utf-8'))['books']
        self.U=np.load(a/'user_vectors.npy');self.index=faiss.read_index(str(a/'books.faiss'));self.buckets=np.array(self.s['buckets'])
        self.rank=DeepFM(self.s['users'],self.s['items'],self.c['dim']);self.rank.load_state_dict(torch.load(a/'DeepFM.pt',map_location='cpu',weights_only=True));self.rank.eval()

    def recommend(self,user_id,k):
        start=time.perf_counter();seen=set(self.s['seen'].get(str(user_id),[]));cold=not 0<=user_id<self.s['users']
        if cold:
            ids=np.argsort(-np.array(self.s['popularity']),kind='stable')[:k];scores=np.array(self.s['popularity'])[ids];strategy='popularity_cold_start'
        else:
            _,candidates=self.index.search(self.U[user_id:user_id+1],min(self.s['items'],self.c['candidate_k']+len(seen)))
            ids=np.array([i for i in candidates[0] if i>=0 and i not in seen][:self.c['candidate_k']],dtype=int)
            x=torch.tensor(np.column_stack([np.full(len(ids),user_id),ids,self.buckets[ids]]),dtype=torch.long)
            with torch.no_grad():scores=self.rank(x).sigmoid().numpy()
            order=np.argsort(-scores,kind='stable')[:k];ids=ids[order];scores=scores[order];strategy='two_tower_faiss_deepfm'
        return {'user_id':user_id,'strategy':strategy,'score_meaning':'training interaction count' if cold else 'sampled-negative model score; not calibrated CTR','latency_ms':(time.perf_counter()-start)*1000,'items':[{'item_id':int(i),**self.books.get(str(i),{'title':f'Book {i}'}),'score':float(s)} for i,s in zip(ids,scores)]}


@asynccontextmanager
async def lifespan(app):
    app.state.rec=Recommender()
    yield

app=FastAPI(title='BookRec',version='1.0.0',lifespan=lifespan)


@app.get('/health')
def health():return {'status':'ok','users':app.state.rec.s['users'],'items':app.state.rec.s['items']}


@app.get('/recommend')
def recommend(user_id:int=Query(0,ge=0),k:int=Query(10,ge=1,le=50)):
    return app.state.rec.recommend(user_id,k)


@app.get('/',response_class=HTMLResponse)
def home():return (ROOT/'bookrec/web.html').read_text(encoding='utf-8')
