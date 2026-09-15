import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from bookrec.api import app

ROOT=Path(__file__).resolve().parents[1]
pytestmark=pytest.mark.skipif(not (ROOT/'artifacts/metrics.json').exists(),reason='Run pipeline first')


def test_temporal_and_mapping():
    frames=[pd.read_csv(ROOT/f'data/processed/{s}.csv') for s in ['train','val','test']]
    a,b,c=frames
    assert pd.to_datetime(a.time).max()<pd.to_datetime(b.time).min()
    assert pd.to_datetime(b.time).max()<pd.to_datetime(c.time).min()
    assert set(b.u)<=set(a.u) and set(c.i)<=set(a.i)
    pairs=[set(zip(x.u,x.i)) for x in frames]
    assert not (pairs[0]&pairs[1] or pairs[0]&pairs[2] or pairs[1]&pairs[2])
    assert a.groupby('u').size().min()>=5 and a.groupby('i').size().min()>=5


def test_api_history_and_cold_start():
    s=json.loads((ROOT/'artifacts/serving.json').read_text())
    with TestClient(app) as client:
        assert client.get('/health').status_code==200
        r=client.get('/recommend?user_id=0&k=10');assert r.status_code==200
        ids=[x['item_id'] for x in r.json()['items']]
        assert len(ids)==len(set(ids))==10
        assert not set(ids)&set(s['seen']['0'])
        assert client.get('/recommend?user_id=999999').json()['strategy']=='popularity_cold_start'
        assert client.get('/recommend?k=51').status_code==422
        assert client.get('/recommend?user_id=-1').status_code==422
        assert 'BookRec' in client.get('/').text


def test_faiss_exact():
    import faiss
    a=ROOT/'artifacts';u=np.load(a/'user_vectors.npy');v=np.load(a/'item_vectors.npy')
    index=faiss.read_index(str(a/'books.faiss'));scores,ids=index.search(u[:10],10)
    expected=np.sort(u[:10]@v.T,axis=1)[:,-10:][:,::-1]
    assert np.allclose(scores,expected,atol=1e-6)
