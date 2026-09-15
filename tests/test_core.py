import numpy as np
import pandas as pd
import torch
from bookrec.data import kcore,sample_pairs
from bookrec.metrics import ranking_metrics
from bookrec.models import DeepFM,TwoTower
from bookrec.pipeline import masked_nce


def test_kcore_cascade():
    x=pd.DataFrame({'user_id':['a','a','b'],'book_id':['x','y','y']})
    assert kcore(x,2).empty


def test_metrics_hand_calculated():
    m=ranking_metrics({0:[2,1,3]},{0:{1,2}},[2],4,np.ones(4))
    assert m['Recall@2']==m['HitRate@2']==m['NDCG@2']==1
    assert m['Coverage@2']==0.5


def test_negatives_exclude_history():
    pairs,y=sample_pairs(pd.DataFrame({'u':[0],'i':[1]}),{0:{1,2}},5,4,np.random.default_rng(42))
    assert set(pairs[y==0,1])=={0,3,4}


def test_mask_all_positive_batch():
    m=TwoTower(1,2,4)
    loss=masked_nce(m,torch.tensor([0,0]),torch.tensor([0,1]),{0:{0,1}},0.1)
    assert abs(loss.item())<1e-6


def test_deepfm_finite_gradients():
    m=DeepFM(3,4,8);loss=m(torch.tensor([[0,1,2],[1,2,3]])).sum();loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in m.parameters())
