import argparse
import json
import platform
import time
import importlib.metadata
from pathlib import Path
import faiss
import numpy as np
import pandas as pd
import torch
from torch import nn
from scipy.sparse import csr_matrix
from sklearn.metrics import roc_auc_score, log_loss
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .data import prepare, sample_pairs
from .models import TwoTower, DeepFM
from .metrics import ranking_metrics


def save_json(p,obj):
    p.write_text(json.dumps(obj,indent=2,ensure_ascii=False),encoding='utf-8')


def histories(df):
    return {int(u):set(g.i.astype(int)) for u,g in df.groupby('u')}


def features(pairs,buckets):
    return torch.tensor(np.column_stack([pairs,buckets[pairs[:,1]]]),dtype=torch.long)


def masked_nce(model,u,i,seen,temp,mask=True):
    logits=model.users(u)@model.items(i).T/temp
    if mask:
        # Other known positives are not negatives; each diagonal remains its positive.
        excluded=torch.tensor([[int(j) in seen[int(v)] for j in i] for v in u],dtype=torch.bool)
        excluded.fill_diagonal_(False)
        logits=logits.masked_fill(excluded,-1e9)
    return nn.functional.cross_entropy(logits,torch.arange(len(u)))


def run(root,c):
    start=time.perf_counter(); torch.set_num_threads(c['threads']);faiss.omp_set_num_threads(c['threads'])
    torch.manual_seed(c['seed']);np.random.seed(c['seed'])
    torch.use_deterministic_algorithms(True)
    art=root/'artifacts';art.mkdir(exist_ok=True)
    splits,eda,manifest=prepare(root,c);train,val,test=splits
    nu,ni=eda['users'],eda['items'];seen=histories(train)
    if min(nu,ni,len(val),len(test))==0: raise ValueError('Empty training or evaluation cohort')
    pop=np.bincount(train.i,minlength=ni).astype(np.float32)
    buckets=np.minimum(9,(np.log1p(pop)/max(np.log1p(pop).max(),1)*9).astype(int))
    matrix=csr_matrix((np.ones(len(train)),(train.u,train.i)),shape=(nu,ni),dtype=np.float32)
    co=(matrix.T@matrix).toarray();norm=np.sqrt(pop[:,None]*pop[None,:]);sim=co/np.maximum(norm,1);np.fill_diagonal(sim,0)
    # Positive user-item membership used by the in-batch false-negative mask.
    metrics={};logs=[];times={};selected={}
    def evaluate_scores(score_fn,df):
        truth=histories(df);pred={}
        for u in truth:
            scores=score_fn(u).copy();scores[list(seen[u])]=-np.inf
            pred[u]=np.argsort(-scores,kind='stable')[:max(c['ks'])]
        return ranking_metrics(pred,truth,c['ks'],ni,pop)
    for name,fn in [('Popularity',lambda u:pop),('ItemCF',lambda u:np.asarray(matrix[u]@sim).ravel())]:
        t=time.perf_counter();metrics[name]={'val':evaluate_scores(fn,val),'test':evaluate_scores(fn,test)};times[name]=time.perf_counter()-t
    pairs=torch.tensor(train[['u','i']].to_numpy(),dtype=torch.long)
    best_model=None
    for name,mask in [('TwoTower',True),('TwoTower_no_positive_mask',False)]:
        torch.manual_seed(c['seed']);model=TwoTower(nu,ni,c['dim']);opt=torch.optim.Adam(model.parameters(),lr=c['lr'],weight_decay=c['weight_decay'])
        best=-1;t=time.perf_counter()
        for epoch in range(1,c['epochs']+1):
            model.train();total=0
            for idx in torch.randperm(len(pairs)).split(c['batch_size']):
                u,i=pairs[idx].T;opt.zero_grad();loss=masked_nce(model,u,i,seen,c['temperature'],mask);loss.backward();opt.step();total+=loss.item()*len(idx)
            model.eval()
            with torch.no_grad(): U=model.users(torch.arange(nu)).numpy();V=model.items(torch.arange(ni)).numpy()
            vm=evaluate_scores(lambda u:U[u]@V.T,val);logs.append({'model':name,'epoch':epoch,'loss':total/len(pairs),'val_ndcg20':vm['NDCG@20']})
            print(json.dumps(logs[-1]),flush=True)
            if vm['NDCG@20']>best:
                best=vm['NDCG@20'];selected[name]=epoch;torch.save(model.state_dict(),art/f'{name}.pt')
        model.load_state_dict(torch.load(art/f'{name}.pt',weights_only=True));model.eval()
        with torch.no_grad(): U=model.users(torch.arange(nu)).numpy();V=model.items(torch.arange(ni)).numpy()
        metrics[name]={'val':evaluate_scores(lambda u:U[u]@V.T,val),'test':evaluate_scores(lambda u:U[u]@V.T,test)};times[name]=time.perf_counter()-t
        if mask: best_model=model;np.save(art/'user_vectors.npy',U);np.save(art/'item_vectors.npy',V)
    U=np.load(art/'user_vectors.npy');V=np.load(art/'item_vectors.npy')
    index=faiss.IndexFlatIP(c['dim']);index.add(np.ascontiguousarray(V));faiss.write_index(index,str(art/'books.faiss'))
    # Fixed sampled validation/test protocols. Other positives in the same split are excluded.
    def evaluation_pairs(df,seed):
        blocked={u:seen[u]|histories(df).get(u,set()) for u in seen}
        return sample_pairs(df,blocked,ni,c['eval_negatives'],np.random.default_rng(seed))
    trp,try_=sample_pairs(train,seen,ni,c['negatives'],np.random.default_rng(c['seed']))
    vp,vy=evaluation_pairs(val,c['seed']+1);tp,ty=evaluation_pairs(test,c['seed']+2)
    np.savez_compressed(art/'ranking_samples.npz',train_pairs=trp,train_y=try_,val_pairs=vp,val_y=vy,test_pairs=tp,test_y=ty)
    X=features(trp,buckets);Y=torch.tensor(try_);VX=features(vp,buckets);TX=features(tp,buckets)
    def predict(m,x):
        with torch.no_grad(): return torch.cat([m(b).sigmoid() for b in x.split(8192)]).numpy()
    rank_model=None
    for name,deep in [('DeepFM',True),('FM_no_deep',False)]:
        torch.manual_seed(c['seed']);m=DeepFM(nu,ni,c['dim'],deep);opt=torch.optim.Adam(m.parameters(),lr=c['lr'],weight_decay=c['weight_decay']);best=float('inf');t=time.perf_counter()
        for epoch in range(1,c['rank_epochs']+1):
            m.train();total=0
            for idx in torch.randperm(len(X)).split(c['batch_size']):
                opt.zero_grad();loss=nn.functional.binary_cross_entropy_with_logits(m(X[idx]),Y[idx]);loss.backward();opt.step();total+=loss.item()*len(idx)
            m.eval();pv=predict(m,VX);vl=log_loss(vy,pv)
            logs.append({'model':name,'epoch':epoch,'loss':total/len(X),'val_logloss':vl});print(json.dumps(logs[-1]),flush=True)
            if vl<best:best=vl;selected[name]=epoch;torch.save(m.state_dict(),art/f'{name}.pt')
        m.load_state_dict(torch.load(art/f'{name}.pt',weights_only=True));m.eval()
        metrics[name]={s:{'AUC':float(roc_auc_score(y,predict(m,x))),'LogLoss':float(log_loss(y,predict(m,x))),'samples':len(y),'positive_rate':float(y.mean())} for s,x,y in [('val',VX,vy),('test',TX,ty)]};times[name]=time.perf_counter()-t
        if deep:rank_model=m
    def candidate_ids(u):
        n=min(ni,c['candidate_k']+len(seen[u]));_,ids=index.search(U[u:u+1],n)
        return np.asarray([int(i) for i in ids[0] if i>=0 and i not in seen[u]][:c['candidate_k']],dtype=int)
    candidate_stats={}
    for split,df in [('val',val),('test',test)]:
        truth=histories(df);pred={};rec=[]
        for u,target in truth.items():
            ids=candidate_ids(u);p=np.column_stack([np.full(len(ids),u),ids]);scores=predict(rank_model,features(p,buckets));pred[u]=ids[np.argsort(-scores,kind='stable')]
            rec.append(len(set(ids)&target)/len(target))
        candidate_stats[split]=float(np.mean(rec));metrics.setdefault('TwoTower_FAISS_DeepFM',{})[split]=ranking_metrics(pred,truth,c['ks'],ni,pop)
    lat=[]
    for u in list(seen)[:200]:
        t=time.perf_counter();ids=candidate_ids(u);predict(rank_model,features(np.column_stack([np.full(len(ids),u),ids]),buckets));lat.append((time.perf_counter()-t)*1000)
    _,ids=index.search(U[:min(100,nu)],20);exact=np.argsort(-(U[:min(100,nu)]@V.T),axis=1)[:,:20]
    env={'python':platform.python_version(),'platform':platform.platform(),'processor':platform.processor(),'device':'cpu','torch_cuda_available':torch.cuda.is_available(),'packages':{p:importlib.metadata.version(p) for p in ['torch','faiss-cpu','numpy','pandas','scipy','scikit-learn','fastapi','uvicorn','matplotlib','pytest','httpx']}}
    save_json(art/'config.json',c);save_json(art/'eda.json',eda);save_json(art/'data_manifest.json',manifest);save_json(art/'environment.json',env)
    save_json(art/'metrics.json',metrics);save_json(art/'training_summary.json',{'selected_epochs':selected,'seconds':times,'total_seconds':time.perf_counter()-start,'candidate_recall@100':candidate_stats,'faiss_exact_set_overlap@20':float(np.mean([len(set(a)&set(b))/20 for a,b in zip(ids,exact)])),'warm_pipeline_latency_ms':{'p50':float(np.percentile(lat,50)),'p95':float(np.percentile(lat,95)),'queries':len(lat)},'rank_train_samples':len(Y),'rank_val_samples':len(vy),'rank_test_samples':len(ty)})
    pd.DataFrame(logs).to_csv(art/'training_log.csv',index=False)
    save_json(art/'serving.json',{'users':nu,'items':ni,'seen':{u:sorted(v) for u,v in seen.items()},'popularity':pop.tolist(),'buckets':buckets.tolist()})
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    for name,g in pd.DataFrame(logs).groupby('model'):axes[0 if name.startswith('Two') else 1].plot(g.epoch,g.loss,label=name)
    for ax in axes:ax.set_xlabel('Epoch');ax.set_ylabel('Training loss');ax.legend()
    fig.tight_layout();fig.savefig(art/'loss_curves.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(13,4));axes[0].bar(list(eda['rating_distribution']),list(eda['rating_distribution'].values()));axes[0].set_title('Raw rating distribution');axes[1].hist(train.groupby('u').size(),bins=40,log=True);axes[1].set_title('Train interactions per user');axes[2].plot(np.arange(1,ni+1)/ni,np.cumsum(np.sort(pop)[::-1])/pop.sum());axes[2].set_title('Item popularity concentration');fig.tight_layout();fig.savefig(art/'eda.png',dpi=150);plt.close(fig)
    rows=[{'model':n,**m['test']} for n,m in metrics.items()];pd.DataFrame(rows).to_csv(art/'comparison.csv',index=False)
    print(json.dumps({'eda':eda,'metrics':metrics},ensure_ascii=False),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--config',default='configs/default.json');args=parser.parse_args()
    root=Path(__file__).resolve().parents[1];run(root,json.loads((root/args.config).read_text()))
