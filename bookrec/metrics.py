import numpy as np


def ranking_metrics(predictions, truth, ks, nitems, popularity):
    result={}
    for k in ks:
        recall=[];hit=[];ndcg=[];recommended=[]
        for u, target in truth.items():
            top=list(predictions[u][:k]); relevant=np.array([i in target for i in top],dtype=float)
            recall.append(relevant.sum()/len(target));hit.append(float(relevant.any()))
            ideal=(1/np.log2(np.arange(min(k,len(target)))+2)).sum()
            ndcg.append(float((relevant/np.log2(np.arange(len(top))+2)).sum()/ideal))
            recommended.extend(top)
        result.update({f'Recall@{k}':float(np.mean(recall)),f'HitRate@{k}':float(np.mean(hit)),f'NDCG@{k}':float(np.mean(ndcg)),f'Coverage@{k}':len(set(recommended))/nitems,f'MeanPopularity@{k}':float(np.mean(popularity[recommended]))})
    result['users']=len(truth)
    return result
