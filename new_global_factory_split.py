import sys
import numpy as np
sys.path.append("..")
import my_graph as graph
from copy import deepcopy
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from itertools import chain
import psutil
import time
import pickle
import gc
from multiprocessing import Pool
from functools import reduce

from args import args
from gf_utils import *

np.set_printoptions(threshold=sys.maxsize)
np.set_printoptions(linewidth=np.nan)

root = "../gfiles/"
# sysargs = sys.argv[1:]
# args = opts(sysargs)

# use_cuda = torch.cuda.is_available()
use_cuda = (args.use_cuda>0)
device = torch.device("cuda:"+str(args.device) if use_cuda else "cpu")
if use_cuda:
    torch.cuda.set_device(args.device)

def job_graphloading(z):
    gpath, savepath, args = z
    if os.path.exists(savepath):
        print('loading old graph from', savepath)
        gr = graph.Graph(gpath=gpath, from_raw=False, args=args)
        indices, values, size = gr.loadFromFile(savepath)
        return gr, indices, values, size
    else:
        print('generating new graph from', gpath)
        gr = graph.Graph(gpath=gpath, from_raw=True, args=args)
        indices, values, size = gr.get_sparse_component()
        if savepath != 'NONE':
            print('dumping graph to', savepath)
            gr.dumpToFile(savepath, indices, values, size)
        return gr, indices, values, size

def job_crossloss(t1, t2, gf):
    loss = 0
    len_real_t1 = len(t1)
    len_real_t2 = len(t2)
    org_w1 = gf.typeToGlobalGraphs[t1].w_sparse
    org_w2 = gf.typeToGlobalGraphs[t2].w_sparse
    org_beta = gf.beta[t1+'#'+t2]
    crossTypeToPred = gf.crossTypeToPred
    crossUnorderToPred = gf.crossUnorderToPred
    pred2idx_t1 = gf.typeToGlobalGraphs[t1].pred2idx
    pred2idx_t2 = gf.typeToGlobalGraphs[t2].pred2idx

    tmp = t1.split('#')
    rt1 = tmp[0]+'_1#'+tmp[1]+'_2' if tmp[0]==tmp[1] else t1
    tmp = t2.split('#')
    rt2 = tmp[0]+'_1#'+tmp[1]+'_2' if tmp[0]==tmp[1] else t2
    might_t1 = [rt1, type_reverse(rt1)]
    might_t2 = [rt2, type_reverse(rt2)]
    org_w1_grad = torch.sparse.FloatTensor(torch.empty([2, 0]).long(), torch.empty([0]), org_w1.size()).to(device)
    org_w2_grad = torch.sparse.FloatTensor(torch.empty([2, 0]).long(), torch.empty([0]), org_w2.size()).to(device)
    org_beta_grad = torch.sparse.FloatTensor(torch.empty([2, 0]).long(), torch.empty([0]), org_beta.size()).to(device)
    for real_t1 in might_t1:
        for real_t2 in might_t2:
            if real_t1 not in crossTypeToPred or real_t2 not in crossTypeToPred:
                continue
            if t1 not in crossUnorderToPred or t2 not in crossUnorderToPred:
                continue
            preds = crossTypeToPred[real_t1] & crossTypeToPred[real_t2]
            if len(preds) == 0:
                continue
            jPreds = crossUnorderToPred[t1] & crossUnorderToPred[t2]
            if len(jPreds) == 0:
                continue
            jPreds_typed = [(p+'#'+jreal_t1, p+'#'+jreal_t2) for p in jPreds for jreal_t1 in might_t1 for jreal_t2 in might_t2 if (p+'#'+jreal_t1 in pred2idx_t1) and (p+'#'+jreal_t2 in pred2idx_t2) and (real_t1==jreal_t1)==(real_t2==jreal_t2)]

            pid1 = torch.LongTensor([pred2idx_t1[p+'#'+real_t1] for p in preds]).to(device)
            pid2 = torch.LongTensor([pred2idx_t2[p+'#'+real_t2] for p in preds]).to(device)
            j_pid1, j_pid2 = list(zip(*jPreds_typed))
            j_pid1 = torch.LongTensor([pred2idx_t1[p] for p in j_pid1]).to(device)
            j_pid2 = torch.LongTensor([pred2idx_t2[p] for p in j_pid2]).to(device)

            w1 = my_index_select(org_w1, index=pid1, dim=0).coalesce()
            w1 = my_index_select(w1, index=j_pid1, dim=1)
            w2 = my_index_select(org_w2, index=pid2, dim=0).coalesce()
            w2 = my_index_select(w2, index=j_pid2, dim=1)

            small_beta = my_index_select(org_beta, index=pid1, dim=0).coalesce()
            small_beta = my_index_select(small_beta, index=pid2, dim=1)
            small_eye = torch.sparse.FloatTensor(torch.arange(len(preds)).repeat(2,1), torch.ones(len(preds))).to(device)
            small_loss = torch.sparse.mm(small_eye * small_beta, (w1-w2)**2)
            loss += torch.sparse.sum(small_loss)
            assert loss >= 0

            #### MANUALLY BACKWARD START
            # Now trying to backward cross loss manually !
            small_w_grad = torch.sparse.mm(small_eye * small_beta, w1-w2).coalesce()
            id_i, id_j = small_w_grad.indices()
            ida = pid1.index_select(dim=0,index=id_i)
            idb = j_pid1.index_select(dim=0,index=id_j)
            grad = torch.sparse.FloatTensor(torch.stack((ida,idb),dim=0), small_w_grad.values(), org_w1.size())#.to(device)
            org_w1_grad += grad
            ida = pid2.index_select(dim=0,index=id_i)
            idb = j_pid2.index_select(dim=0,index=id_j)
            grad = torch.sparse.FloatTensor(torch.stack((ida,idb),dim=0), small_w_grad.values(), org_w2.size())#.to(device)
            org_w2_grad -= grad
            
            small_beta_grad = 0.5*(w1-w2)**2
            if small_beta_grad._nnz() == 0:
                continue
            small_beta_grad = torch.sparse.sum(small_beta_grad,dim=1).coalesce()
            id_i = small_beta_grad.indices()[0]
            ida = pid1.index_select(dim=0,index=id_i)
            idb = pid2.index_select(dim=0,index=id_i)
            grad = torch.sparse.FloatTensor(torch.stack((ida,idb),dim=0), small_beta_grad.values(), org_beta.size())#.to(device)
            org_beta_grad += grad
            #### MANUALLY BACKWARD END

    org_w1_grad = org_w1_grad.coalesce()
    org_w2_grad = org_w2_grad.coalesce()
    org_beta_grad = org_beta_grad.coalesce()
    return loss, (t1, org_w1_grad), (t2, org_w2_grad), (t1+'#'+t2, org_beta_grad)
    
def job_resoloss(t, gf):
    epsilon = gf.epsilon1
    w = gf.typeToGlobalGraphs[t].w_sparse.coalesce()
    fake = torch.sparse.FloatTensor(torch.empty([2, 0]).long(), torch.empty([0]), w.size()).to(device)
    if w._nnz() == 0:
        # print(t, 'resolved')
        return 0, (t, fake)
    w_t = w.t()
    w_indices = w.indices()[:, (w.values()>epsilon)]
    w_indices = w_indices[:, w_indices[0]!=w_indices[1]]
    ij_mask = torch.sparse.FloatTensor(w_indices, torch.ones(w_indices.size(1)).to(device), w.size())#.to(device)
    ij_mask = (ij_mask * ij_mask.t()).coalesce()
    w_indices = ij_mask.indices()
    if ij_mask._nnz() == 0:
        # print(t, 'resolved')
        return 0, (t, fake)

    w2 = w**2
    w2_sum1 = torch.sparse.sum(w2, dim=1)
    w2_sum2 = torch.sparse.sum(w2, dim=0)
    del w2
    # wikwjk = torch.sparse.mm(w, w_t)
    # wkiwkj = torch.sparse.mm(w_t, w)
    del w_t
    w_mask = torch.sparse.FloatTensor(w.indices(), torch.ones(w.indices().size(1)).to(device), w.size())#.to(device)

    # \sum_{ijk\in G,k\ne i,k\ne j}I(w_{ij})I(w_{ji})[(w_{ik}-w_{jk})^2+(w_{ki}-w_{kj})^2]\\
    # =\sum_{ij\in G,i\ne j}I(w_{ij})I(w_{ji})\sum_{k\in G,k\ne i,k\ne j}(w_{ik}^2+w_{jk}^2+w_{ki}^2+w_{kj}^2-2w_{ik}w_{jk}-2w_{ki}w_{kj})\\
    # =\sum_{ij\in G,i\ne j}I(w_{ij})I(w_{ji})(-2(1-w_{ij})^2-2(1-w_{ji})^2\\
    # +\sum_{k\in G}w_{ik}^2+w_{jk}^2+w_{ki}^2+w_{kj}^2-2w_{ik}w_{jk}-2w_{ki}w_{kj})\\
    # =\sum_{ij\in G,i\ne j}I(w_{ij})I(w_{ji})(4(2w_{ij}-1-w_{ij}^2)+\sum_{k\in G}w_{ik}^2+w_{jk}^2+w_{ki}^2+w_{kj}^2-2w_{ik}w_{jk}-2w_{ki}w_{kj})\\

    # square items
    # We need to directly calculate the grad, as index_select used here!
    loss = torch.sum(my_index_select(w2_sum1, index=w_indices[0], dim=0))
    loss += torch.sum(my_index_select(w2_sum1, index=w_indices[1], dim=0))
    loss += torch.sum(my_index_select(w2_sum2, index=w_indices[0], dim=0))
    loss += torch.sum(my_index_select(w2_sum2, index=w_indices[1], dim=0))
    del w2_sum1
    del w2_sum2
    mask_diag = (torch.sparse.sum(ij_mask,dim=1) + torch.sparse.sum(ij_mask,dim=0)).coalesce()
    mask_diag = torch.sparse.FloatTensor(mask_diag.indices().repeat(2,1), mask_diag.values(), w.size())#.to(device)
    grad = torch.sparse.mm(mask_diag, w) + torch.sparse.mm(w, mask_diag)
    del mask_diag

    # k compensation items
    masked_w = w*ij_mask
    loss += 8*torch.sparse.sum(masked_w)-4*torch.sparse.sum(masked_w**2)-4*torch.sparse.sum(ij_mask)
    grad += 4*ij_mask-4*masked_w

    # cross multiply items, the loss is estimated by SVD method to avoid the mm operation
    if w.size(0) > 100:
        u, sigma, v = torch.svd_lowrank(w, q=min(50, w.size(0)))
        u = sigma.unsqueeze(0)*u
        v = sigma.unsqueeze(0)*v
        l1 = sum(map(lambda x: torch.sum(v[x[0]] * v[x[1]]), w_indices.t().tolist()))
        l2 = sum(map(lambda x: torch.sum(u[x[0]] * u[x[1]]), w_indices.t().tolist()))
        loss -= 2*(l1+l2)
    else:
        wikwjk = torch.sparse.mm(w, w.t())
        wkiwkj = torch.sparse.mm(w.t(), w)
        loss -= 2*torch.sparse.sum(wikwjk*ij_mask)
        loss -= 2*torch.sparse.sum(wkiwkj*ij_mask)
        del wikwjk
        del wkiwkj
    del w_indices
    # del ij_mask
    grad -= 2*sparserow_leftmm(ij_mask, w)
    grad -= 2*sparserow_rightmm(w, ij_mask)
    del w_mask
    # del masked_w_sum
    # grad += 2*w
    
    grad = grad.coalesce()
    return loss, (t, grad)

def job_transloss(t, gf):
    # L=\sum_{i\ne k}ReLU(log(\sum_{j\ne i,j\ne k}w_{ij}w_{jk})-logw_{ik})\\
    # \frac{\partial L}{\partial w_{ab}}=-\frac1{w_{ik}}I(ReLU,a\ne b)\quad\quad (ab=ik)\\
    # +\sum_{k\ne a,k\ne b} I(ReLU)\frac{w_{bk}}{\sum_jw_{aj}w_{jk}}\quad\quad (ab=ij)\\
    # +\sum_{i\ne a,i\ne b} I(ReLU)\frac{w_{ia}}{\sum_jw_{ij}w_{jb}}\quad\quad (ab=jk)\\
    # let K=\frac{1}{W\cdot W},D=W^T\\
    # \sum_{k\ne a,k\ne b} I(ReLU)\frac{w_{bk}}{\sum_jw_{aj}w_{jk}} \to \sum_{k}K_{ak}D_{kb}\\
    # \sum_{i\ne a,i\ne b} I(ReLU)\frac{w_{ia}}{\sum_jw_{ij}w_{jb}} \to \sum_k D_{ai}K_{ib}
    epsilon = gf.epsilon_trans
    # w = gr.w_sparse.coalesce()
    w = gf.typeToGlobalGraphs[t].w_sparse.coalesce()

    w_indices = w.indices()
    w_mask = (w.values()>epsilon)*(w_indices[0]!=w_indices[1])
    w_indices = w_indices[:, w_mask]
    w_values = w.values()[w_mask]
    new_w = torch.sparse.FloatTensor(w_indices, w_values, w.size()).coalesce()
    # print('new_w', new_w.to_dense())
    new_w_mask = torch.sparse.FloatTensor(w_indices, torch.ones_like(w_values).to(device), w.size())
    w_log = torch.sparse.FloatTensor(w.indices(), (w.values()+1e-10).log(), w.size())
    K = sparserow_leftmm(new_w, new_w).coalesce()
    # print('K', K.to_dense())
    K_log = torch.sparse.FloatTensor(K.indices(), (K.values()+1e-10).log(), K.size())
    before_relu = (K_log - w_log).coalesce()
    relu_mask = (before_relu.values()>1e-10) * (before_relu.indices()[0]!=before_relu.indices()[1])
    loss = before_relu.values()[relu_mask].sum()
    del K_log, w_log
    
    before_relu = (K - w).coalesce()
    relu_mask = (before_relu.values()>1e-10) * (before_relu.indices()[0]!=before_relu.indices()[1])
    relu_mask = before_relu.indices()[:,relu_mask]
    del before_relu
    relu_mask_tensor = torch.sparse.FloatTensor(relu_mask, torch.ones(relu_mask.size(1)).to(device), w.size())
    del relu_mask
    # print('relu_mask', relu_mask_tensor.to_dense())
    fake_w = (w + 1e-10*relu_mask_tensor).coalesce()
    grad = torch.sparse.FloatTensor(fake_w.indices(), (fake_w.values()+1e-10).reciprocal(), w.size()) * -relu_mask_tensor
    del fake_w
    D = new_w.t()
    K = torch.sparse.FloatTensor(K.indices(), (K.values()+1e-10).reciprocal(), K.size()) * relu_mask_tensor
    # print('grad for wik', grad)
    # print('masked K', K)
    grad2 = sparserow_rightmm(K, D).coalesce()
    grad2 += sparserow_leftmm(D.coalesce(), K.coalesce()).coalesce()
    del D, K
    grad += grad2*new_w_mask
    # grad.values().clip_(min=-10, max=10)
    grad = grad.coalesce()
    try:
        grad.values().clip_(min=-5)
    except Exception as e:
        # print(grad)
        # raise e
        pass
    del grad2

    return loss, (t, (grad*gf.lambda_trans).coalesce())

def job_transloss2(t, gf):
    # L=\sum_{i\ne k,j\ne i,j\ne k}I(w_{ij}>\epsilon,w_{jk}>\epsilon)ReLU(logw_{ij}+logw_{jk}-logw_{ik})
    epsilon = gf.epsilon_trans
    loss = 0
    w = gf.typeToGlobalGraphs[t].w_sparse.coalesce()

    w_cpu = w.cpu()

    w_indices = w.indices()
    w_mask = (w.values()>epsilon)*(w_indices[0]!=w_indices[1])
    w_indices = w_indices[:, w_mask]
    # w_values = w.values()[w_mask]
    ij_mask = torch.sparse.FloatTensor(w_indices, torch.ones(w_indices.size(1)).to(device), w.size()).coalesce()#.to(device)
    del w_indices, w_mask
    new_w = (ij_mask * w).coalesce()
    nonzero_rows = set(ij_mask.indices()[0].tolist()) & set(ij_mask.indices()[1].tolist())
    del ij_mask
    new_w_t = new_w.t().coalesce()

    grad_ik = torch.sparse.FloatTensor(torch.empty([2, 0]).long(), torch.empty([0]), w.size()).to(device)
    grad_jk_indices = []
    grad_jk_values = []
    grad_ij_indices = []
    grad_ij_values = []
    N = w.size(0)
    for n, j in enumerate(nonzero_rows):
        # print(j)
        torch.cuda.empty_cache()
        wij = new_w_t[j].unsqueeze(1).coalesce()
        wjk = new_w[j].unsqueeze(0).coalesce()
        if N > 20000:
            wijid = wij.indices()[0]
            wjkid = wjk.indices()[1]
            wijid = wijid.unsqueeze(1).repeat(1, wjkid.size(0))
            wjkid = wjkid.unsqueeze(0).repeat(wijid.size(0), 1)
            wijv = wij.values().unsqueeze(1)
            wjkv = wjk.values().unsqueeze(0)
            wijwjk = torch.sparse.FloatTensor(torch.stack([wijid, wjkid],dim=0).flatten(start_dim=1), wijv.mm(wjkv).flatten(), w.size())
            del wijid, wjkid, wijv, wjkv
        else:
            wijwjk = torch.sparse.mm(wij, wjk).coalesce()
        ik_to_optimize = (wijwjk - w).coalesce()
        ik_mask = ik_to_optimize.indices()
        ik_mask = ik_mask[:,(ik_to_optimize.values()>1e-10)*(ik_mask[0]!=ik_mask[1])]
        del ik_to_optimize
        ik_mask = torch.sparse.FloatTensor(ik_mask, torch.ones(ik_mask.size(1)).to(device), w.size()).coalesce()
        wik = ik_mask.cpu() * w_cpu
        new_w_processed = (wik.to(device) + 1e-10 * ik_mask).coalesce()
        del wik
        if new_w_processed._nnz() == 0:
            continue
        wijwjk_processed = (wijwjk * ik_mask).coalesce()
        del wijwjk
        loss += wijwjk_processed.values().log().sum() - new_w_processed.values().log().sum()
        grad_ik -= torch.sparse.FloatTensor(new_w_processed.indices(), new_w_processed.values().reciprocal(), grad_ik.size())
        del new_w_processed

        sum_on_i = (torch.sparse.sum(ik_mask, dim=0).unsqueeze(0) * wjk).coalesce()
        sum_indices = sum_on_i.indices()
        sum_indices[0] = j
        grad_jk_indices.append(sum_indices)
        grad_jk_values.append(sum_on_i.values().reciprocal())

        sum_on_k = (torch.sparse.sum(ik_mask, dim=1).unsqueeze(1) * wij).coalesce()
        sum_indices = sum_on_k.indices()
        sum_indices[1] = j
        grad_ij_indices.append(sum_indices)
        grad_ij_values.append(sum_on_k.values().reciprocal())

    if len(grad_jk_indices) > 0:
        grad_ik += torch.sparse.FloatTensor(torch.cat(grad_jk_indices, dim=1), torch.cat(grad_jk_values, dim=0), grad_ik.size())
    if len(grad_ij_indices) > 0:
        grad_ik += torch.sparse.FloatTensor(torch.cat(grad_ij_indices, dim=1), torch.cat(grad_ij_values, dim=0), grad_ik.size())
    grad_ik = grad_ik.coalesce()
    w_mask = grad_ik.indices()[0]!=grad_ik.indices()[1]
    grad_ik = torch.sparse.FloatTensor(grad_ik.indices()[:,w_mask], grad_ik.values()[w_mask], grad_ik.size()).coalesce()

    try:
        grad_ik.values().clip_(min=-5)
    except Exception as e:
        # print(grad)
        # raise e
        pass

    return loss, (t, (grad_ik*gf.lambda_trans).coalesce())

def job_transloss3(t, gf):
    # L=\sum_{i\ne k,j\ne i,j\ne k}I(w_{ij}>\epsilon,w_{jk}>\epsilon,logw_{ij}+logw_{jk}>logw_{ik})-logw_{ik}
    epsilon = gf.epsilon_trans
    loss = 0
    w = gf.typeToGlobalGraphs[t].w_sparse.coalesce()

    w_cpu = w.cpu()

    w_indices = w.indices()
    w_mask = (w.values()>epsilon)*(w_indices[0]!=w_indices[1])
    w_indices = w_indices[:, w_mask]
    ij_mask = torch.sparse.FloatTensor(w_indices, torch.ones(w_indices.size(1)).to(device), w.size()).coalesce()#.to(device)
    del w_indices, w_mask
    new_w = (ij_mask * w).coalesce()
    nonzero_rows = set(ij_mask.indices()[0].tolist()) & set(ij_mask.indices()[1].tolist())
    del ij_mask
    new_w_t = new_w.t().coalesce()

    grad_ik = torch.sparse.FloatTensor(torch.empty([2, 0]).long(), torch.empty([0]), w.size()).to(device)
    N = w.size(0)
    for n, j in enumerate(nonzero_rows):
        torch.cuda.empty_cache()
        wij = new_w_t[j].unsqueeze(1).coalesce()
        wjk = new_w[j].unsqueeze(0).coalesce()
        if N > 20000:
            wijid = wij.indices()[0]
            wjkid = wjk.indices()[1]
            wijid = wijid.unsqueeze(1).repeat(1, wjkid.size(0))
            wjkid = wjkid.unsqueeze(0).repeat(wijid.size(0), 1)
            wijv = wij.values().unsqueeze(1)
            wjkv = wjk.values().unsqueeze(0)
            wijwjk = torch.sparse.FloatTensor(torch.stack([wijid, wjkid],dim=0).flatten(start_dim=1), wijv.mm(wjkv).flatten(), w.size())
            del wijid, wjkid, wijv, wjkv
        else:
            wijwjk = torch.sparse.mm(wij, wjk).coalesce()
        del wij, wjk
        ik_to_optimize = (wijwjk - w).coalesce()
        ik_mask = ik_to_optimize.indices()
        ik_mask = ik_mask[:,(ik_to_optimize.values()>1e-10)*(ik_mask[0]!=ik_mask[1])]
        del ik_to_optimize
        ik_mask = torch.sparse.FloatTensor(ik_mask, torch.ones(ik_mask.size(1)).to(device), w.size()).coalesce()
        wik = ik_mask.cpu() * w_cpu
        new_w_processed = (wik.to(device) + 1e-10 * ik_mask).coalesce()
        del wik
        if new_w_processed._nnz() == 0:
            continue
        wijwjk_processed = (wijwjk * ik_mask).coalesce()
        del wijwjk, ik_mask
        loss += wijwjk_processed.values().log().sum() - new_w_processed.values().log().sum()
        del wijwjk_processed
        grad_ik -= torch.sparse.FloatTensor(new_w_processed.indices(), new_w_processed.values().reciprocal(), grad_ik.size())
        del new_w_processed
        grad_ik = grad_ik.coalesce()

    grad_ik = grad_ik.coalesce()
    w_mask = grad_ik.indices()[0]!=grad_ik.indices()[1]
    grad_ik = torch.sparse.FloatTensor(grad_ik.indices()[:,w_mask], grad_ik.values()[w_mask], grad_ik.size()).coalesce()

    try:
        grad_ik.values().clip_(min=-5)
    except Exception as e:
        # print(grad)
        # raise e
        pass

    return loss, (t, (grad_ik*gf.lambda_trans).coalesce())

def job_transloss4(t, gf):
    # L=\sum_{i\ne k,j\ne i,j\ne k}I(w_{ij}>\epsilon,w_{jk}>\epsilon,logw_{ij}+logw_{jk}>logw_{ik})-w_{ij}w_{jk}logw_{ik}
    epsilon = gf.epsilon_trans
    loss = 0
    w = gf.typeToGlobalGraphs[t].w_sparse.coalesce()

    w_cpu = w.cpu()

    w_indices = w.indices()
    w_mask = (w.values()>epsilon)*(w_indices[0]!=w_indices[1])
    w_indices = w_indices[:, w_mask]
    ij_mask = torch.sparse.FloatTensor(w_indices, torch.ones(w_indices.size(1)).to(device), w.size()).coalesce()#.to(device)
    del w_indices, w_mask
    new_w = (ij_mask * w).coalesce()
    nonzero_rows = set(ij_mask.indices()[0].tolist()) & set(ij_mask.indices()[1].tolist())
    del ij_mask
    new_w_t = new_w.t().coalesce()

    grad_ik = torch.sparse.FloatTensor(torch.empty([2, 0]).long(), torch.empty([0]), w.size()).to(device)
    grad_jk_indices = []
    grad_jk_values = []
    grad_ij_indices = []
    grad_ij_values = []
    N = w.size(0)
    for n, j in enumerate(nonzero_rows):
        # print(j)
        wij = new_w_t[j].unsqueeze(1).coalesce()
        wjk = new_w[j].unsqueeze(0).coalesce()
        if N > 20000:
            wijid = wij.indices()[0]
            wjkid = wjk.indices()[1]
            wijid = wijid.unsqueeze(1).repeat(1, wjkid.size(0))
            wjkid = wjkid.unsqueeze(0).repeat(wijid.size(0), 1)
            wijv = wij.values().unsqueeze(1)
            wjkv = wjk.values().unsqueeze(0)
            wijwjk = torch.sparse.FloatTensor(torch.stack([wijid, wjkid],dim=0).flatten(start_dim=1), wijv.mm(wjkv).flatten(), w.size())
        else:
            wijwjk = torch.sparse.mm(wij, wjk).coalesce()
        ik_to_optimize = (wijwjk - w).coalesce()
        ik_mask = ik_to_optimize.indices()
        ik_mask = ik_mask[:,(ik_to_optimize.values()>1e-10)*(ik_mask[0]!=ik_mask[1])]
        del ik_to_optimize
        ik_mask = torch.sparse.FloatTensor(ik_mask, torch.ones(ik_mask.size(1)).to(device), w.size()).coalesce()
        wik = ik_mask.cpu() * w_cpu
        new_w_processed = (wik.to(device) + 1e-10 * ik_mask).coalesce()
        if new_w_processed._nnz() == 0:
            continue
        wijwjk_processed = (wijwjk * ik_mask).coalesce()
        del wijwjk
        new_w_processed2 = torch.sparse.FloatTensor(new_w_processed.indices(), new_w_processed.values().reciprocal(), grad_ik.size()).coalesce()
        try:
            new_w_processed2.values().clip_(max=5)
        except Exception as e:
            pass
        grad_ik -= (new_w_processed2 * wijwjk_processed).coalesce()

        new_w_processed2 = torch.sparse.FloatTensor(new_w_processed.indices(), new_w_processed.values().log(), grad_ik.size()).coalesce()
        loss -= (wijwjk_processed * new_w_processed2).coalesce().values().sum()
        try:
            new_w_processed2.values().clip_(max=5)
        except Exception as e:
            pass
        sum_on_i = torch.sparse.mm(wij.t().coalesce(), new_w_processed2).coalesce()
        sum_indices = sum_on_i.indices()
        sum_indices[0] = j
        grad_jk_indices.append(sum_indices)
        grad_jk_values.append(sum_on_i.values().reciprocal())

        sum_on_k = torch.sparse.mm(new_w_processed2, wjk.t().coalesce()).coalesce()
        sum_indices = sum_on_k.indices()
        sum_indices[1] = j
        grad_ij_indices.append(sum_indices)
        grad_ij_values.append(sum_on_k.values().reciprocal())
        del new_w_processed2

    if len(grad_jk_indices) > 0:
        grad_ik -= torch.sparse.FloatTensor(torch.cat(grad_jk_indices, dim=1), torch.cat(grad_jk_values, dim=0), grad_ik.size())
    if len(grad_ij_indices) > 0:
        grad_ik -= torch.sparse.FloatTensor(torch.cat(grad_ij_indices, dim=1), torch.cat(grad_ij_values, dim=0), grad_ik.size())
    grad_ik = grad_ik.coalesce()
    w_mask = grad_ik.indices()[0]!=grad_ik.indices()[1]
    grad_ik = torch.sparse.FloatTensor(grad_ik.indices()[:,w_mask], grad_ik.values()[w_mask], grad_ik.size()).coalesce()

    try:
        grad_ik.values().clip_(min=-5)
    except Exception as e:
        # print(grad)
        # raise e
        pass

    return loss, (t, (grad_ik*gf.lambda_trans).coalesce())

def job_writeToFile(z):
    pred2idx, rawtype, allPreds, indices, values, size, threshold, gpath = z
    thisMat = torch.sparse.FloatTensor(indices, values, size)
    with open(gpath, 'w') as op:
        N = len(pred2idx)
        op.write(rawtype + "  type propagation num preds: " + str(N)+"\n")

        for pred in allPreds:
            scores = []

            predIdx = pred2idx[pred]
            thisRow = thisMat[predIdx].coalesce()
            neighs = thisRow.indices()[0]
            neighVals = thisRow.values()

            op.write("predicate: " + pred + "\n")
            op.write("max num neighbors: " + str(len(neighVals)) + "\n")
            op.write("\n")

            # print neighVals
            for i,neigh in enumerate(neighs):
                pred2 = allPreds[neigh]
                w = float(neighVals[i])
                if w < threshold:
                    continue
                scores.append((pred2,w))

            scores = sorted(scores,key = lambda x: x[1],reverse=True)

            op.write("global sims\n")
            s = ""
            for pred2,w in scores:
                s += pred2 + " " + str(w)+"\n"
            op.write(s+"\n")

class global_factory:

    def __init__(self, engG_dir_addr, lambda1, lambda2, epsilon1, args):
        files = os.listdir(engG_dir_addr)
        files = list(np.sort(files))
        num_f = 0
        self.typeToLocalGraphs = {}
        self.crossGraphPreds = {}
        self.crossGraphPredsTyped = {}
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.epsilon1 = epsilon1
        self.epsilon_trans = args.epsilon_trans
        self.lambda_trans = args.lambda_trans
        self.args = args
        print(args)

        start_time = time.time()

        dump_path = engG_dir_addr+"new_typeToLocalGraphs/"
        if not os.path.exists(dump_path):
            os.mkdir(dump_path)
        ## MULTITHREAD
        fs = [(engG_dir_addr+f, dump_path+f[:-4], args) for f in files if f[-4:] == '.txt']
        fs.reverse()
        # fs = [(engG_dir_addr+'location#thing_sim.txt', dump_path+'location#thing_sim', args)]
        # pool = Pool(40) 
        # results = pool.map(job_graphloading, fs)
        # pool.close() 
        # pool.join()
        results = [job_graphloading(z) for z in fs]
        # map(lambda a,b,c,d: a.composing(b,c,d), results)
        for a,b,c,d in results:
            a.composing(b,c,d)
        self.typeToLocalGraphs = {gr.rawtype:gr for gr,_,_,_ in results}
        del results
        for gr_type, gr in self.typeToLocalGraphs.items():
        ## MULTITHREAD
            for pname in gr.pred2idx:
                p = pname.strip().split("#")
                p_type = "#".join(p[1:])
                p = p[0]
                if p not in self.crossGraphPreds:
                    self.crossGraphPreds[p] = []
                self.crossGraphPreds[p].append(gr_type)
                if p not in self.crossGraphPredsTyped:
                    self.crossGraphPredsTyped[p] = set()
                self.crossGraphPredsTyped[p].add(p_type)

        print ('factory occupied memory: %.4f GB' % (psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024 / 1024), 'total time:', time.time()-start_time)

        self.crossGraphPreds = {p:t for p,t in self.crossGraphPreds.items() if len(t) > 1}
        self.crossGraphPredsTyped = {p:t for p,t in self.crossGraphPredsTyped.items() if p in self.crossGraphPreds}
        print("len of crossGraphPreds:", len(self.crossGraphPreds))
        print ('factory occupied memory: %.4f GB' % (psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024 / 1024), 'total time:', time.time()-start_time)

        self.crossTypeToPred = {}
        self.crossUnorderToPred = {}
        for p,ts in self.crossGraphPredsTyped.items():
            for t in ts:
                if t not in self.crossTypeToPred:
                    self.crossTypeToPred[t] = set()
                self.crossTypeToPred[t].add(p)
            for t in self.crossGraphPreds[p]:
                if t not in self.crossUnorderToPred:
                    self.crossUnorderToPred[t] = set()
                self.crossUnorderToPred[t].add(p)
        print("len of crossGraphPredsTyped:", len(self.crossGraphPredsTyped))
        print ('factory occupied memory: %.4f GB' % (psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024 / 1024), 'total time:', time.time()-start_time)

        self.typeToGlobalGraphs = {}
        listkeys = list(self.typeToLocalGraphs.keys())
        if args.gload_path == 'NONE':
            self.start_epoch = 0
            for k in listkeys:
                self.typeToGlobalGraphs[k] = self.typeToLocalGraphs[k]
                self.typeToLocalGraphs[k] = graph.dummyGraph(self.typeToGlobalGraphs[k].w_sparse)
        else:
            # assert self.args.gload_path == self.args.write_addr

            global_addr = "../gfiles/" + self.args.gload_path +"/"
            epochs = os.listdir(global_addr)
            epochs = [e for e in epochs if os.path.exists(global_addr+e+'/written_work#written_work_gsim.txt') and os.path.exists(global_addr+e+'/art#art_gsim.txt') and e[:5] == 'epoch']
            epochs = max(epochs, key=lambda x:int(x[5:]))
            self.start_epoch = int(epochs[5:]) + 1

            orig_featidx = args.featIdx
            args.featIdx = 0
            files = list(np.sort(os.listdir(global_addr+epochs)))
            fs = [(global_addr+epochs+'/'+f, 'NONE', args) for f in files if f[-4:] == '.txt']
            fs.reverse()
            results = [job_graphloading(z) for z in fs]
            for a,b,c,d in results:
                a.composing(b,c,d)
                self.typeToGlobalGraphs[a.rawtype] = a
                self.typeToLocalGraphs[a.rawtype] = graph.dummyGraph(self.typeToLocalGraphs[a.rawtype].w_sparse)
            for t in self.typeToLocalGraphs.keys():
                assert t in self.typeToGlobalGraphs
            args.featIdx = orig_featidx

        print("global graph copied")
        print ('factory occupied memory: %.4f GB' % (psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024 / 1024), 'total time:', time.time()-start_time)

        self.beta = nn.ParameterDict({t1+'#'+t2: nn.Parameter(
            torch.sparse_coo_tensor(torch.empty([2, 0]), torch.empty([0]), [len(self.typeToGlobalGraphs[t1].pred2idx), len(self.typeToGlobalGraphs[t2].pred2idx)]).coalesce()
        ) for t1 in self.typeToGlobalGraphs.keys() for t2 in self.typeToGlobalGraphs.keys()})
        self.beta.to(device)
        
        # HOW TO USE GRAD IN SPARSE CASE?
        for k in self.typeToGlobalGraphs.keys():
            self.typeToGlobalGraphs[k].w_sparse = nn.Parameter(self.typeToGlobalGraphs[k].w_sparse.coalesce())

        print("parameters built")
        print ('factory occupied memory: %.4f GB' % (psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024 / 1024), 'total time:', time.time()-start_time)
        self.sgd = optim.SGD(chain(self.beta.parameters(), (gr.w_sparse for _,gr in self.typeToGlobalGraphs.items())), lr = self.args.lr)

    def train_one_epoch(self, e):
        start_time = time.time()
        self.sgd.zero_grad()
        loss = 0
        tll, trl, tbl, tcl, twl, ttl = 0, 0, 0, 0, 0, 0
        for k in self.typeToGlobalGraphs.keys():
            ll, rl, bl, cl, wl, tl = self.backward(k)
            self.sgd.step()
            self.sgd.zero_grad()
            tll += float(ll)
            trl += float(rl)
            tbl += float(bl)
            tcl += float(cl)
            twl += float(wl)
            ttl += float(tl)
            print('trained', k, 'time', time.time()-start_time)
        
        print('local loss:', tll)
        print('cross loss:', tcl)
        print('resolution loss:', trl)
        print('beta normalization', tbl)
        print('w normalization', twl)
        print('transitivity normalization', ttl)
        print('current total loss:', tll+tcl+trl+tbl+ttl)
        with torch.no_grad():
            for k in self.beta.keys():
                # print(k, 'value', self.beta[k], 'grad', self.beta[k].grad)
                self.beta[k].data = self.beta[k].coalesce()
                self.beta[k].data.values().clip_(min=0,max=1)
                self.beta[k].data = self.beta[k].coalesce()
                assert self.beta[k].is_coalesced()
                # print('after clip', self.beta[k])
            for _,gr in self.typeToGlobalGraphs.items():
                # print(gr.rawtype, 'value', gr.w_sparse, 'grad', gr.w_sparse.grad)
                N = gr.w_sparse.size(0)
                theone = torch.sparse.FloatTensor(torch.arange(N).repeat(2,1), torch.ones(N)).to(device)
                gr.w_sparse.data = (gr.w_sparse+theone).coalesce()
                gr.w_sparse.data.values().clip_(min=0,max=1)
                gr.w_sparse.data = gr.w_sparse.coalesce()
                assert gr.w_sparse.is_coalesced()
                # print('after clip', gr.w_sparse)
        print("total time:", time.time()-start_time)

    def backward(self, k):
        start_time = time.time()
        localloss = self.local_backward(k)
        crossloss = self.cross_backward(k) if self.args.use_cross > 0 and k!='thing#thing' else 0
        resoloss = self.resolution_backward(k) if self.args.use_reso > 0 else 0
        transloss = self.transitivity_backward(k) if self.lambda_trans > 0 else 0
        betaloss = self.betanorm_backward(k) if self.lambda2 > 0 else 0
        wloss = self.w_norm_backward(k) if self.lambda1 > 0 else 0
        return localloss, 0.5 * resoloss, 0.5 * self.lambda2 * betaloss, 0.5 * crossloss, self.lambda1 * wloss, self.lambda_trans * transloss

    def local_backward(self, k):
        local_g = self.typeToLocalGraphs[k].w_sparse
        global_g = self.typeToGlobalGraphs[k].w_sparse
        loss = torch.sparse.sum((local_g-global_g)**2)
        self.typeToGlobalGraphs[k].w_sparse.grad = 2*(global_g-local_g)
        return loss

    def cross_backward(self, k):
        keys = [(k,t2) for t2 in self.typeToGlobalGraphs.keys() if k != t2]
        loss = 0
        # for l, (t1,s1), (t2,s2), (t3,s3) in results:
        for i, key in enumerate(keys):
            if key[1] == 'thing#thing':
                continue
            l, (t1,s1), (t2,s2), (t3,s3) = job_crossloss(key[0],key[1],self)
            self.typeToGlobalGraphs[t1].w_sparse.grad += s1
            if self.typeToGlobalGraphs[t2].w_sparse.grad is None:
                self.typeToGlobalGraphs[t2].w_sparse.grad = s2
            else:
                self.typeToGlobalGraphs[t2].w_sparse.grad += s2
            if self.beta[t3].grad is None:
                self.beta[t3].grad = s3
            else:
                self.beta[t3].grad += s3
            loss += l
        print('crossed', k)
        return loss

    def resolution_backward(self, k):
        loss, (t,s) = job_resoloss(k, self)
        self.typeToGlobalGraphs[t].w_sparse.grad += s
        return loss

    def transitivity_backward(self, k):
        # loss, (t,s) = job_transloss(k, self)
        # loss, (t,s) = job_transloss2(k, self)
        job_trans_list = [job_transloss, job_transloss2, job_transloss3, job_transloss4]
        foo = job_trans_list[self.args.trans_method]
        loss, (t,s) = foo(k, self)
        print('transitivity', k)
        self.typeToGlobalGraphs[t].w_sparse.grad += s
        return loss

    def betanorm_backward(self, k):
        loss = 0
        for key, v in self.beta.items():
            if key.split('#')[0] != k:
                continue
            loss += torch.sparse.sum(v**2-2*v)
            if v.grad is None:
                v.grad = 0.5 * self.lambda2 * 2*v
            else:
                v.grad += 0.5 * self.lambda2 * 2*v
            grad = v.grad.coalesce()
            v.grad = torch.sparse.FloatTensor(grad.indices(), grad.values()-self.lambda2, grad.size())
        return loss
    
    def w_norm_backward(self, k):
        gr = self.typeToGlobalGraphs[k]
        w = gr.w_sparse.coalesce()
        loss = w.values().abs().sum()
        grad = torch.sparse.FloatTensor(w.indices(), w.values().sign(), w.size())
        if gr.w_sparse.grad is None:
            gr.w_sparse.grad = self.lambda1 * grad
        else:
            gr.w_sparse.grad += self.lambda1 * grad
        return loss

if __name__ == "__main__":

    if args and args.featIdx is not None:
        graph.Graph.featIdx = args.featIdx
    if args and args.threshold:
        graph.Graph.threshold = args.threshold

    engG_dir_addr = "../gfiles/" + args.gpath +"/"
    write_addr = "../gfiles/" + args.writepath + "/"
    if not os.path.exists(write_addr):
        os.mkdir(write_addr)

    print("start building factory")
    model = global_factory(engG_dir_addr, args.lambda1, args.lambda2, args.epsilon1, args)

    for e in range(model.start_epoch, args.n_epoch):
        print('epoch:', e)
        start_time = time.time()
        model.train_one_epoch(e)
        epoch_addr = write_addr+'epoch'+str(e)+'/'
        if not os.path.exists(epoch_addr):
            os.mkdir(epoch_addr)
        for k, gr in model.typeToGlobalGraphs.items():
            s = gr.rawtype
            s = epoch_addr+s+"_gsim.txt"
            print('writing to', s)
            gr.writeGraphToFile(s)
        print("time:", time.time()-start_time)
