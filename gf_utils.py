import torch

def sparserow_leftmm(a, b): # the non-zero rows of a should be less (sparse), a and b should be square sparse matrix
    w_indices = a.indices()
    avail_row = set(w_indices[0].tolist())
    ids = []
    vs = []
    # print(len(avail_row))
    if a._nnz() == 0:
        return torch.sparse.FloatTensor(torch.empty([2, 0]).long(), torch.empty([0]), a.size()).to(a.device)
    for r in avail_row:
        res = torch.sparse.mm(a[r].unsqueeze(0), b).coalesce()
        new_ind = res.indices()
        new_ind[0] = r
        ids.append(new_ind)
        vs.append(res.values())
    output = torch.sparse.FloatTensor(torch.cat(ids, dim=1), torch.cat(vs, dim=0), b.size())
    # print(output)
    return output

def sparserow_rightmm(a, b): # the non-zero columns of b should be less (sparse), a and b should be square sparse matrix
    output = sparserow_leftmm(b.t().coalesce(), a.t().coalesce()).t()
    return output

def type_reverse(pred):
    t = pred.split('#')
    if len(t) == 3:
        return "#".join([t[0], t[2], t[1]])
    return t[1]+"#"+t[0]

def my_index_select(d, index, dim):
    if index.size(0) == 0:
        return d.index_select(index=index,dim=dim)
    if len(d.size()) == 1:
        r = torch.stack([d[i] for i in index], dim=0)
        # print(r)
        return r
    if dim == 0:
        return torch.stack([d[i] for i in index], dim=0)
    elif dim == 1:
        return my_index_select(d.t(), index, 0).t()
    raise NotImplemented("dim should be 0 or 1")