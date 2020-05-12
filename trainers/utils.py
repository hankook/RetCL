import dgl
import faiss
import torch
import random
from datasets import MoleculeDict, Reaction, build_dataloader


def convert_tensor(inputs, device=None, detach=True):
    if isinstance(inputs, list):
        return list(convert_tensor(x, device=device, detach=detach) for x in inputs)
    elif isinstance(inputs, tuple):
        return tuple(convert_tensor(x, device=device, detach=detach) for x in inputs)
    elif isinstance(inputs, torch.Tensor):
        if device is not None:
            inputs = inputs.to(device)
        if detach:
            inputs = inputs.detach()
        return inputs
    else:
        return inputs


def collect_embeddings(module, mol_dict, batch_size=512, cpu=True, device=None):
    module.eval()
    dataloader = build_dataloader(mol_dict, batch_size=batch_size)
    out_device = torch.device('cpu') if cpu else device
    with torch.no_grad():
        embeddings = []
        for i, molecules in enumerate(dataloader):
            es = module(prepare_molecules(molecules, device=device))
            es = convert_tensor(es, device=out_device)
            embeddings.append(es)

        if not isinstance(embeddings[0], torch.Tensor):
            embeddings = [torch.cat(e, 0) for e in zip(*embeddings)]
        else:
            embeddings = torch.cat(embeddings, 0)

    return embeddings


def prepare_reactions(reactions, nearest_neighbors=None, additional_molecules=None, device=None):
    molecule_dict = MoleculeDict()
    for r in reactions:
        for m in [r.product] + r.reactants:
            molecule_dict.add(m)

    if additional_molecules is not None:
        for m in additional_molecules:
            molecule_dict.add(m)

    if nearest_neighbors is not None:
        for i in range(len(molecule_dict)):
            for mol in nearest_neighbors[molecule_dict[i].smiles]:
                molecule_dict.add(mol)

    graphs = dgl.batch([m.graph for m in molecule_dict])
    if device is not None:
        graphs = graphs.to(device)
    converted_reactions = []
    for r in reactions:
        converted_reactions.append(Reaction(
            product=molecule_dict.index(r.product),
            reactants=[molecule_dict.index(x) for x in r.reactants],
            label=r.label))

    return converted_reactions, graphs


def prepare_molecules(molecules, device=None):
    graphs = dgl.batch([m.graph for m in molecules])
    if device is not None:
        graphs = graphs.to(device)
    return graphs


def swig_ptr_from_float_tensor(x):
    assert x.is_contiguous()
    assert x.dtype == torch.float32
    return faiss.cast_integer_to_float_ptr(
        x.storage().data_ptr() + x.storage_offset() * 4)

def swig_ptr_from_long_tensor(x):
    assert x.is_contiguous()
    assert x.dtype == torch.int64, 'dtype=%s' % x.dtype
    return faiss.cast_integer_to_long_ptr(
        x.storage().data_ptr() + x.storage_offset() * 8)


def knn_search(res, xb, xq, k, D=None, I=None,
               metric=faiss.METRIC_INNER_PRODUCT):
    assert xb.device == xq.device

    nq, d = xq.size()
    if xq.is_contiguous():
        xq_row_major = True
    elif xq.t().is_contiguous():
        xq = xq.t()    # I initially wrote xq:t(), Lua is still haunting me :-)
        xq_row_major = False
    else:
        raise TypeError('matrix should be row or column-major')

    xq_ptr = swig_ptr_from_float_tensor(xq)

    nb, d2 = xb.size()
    assert d2 == d
    if xb.is_contiguous():
        xb_row_major = True
    elif xb.t().is_contiguous():
        xb = xb.t()
        xb_row_major = False
    else:
        raise TypeError('matrix should be row or column-major')
    xb_ptr = swig_ptr_from_float_tensor(xb)

    if D is None:
        D = torch.empty(nq, k, device=xb.device, dtype=torch.float32)
    else:
        assert D.shape == (nq, k)
        assert D.device == xb.device

    if I is None:
        I = torch.empty(nq, k, device=xb.device, dtype=torch.int64)
    else:
        assert I.shape == (nq, k)
        assert I.device == xb.device

    D_ptr = swig_ptr_from_float_tensor(D)
    I_ptr = swig_ptr_from_long_tensor(I)

    faiss.bruteForceKnn(res, metric,
                        xb_ptr, xb_row_major, nb,
                        xq_ptr, xq_row_major, nq,
                        d, k, D_ptr, I_ptr)

    return D, I

