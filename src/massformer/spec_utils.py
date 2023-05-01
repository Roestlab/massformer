import numpy as np
import torch as th
import torch.nn.functional as F
import torch_scatter as th_s

from massformer.misc_utils import EPS


def bin_func(mzs, ints, mz_max, mz_bin_res, ints_thresh, return_index):

    mzs = np.array(mzs, dtype=np.float32)
    bins = np.arange(
        mz_bin_res,
        mz_max +
        mz_bin_res,
        step=mz_bin_res).astype(
        np.float32)
    bin_idx = np.searchsorted(bins, mzs, side="right")
    if return_index:
        return bin_idx.tolist()
    else:
        ints = np.array(ints, dtype=np.float32)
        bin_spec = np.zeros([len(bins)], dtype=np.float32)
        for i in range(len(mzs)):
            if bin_idx[i] < len(bin_spec) and ints[i] >= ints_thresh:
                bin_spec[bin_idx[i]] = max(bin_spec[bin_idx[i]], ints[i])
        if np.all(bin_spec == 0.):
            print("> warning: bin_spec is all zeros!")
            bin_spec[-1] = 1.
        return bin_spec


def unprocess_spec(spec, transform):

    # transform signal
    if transform == "log10":
        max_ints = float(np.log10(1000. + 1.))
        def untransform_fn(x): return 10**x - 1.
    elif transform == "log10over3":
        max_ints = float(np.log10(1000. + 1.) / 3.)
        def untransform_fn(x): return 10**(3 * x) - 1.
    elif transform == "loge":
        max_ints = float(np.log(1000. + 1.))
        def untransform_fn(x): return th.exp(x) - 1.
    elif transform == "sqrt":
        max_ints = float(np.sqrt(1000.))
        def untransform_fn(x): return x**2
    elif transform == "linear":
        raise NotImplementedError
    elif transform == "none":
        max_ints = 1000.
        def untransform_fn(x): return x
    else:
        raise ValueError("invalid transform")
    spec = spec / (th.max(spec, dim=-1, keepdim=True)[0] + EPS) * max_ints
    spec = untransform_fn(spec)
    spec = th.clamp(spec, min=0.)
    assert not th.isnan(spec).any()
    return spec


def process_spec(spec, transform, normalization):

    # scale spectrum so that max value is 1000
    spec = spec / (th.max(spec, dim=-1, keepdim=True)[0] + EPS) * 1000.
    # transform signal
    if transform == "log10":
        spec = th.log10(spec + 1)
    elif transform == "log10over3":
        spec = th.log10(spec + 1) / 3
    elif transform == "loge":
        spec = th.log(spec + 1)
    elif transform == "sqrt":
        spec = th.sqrt(spec)
    elif transform == "linear":
        raise NotImplementedError
    elif transform == "none":
        pass
    else:
        raise ValueError("invalid transform")
    # normalize
    if normalization == "l1":
        spec = F.normalize(spec, p=1, dim=-1, eps=EPS)
    elif normalization == "l2":
        spec = F.normalize(spec, p=2, dim=-1, eps=EPS)
    elif normalization == "none":
        pass
    else:
        raise ValueError("invalid normalization")
    assert not th.isnan(spec).any()
    return spec


def merge_spec(spec, group_id, transform, normalization, *other_ids):

    un_group_id, un_group_idx = th.unique(group_id, dim=0, return_inverse=True)
    spec_u = unprocess_spec(spec, transform)
    spec_merge_u = th_s.scatter_mean(
        spec_u, un_group_idx, dim=0, dim_size=un_group_id.shape[0])
    spec_merge = process_spec(spec_merge_u, transform, normalization)
    other_ids_merge = []
    for other_id in other_ids:
        other_id_merge = th_s.scatter_mean(
            other_id,
            un_group_idx,
            dim=0,
            dim_size=un_group_id.shape[0]).type(
            other_id.dtype)
        other_ids_merge.append(other_id_merge)
    return (spec_merge, un_group_id) + tuple(other_ids_merge)
