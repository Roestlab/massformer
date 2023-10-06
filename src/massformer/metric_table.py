import torch as th
import torch_scatter as th_s
import numpy as np
import wandb
import os

from massformer.plot_utils import plot_sim_hist


class MetricTable:

    def __init__(
            self,
            keys,
            vals,
            mol_id,
            group_id,
            prefix,
            loss=False,
            merged=False):

        self.loss = loss
        self.merged = merged
        self.val_d = {}
        for key, val in zip(keys, vals):
            self.val_d[key] = val
        self.mol_id = mol_id
        self.group_id = group_id
        self.prefix = prefix
        # these store aggregated metrics and histogram images
        self.agg_cache = {}
        self.hist_cache = {}

    def clear_cache(self):
        self.agg_cache = {}
        self.hist_cache = {}

    def get_val_mol_id(self, key, groupby_mol=False):
        if groupby_mol:
            # averages sims by mol
            un_mol_id, un_mol_idx = th.unique(
                self.mol_id, dim=0, return_inverse=True)
            mol_val = th_s.scatter_mean(
                self.val_d[key],
                un_mol_idx,
                dim=0,
                dim_size=un_mol_id.shape[0])
            return mol_val, un_mol_id
        else:
            # averages sims by spec
            return self.val_d[key], self.mol_id

    def aggregate(self, key, how="mean", groupby_mol=False):

        agg_str = self.get_val_str(key, groupby_mol) + f"_{how}"
        if agg_str in self.agg_cache:
            return self.agg_cache[agg_str]
        val, mol_id = self.get_val_mol_id(key, groupby_mol=groupby_mol)
        if how == "mean":
            agg_val = th.mean(val, dim=0).float()
        else:
            raise ValueError
        # update cache
        self.agg_cache[agg_str] = agg_val
        return agg_str, agg_val

    def histogram(self, key, groupby_mol=False):

        hist_str = self.get_val_str(key, groupby_mol) + "_hist"
        if hist_str in self.hist_cache:
            return self.hist_cache[hist_str]
        val, mol_id = self.get_val_mol_id(key, groupby_mol=groupby_mol)
        hist = plot_sim_hist(val.numpy(), title=hist_str)
        hist_val = wandb.Image(hist)
        # update cache
        self.hist_cache[hist_str] = hist_val
        return hist_str, hist_val

    def get_val_str(self, key, groupby_mol):

        if groupby_mol:
            entry_str = "mol"
        else:
            entry_str = "spec"
        if self.merged:
            entry_str = "m_" + entry_str
        if self.loss:
            entry_str += "_loss"
        else:
            entry_str += "_sim"
        entry_str += f"_{key}"
        return entry_str

    def compute(
            self,
            compute_agg=False,
            compute_hist=False,
            groupby_mol=False):

        self.clear_cache()
        if groupby_mol:
            groupby_mols = (True, False)
        else:
            groupby_mols = (False,)
        for gm in groupby_mols:
            # sims
            for key in self.val_d.keys():
                # updates cache
                if compute_agg:
                    self.aggregate(key, groupby_mol=gm)
                if compute_hist and not self.loss:
                    self.histogram(key, groupby_mol=gm)

    def unload_cache(self, prefix=True, agg=True, hist=True):

        cache = {}
        if agg:
            cache = {**cache, **self.agg_cache}
        if hist:
            cache = {**cache, **self.hist_cache}
        if prefix:
            p_cache = {}
            for k, v in cache.items():
                p_cache[self.prefix + "_" + k] = v
            cache = p_cache
        return cache

    def export_val(
            self,
            key,
            prefix=False,
            groupby_mol=False,
            mol_id=False,
            group_id=False):

        val_str = self.get_val_str(key, groupby_mol)
        if prefix:
            val_str = f"{prefix}_{val_str}"
        val = self.val_d[key]
        export_d = {val_str: val}
        if mol_id:
            export_d["mol_id"] = self.mol_id
        if group_id:
            export_d["group_id"] = self.group_id
        return export_d

    def get_table_str(self):

        table_str = self.prefix
        if self.merged:
            table_str += "_m"
        if self.loss:
            table_str += "_loss"
        else:
            table_str += "_sim"
        return table_str

    def save(self, path):

        save_d = {}
        for k, v in self.__dict__.items():
            if not (k in ["agg_cache", "hist_cache"]):
                save_d[k] = v
        path = os.path.splitext(path)[0] + ".pt"
        th.save(save_d, path)

    @classmethod
    def load(cls, path):

        save_d = th.load(path)
        keys, vals = [], []
        for k, v in save_d["val_d"].items():
            keys.append(k)
            vals.append(v)
        table = MetricTable(
            keys,
            vals,
            save_d["mol_id"],
            save_d["group_id"],
            save_d["prefix"],
            loss=save_d["loss"],
            merged=save_d["merged"]
        )
        return table

    def __eq__(self, other):

        if not isinstance(other, MetricTable):
            return False
        self_dict = self.__dict__.items()
        other_dict = other.__dict__.items()
        for k, v in self_dict:
            if not (k in ["agg_cache", "hist_cache", "val_d"]):
                if isinstance(v, th.Tensor) and not th.all(other_dict[k] == v):
                    return False
                if not isinstance(v, th.Tensor) and other_dict[k] != v:
                    return False
        # val_d only contains tensors
        for k, v in self_dict["val_d"].items():
            if not (
                    k in other_dict["val_d"]) or not th.all(
                    v == other_dict["val_d"][k]):
                return False
        return True
