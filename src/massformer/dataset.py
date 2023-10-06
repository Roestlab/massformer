import torch as th
import torch.utils.data as th_data
import torch.nn.functional as F
import json
import pandas as pd
import numpy as np
import os
from pprint import pprint
import dgl
import dgllife.utils as chemutils
import torch_geometric.data
from tqdm import tqdm
import itertools
from sklearn.decomposition import LatentDirichletAllocation

from massformer.misc_utils import EPS, np_temp_seed, np_scatter_add, np_one_hot, flatten_lol, timeout, none_or_nan
import massformer.data_utils as data_utils
from massformer.data_utils import H_MASS, O_MASS, N_MASS, NA_MASS
import massformer.gf_data_utils as gf_data_utils
import massformer.esp_data_utils as esp_data_utils
import massformer.spec_utils as spec_utils


def data_to_device(data_d, device, non_blocking):

    new_data_d = {}
    for k, v in data_d.items():
        if isinstance(v, th.Tensor) or isinstance(v, dgl.DGLGraph) or isinstance(v, torch_geometric.data.Data):
            new_data_d[k] = v.to(device, non_blocking=non_blocking)
        elif isinstance(v, dict):
            new_v = {}
            for kk, vv in v.items():
                new_v[kk] = vv.to(device, non_blocking=non_blocking)
            new_data_d[k] = new_v
        else:
            new_data_d[k] = v
    return new_data_d


class TrainSubset(th_data.Subset):

    def __getitem__(self, idx):
        return self.dataset.__getitem__(self.indices[idx])


class BaseDataset(th_data.Dataset):

    def __init__(self, *dset_types, **kwargs):

        self.is_fp_dset = "fp" in dset_types
        self.is_graph_dset = "graph" in dset_types
        self.is_gf_v2_dset = "gf_v2" in dset_types
        self.is_esp_dset = "esp" in dset_types
        assert (self.is_fp_dset or self.is_graph_dset or self.is_gf_v2_dset or self.is_esp_dset)
        for k, v in kwargs.items():
            setattr(self, k, v)
        assert os.path.isdir(self.proc_dp), self.proc_dp
        self.spec_df = pd.read_pickle(
            os.path.join(self.proc_dp, "spec_df.pkl"))
        self.mol_df = pd.read_pickle(os.path.join(self.proc_dp, "mol_df.pkl"))
        self._select_spec()
        self._setup_spec_metadata_dicts()
        # use mol_id as index for speedy access
        self.mol_df = self.mol_df.set_index(
            "mol_id", drop=False).sort_index().rename_axis(None)
        self.casmi_info = {"inchikey_s": set(), "scaffold": set()}

    def _select_spec(self):

        masks = []
        # dataset mask
        dset_mask = self.spec_df["dset"].isin(
            self.primary_dset + self.secondary_dset)
        masks.append(dset_mask)
        # instrument type
        inst_type_mask = self.spec_df["inst_type"].isin(self.inst_type)
        masks.append(inst_type_mask)
        # frag mode
        frag_mode_mask = self.spec_df["frag_mode"].isin(self.frag_mode)
        masks.append(frag_mode_mask)
        # ion mode
        ion_mode_mask = self.spec_df["ion_mode"] == self.ion_mode
        masks.append(ion_mode_mask)
        # precursor type
        if self.ion_mode == "P":
            prec_type_mask = self.spec_df["prec_type"].isin(self.pos_prec_type)
        elif self.ion_mode == "N":
            prec_type_mask = self.spec_df["prec_type"].isin(self.neg_prec_type)
        else:
            assert self.ion_mode == "EI"
            prec_type_mask = self.spec_df["prec_type"] == "EI"
        masks.append(prec_type_mask)
        # resolution
        if self.res != []:
            res_mask = self.spec_df["res"].isin(self.res)
            masks.append(res_mask)
        # collision energy
        ce_mask = ~(self.spec_df["ace"].isna() & self.spec_df["nce"].isna())
        masks.append(ce_mask)
        # spectrum type
        if self.ion_mode == "EI":
            spec_type_mask = self.spec_df["spec_type"] == "EI"
        else:
            spec_type_mask = self.spec_df["spec_type"] == "MS2"
        masks.append(spec_type_mask)
        # maximum mz allowed
        mz_mask = self.spec_df["peaks"].apply(
            lambda peaks: max(peak[0] for peak in peaks) < self.mz_max)
        masks.append(mz_mask)
        # precursor mz
        prec_mz_mask = ~self.spec_df["prec_mz"].isna()
        masks.append(prec_mz_mask)
        # single molecule
        multi_mol_ids = self.mol_df[self.mol_df["smiles"].str.contains(
            "\\.")]["mol_id"]
        single_mol_mask = ~self.spec_df["mol_id"].isin(multi_mol_ids)
        masks.append(single_mol_mask)
        # neutral molecule
        charges = self.mol_df["mol"].apply(data_utils.mol_to_charge)
        charged_ids = self.mol_df[charges != 0]["mol_id"]
        neutral_mask = ~self.spec_df["mol_id"].isin(charged_ids)
        # print(neutral_mask.sum())
        masks.append(neutral_mask)
        # put them together
        all_mask = masks[0]
        for mask in masks:
            all_mask = all_mask & mask
        if np.sum(all_mask) == 0:
            raise ValueError("select removed all items")
        self.spec_df = self.spec_df[all_mask].reset_index(drop=True)
        self._setup_ce()
        # get group_id
        n_before_group = self.spec_df.shape[0]
        group_df = self.spec_df.drop(
            columns=[
                "spec_id",
                "peaks",
                "nce",
                "ace",
                "res",
                "prec_mz",
                "ri",
                "col_gas"])
        assert not group_df.isna().any().any()
        group_df = group_df.drop_duplicates()
        group_df.loc[:, "group_id"] = np.arange(group_df.shape[0])
        self.spec_df = self.spec_df.merge(group_df, how="inner")
        del group_df
        n_after_group = self.spec_df.shape[0]
        assert n_before_group == n_after_group
        # compute stats
        for dset in self.primary_dset + self.secondary_dset:
            dset_spec_df = self.spec_df[self.spec_df["dset"] == dset]
            num_specs = dset_spec_df["spec_id"].nunique()
            num_mols = dset_spec_df["mol_id"].nunique()
            num_groups = dset_spec_df["group_id"].nunique()
            num_ces_per_group = dset_spec_df[["group_id", self.ce_key]].groupby(
                by="group_id").count()[self.ce_key].mean()
            print(f">>> {dset}")
            print(
                f"> num_spec = {num_specs}, num_mol = {num_mols}, num_group = {num_groups}, num_ce_per_group = {num_ces_per_group}")
        # subsample
        if self.subsample_size > 0:
            self.spec_df = self.spec_df.groupby("mol_id").sample(
                n=self.subsample_size, random_state=self.subsample_seed, replace=True)
            self.spec_df = self.spec_df.reset_index(drop=True)
        else:
            self.spec_df = self.spec_df
        # num_entries
        if self.num_entries > 0:
            self.spec_df = self.spec_df.sample(
                n=self.num_entries,
                random_state=self.subsample_seed,
                replace=False)
            self.spec_df = self.spec_df.reset_index(drop=True)
        # only keep mols with spectra
        self.mol_df = self.mol_df[self.mol_df["mol_id"].isin(
            self.spec_df["mol_id"])]
        self.mol_df = self.mol_df.reset_index(drop=True)

    def _setup_ce(self):

        if self.convert_ce:
            if self.ce_key == "ace":
                other_ce_key = "nce"
                ce_conversion_fn = data_utils.nce_to_ace
            else:
                other_ce_key = "ace"
                ce_conversion_fn = data_utils.ace_to_nce
            convert_mask = self.spec_df[self.ce_key].isna()
            assert not self.spec_df.loc[convert_mask,
                                        other_ce_key].isna().any()
            self.spec_df.loc[convert_mask, self.ce_key] = self.spec_df[convert_mask].apply(
                ce_conversion_fn, axis=1)
            assert not self.spec_df[self.ce_key].isna().any()
        else:
            self.spec_df = self.spec_df.dropna(
                axis=0, subset=[
                    self.ce_key]).reset_index(
                drop=True)

    def _setup_spec_metadata_dicts(self):

        # featurize spectral metadata
        # we can assume that the dataset is filtered (using the method above)
        # to only include these values
        inst_type_list = self.inst_type
        if self.ion_mode == "P":
            prec_type_list = self.pos_prec_type
        elif self.ion_mode == "N":
            prec_type_list = self.neg_prec_type
        else:
            assert self.ion_mode == "EI"
            prec_type_list = ["EI"]
        frag_mode_list = self.frag_mode
        self.inst_type_c2i = {
            string: i for i,
            string in enumerate(inst_type_list)}
        self.inst_type_i2c = {
            i: string for i,
            string in enumerate(inst_type_list)}
        self.prec_type_c2i = {
            string: i for i,
            string in enumerate(prec_type_list)}
        self.prec_type_i2c = {
            i: string for i,
            string in enumerate(prec_type_list)}
        self.frag_mode_c2i = {
            string: i for i,
            string in enumerate(frag_mode_list)}
        self.frag_mode_i2c = {
            i: string for i,
            string in enumerate(frag_mode_list)}
        self.num_inst_type = len(inst_type_list)
        self.num_prec_type = len(prec_type_list)
        self.num_frag_mode = len(frag_mode_list)
        self.max_ce = self.spec_df[self.ce_key].max()
        self.mean_ce = self.spec_df[self.ce_key].mean()
        self.std_ce = self.spec_df[self.ce_key].std()

    def __getitem__(self, idx):

        spec_entry = self.spec_df.iloc[idx]
        mol_id = spec_entry["mol_id"]
        # mol_entry = self.mol_df[self.mol_df["mol_id"] == mol_id].iloc[0]
        mol_entry = self.mol_df.loc[mol_id]
        data = self.process_entry(spec_entry, mol_entry["mol"])
        return data

    def __len__(self):

        return self.spec_df.shape[0]

    def bin_func(self, mzs, ints, return_index=False):

        assert self.ints_thresh == 0., self.ints_thresh
        return spec_utils.bin_func(
            mzs,
            ints,
            self.mz_max,
            self.mz_bin_res,
            self.ints_thresh,
            return_index)

    def transform_func(self, spec):

        if self.process_spec_old:
            spec = spec_utils.process_spec_old(
                spec,
                self.transform,
                self.spectrum_normalization,
                self.ints_thresh)
        else:
            spec = spec_utils.process_spec(
                th.as_tensor(spec),
                self.transform,
                self.spectrum_normalization)
            spec = spec.numpy()
        return spec

    def get_split_masks(
            self,
            val_frac,
            test_frac,
            sec_frac,
            split_key,
            split_seed,
            ignore_casmi):

        assert split_key in ["inchikey_s", "scaffold"], split_key
        assert len(self.secondary_dset) <= 1, self.secondary_dset
        # primary
        prim_mask = self.spec_df["dset"].isin(self.primary_dset)
        prim_mol_id = self.spec_df[prim_mask]["mol_id"].unique()
        prim_key = set(
            self.mol_df[self.mol_df["mol_id"].isin(prim_mol_id)][split_key])
        # secondary
        sec_mask = self.spec_df["dset"].isin(self.secondary_dset)
        sec_mol_id = self.spec_df[sec_mask]["mol_id"].unique()
        sec_key = set(
            self.mol_df[self.mol_df["mol_id"].isin(sec_mol_id)][split_key])
        sec_key_list = sorted(list(sec_key))
        # print(sec_key_list[:5])
        # sample secondary keys
        with np_temp_seed(split_seed):
            sec_num = round(len(sec_key_list) * sec_frac)
            sec_key_list = np.random.choice(
                sec_key_list, size=sec_num, replace=False).tolist()
            sec_key = set(sec_key_list)
            # print(sec_key_list[:5])
            sec_mol_id = self.mol_df[self.mol_df[split_key].isin(
                sec_key_list) & self.mol_df["mol_id"].isin(sec_mol_id)]["mol_id"].unique()
            sec_mask = self.spec_df["mol_id"].isin(sec_mol_id) & sec_mask
            # print(split_seed,sec_num,sec_mask.sum())
        # get keys (secondary might same compounds as primary does!)
        prim_only_key = prim_key - sec_key
        sec_only_key = sec_key
        prim_only_key_list = sorted(list(prim_only_key))
        both_key = prim_key & sec_key
        # compute number for each split
        test_num = round(len(prim_only_key_list) * test_frac)
        val_num = round(len(prim_only_key_list) * val_frac)
        # make sure that test set gets all of the casmi keys!
        if not ignore_casmi:
            prim_only_and_casmi = prim_only_key & self.casmi_info[split_key]
        else:
            prim_only_and_casmi = set()
        if test_num > 0:
            assert len(prim_only_and_casmi) <= test_num
            test_num -= len(prim_only_and_casmi)
        prim_only_no_casmi_key_list = [
            k for k in prim_only_key_list if not (
                k in prim_only_and_casmi)]
        assert len(set(prim_only_no_casmi_key_list) & prim_only_and_casmi) == 0
        # do the split
        with np_temp_seed(split_seed):
            prim_only_test_num = max(test_num - len(both_key), 0)
            test_key = set(
                np.random.choice(
                    prim_only_no_casmi_key_list,
                    size=prim_only_test_num,
                    replace=False))
            test_key = test_key.union(prim_only_and_casmi).union(both_key)
            train_val_key = prim_only_key - test_key
            val_key = set(
                np.random.choice(
                    sorted(
                        list(train_val_key)),
                    size=val_num,
                    replace=False))
            train_key = train_val_key - val_key
            assert len(train_key & sec_only_key) == 0
            assert len(val_key & sec_only_key) == 0
            # assert len(test_key & sec_only_key) == 0
            assert len(train_key & prim_only_and_casmi) == 0
            assert len(val_key & prim_only_and_casmi) == 0
        # get ids and create masks
        train_mol_id = self.mol_df["mol_id"][self.mol_df[split_key].isin(
            list(train_key))].unique()
        val_mol_id = self.mol_df["mol_id"][self.mol_df[split_key].isin(
            list(val_key))].unique()
        test_mol_id = self.mol_df["mol_id"][self.mol_df[split_key].isin(
            list(test_key))].unique()
        train_mask = self.spec_df["mol_id"].isin(train_mol_id)
        val_mask = self.spec_df["mol_id"].isin(val_mol_id)
        test_mask = self.spec_df["mol_id"].isin(test_mol_id)
        prim_mask = train_mask | val_mask | test_mask
        prim_mol_id = pd.Series(
            list(set(train_mol_id) | set(val_mol_id) | set(test_mol_id)))
        # note: primary can include secondary molecules in the test split
        sec_masks = [(self.spec_df["dset"] == dset) & (
            self.spec_df["mol_id"].isin(sec_mol_id)) for dset in self.secondary_dset]
        assert (train_mask & val_mask & test_mask).sum() == 0
        print("> primary")
        print("splits: train, val, test, total")
        print(
            f"spec: {train_mask.sum()}, {val_mask.sum()}, {test_mask.sum()}, {prim_mask.sum()}")
        print(
            f"mol: {len(train_mol_id)}, {len(val_mol_id)}, {len(test_mol_id)}, {len(prim_mol_id)}")
        if len(self.secondary_dset) > 0:
            print("> secondary")
        for sec_idx, sec_dset in enumerate(self.secondary_dset):
            cur_sec = self.spec_df[sec_masks[sec_idx]]
            cur_sec_mol_id = cur_sec["mol_id"]
            cur_both_mol_mask = self.spec_df["mol_id"].isin(
                set(prim_mol_id) & set(cur_sec_mol_id))
            cur_prim_both = self.spec_df[prim_mask & cur_both_mol_mask]
            cur_sec_both = self.spec_df[sec_masks[sec_idx] & cur_both_mol_mask]
            print(
                f"{sec_dset} spec = {cur_sec.shape[0]}, mol = {cur_sec_mol_id.nunique()}")
            print(
                f"{sec_dset} overlap: prim spec = {cur_prim_both.shape[0]}, sec spec = {cur_sec_both.shape[0]}, mol = {cur_prim_both['mol_id'].nunique()}")
        return train_mask, val_mask, test_mask, sec_masks

    def get_spec_feats(self, spec_entry):

        # convert to a dense vector
        mol_id = th.tensor(spec_entry["mol_id"]).unsqueeze(0)
        spec_id = th.tensor(spec_entry["spec_id"]).unsqueeze(0)
        group_id = th.tensor(spec_entry["group_id"]).unsqueeze(0)
        mzs = [peak[0] for peak in spec_entry["peaks"]]
        ints = [peak[1] for peak in spec_entry["peaks"]]
        prec_mz = spec_entry["prec_mz"]
        prec_mz_bin = self.bin_func([prec_mz], None, return_index=True)[0]
        prec_diff = max(mz - prec_mz for mz in mzs)
        num_peaks = len(mzs)
        bin_spec = self.transform_func(self.bin_func(mzs, ints))
        spec = th.as_tensor(bin_spec, dtype=th.float32).unsqueeze(0)
        col_energy = spec_entry[self.ce_key]
        inst_type = spec_entry["inst_type"]
        prec_type = spec_entry["prec_type"]
        frag_mode = spec_entry["frag_mode"]
        charge = data_utils.get_charge(prec_type)
        inst_type_idx = self.inst_type_c2i[inst_type]
        prec_type_idx = self.prec_type_c2i[prec_type]
        frag_mode_idx = self.frag_mode_c2i[frag_mode]
        # same as prec_mz_bin but tensor
        prec_mz_idx = th.tensor(
            min(prec_mz_bin, spec.shape[1] - 1)).unsqueeze(0)
        assert prec_mz_idx < spec.shape[1], (prec_mz_bin,
                                             prec_mz_idx, spec.shape)
        if self.preproc_ce == "normalize":
            col_energy_meta = th.tensor(
                [(col_energy - self.mean_ce) / (self.std_ce + EPS)], dtype=th.float32)
        elif self.preproc_ce == "quantize":
            ce_bins = np.arange(0, 161, step=20)  # 8 bins
            ce_idx = np.digitize(col_energy, bins=ce_bins, right=False)
            col_energy_meta = th.ones([len(ce_bins) + 1], dtype=th.float32)
            col_energy_meta[ce_idx] = 1.
        else:
            assert self.preproc_ce == "none", self.preproc_ce
            col_energy_meta = th.tensor([col_energy], dtype=th.float32)
        inst_type_meta = th.as_tensor(
            np_one_hot(
                inst_type_idx,
                num_classes=self.num_inst_type),
            dtype=th.float32)
        prec_type_meta = th.as_tensor(
            np_one_hot(
                prec_type_idx,
                num_classes=self.num_prec_type),
            dtype=th.float32)
        frag_mode_meta = th.as_tensor(
            np_one_hot(
                frag_mode_idx,
                num_classes=self.num_frag_mode),
            dtype=th.float32)
        spec_meta_list = [
            col_energy_meta,
            inst_type_meta,
            prec_type_meta,
            frag_mode_meta,
            col_energy_meta]
        spec_meta = th.cat(spec_meta_list, dim=0).unsqueeze(0)
        spec_feats = {
            "spec": spec,
            "prec_mz": [prec_mz],
            "prec_mz_bin": [prec_mz_bin],
            "prec_diff": [prec_diff],
            "num_peaks": [num_peaks],
            "inst_type": [inst_type],
            "prec_type": [prec_type],
            "frag_mode": [frag_mode],
            "col_energy": [col_energy],
            "charge": [charge],
            "spec_meta": spec_meta,
            "mol_id": mol_id,
            "spec_id": spec_id,
            "group_id": group_id,
            "prec_mz_idx": prec_mz_idx
        }
        if "casmi_id" in spec_entry:
            spec_feats["casmi_id"] = th.tensor(
                spec_entry["casmi_id"]).unsqueeze(0)
        if "lda_topic" in spec_entry:
            spec_feats["lda_topic"] = th.tensor(
                spec_entry["lda_topic"]).unsqueeze(0)
        return spec_feats

    def get_dataloaders(self, run_d):

        val_frac = run_d["val_frac"]
        test_frac = run_d["test_frac"]
        sec_frac = run_d["sec_frac"]
        split_key = run_d["split_key"]
        split_seed = run_d["split_seed"]
        ignore_casmi = run_d["ignore_casmi_in_split"]
        assert run_d["batch_size"] % run_d["grad_acc_interval"] == 0
        batch_size = run_d["batch_size"] // run_d["grad_acc_interval"]
        num_workers = run_d["num_workers"]
        pin_memory = run_d["pin_memory"] if run_d["device"] != "cpu" else False

        train_mask, val_mask, test_mask, sec_masks = self.get_split_masks(
            val_frac, test_frac, sec_frac, split_key, split_seed, ignore_casmi)
        all_idx = np.arange(len(self))
        # th_data.RandomSampler()
        train_ss = TrainSubset(self, all_idx[train_mask])
        # th_data.RandomSampler(th_data.Subset(self,all_idx[val_mask]))
        val_ss = th_data.Subset(self, all_idx[val_mask])
        # th_data.RandomSampler(th_data.Subset(self,all_idx[test_mask]))
        test_ss = th_data.Subset(self, all_idx[test_mask])
        sec_ss = [th_data.Subset(self, all_idx[sec_mask])
                  for sec_mask in sec_masks]

        collate_fn = self.get_collate_fn()
        if len(train_ss) > 0:
            train_dl = th_data.DataLoader(
                train_ss,
                batch_size=batch_size,
                collate_fn=collate_fn,
                num_workers=num_workers,
                pin_memory=pin_memory,
                shuffle=True,
                drop_last=True  # this is to prevent single data batches that mess with batchnorm
            )
            train_dl_2 = th_data.DataLoader(
                train_ss,
                batch_size=batch_size,
                collate_fn=collate_fn,
                num_workers=num_workers,
                pin_memory=pin_memory,
                shuffle=False,
                drop_last=False
            )
        else:
            train_dl = train_dl_2 = None
        if len(val_ss) > 0:
            val_dl = th_data.DataLoader(
                val_ss,
                batch_size=batch_size,
                collate_fn=collate_fn,
                num_workers=num_workers,
                pin_memory=pin_memory,
                shuffle=False,
                drop_last=False
            )
        else:
            val_dl = None
        if len(test_ss) > 0:
            test_dl = th_data.DataLoader(
                test_ss,
                batch_size=batch_size,
                collate_fn=collate_fn,
                num_workers=num_workers,
                pin_memory=pin_memory,
                shuffle=False,
                drop_last=False
            )
        else:
            test_dl = None
        sec_dls = []
        for ss in sec_ss:
            dl = th_data.DataLoader(
                ss,
                batch_size=batch_size,
                collate_fn=collate_fn,
                num_workers=num_workers,
                pin_memory=pin_memory,
                shuffle=False,
                drop_last=False
            )
            sec_dls.append(dl)

        # set up dl_dict
        dl_dict = {}
        dl_dict["train"] = train_dl
        dl_dict["primary"] = {
            "train": train_dl_2,
            "val": val_dl,
            "test": test_dl
        }
        dl_dict["secondary"] = {}
        for sec_idx, sec_dset in enumerate(self.secondary_dset):
            dl_dict["secondary"][f"{sec_dset}"] = sec_dls[sec_idx]

        # set up split_id_dict
        split_id_dict = {}
        split_id_dict["primary"] = {}
        split_id_dict["primary"]["train"] = self.spec_df.iloc[all_idx[train_mask]
                                                              ]["spec_id"].to_numpy()
        split_id_dict["primary"]["val"] = self.spec_df.iloc[all_idx[val_mask]
                                                            ]["spec_id"].to_numpy()
        split_id_dict["primary"]["test"] = self.spec_df.iloc[all_idx[test_mask]
                                                             ]["spec_id"].to_numpy()
        split_id_dict["secondary"] = {}
        for sec_idx, sec_dset in enumerate(self.secondary_dset):
            split_id_dict["secondary"][sec_dset] = self.spec_df.iloc[all_idx[sec_masks[sec_idx]]
                                                                     ]["spec_id"].to_numpy()

        return dl_dict, split_id_dict

    def get_track_dl(
            self,
            idx,
            num_rand_idx=0,
            topk_idx=None,
            bottomk_idx=None,
            other_idx=None,
            spec_ids=None):

        track_seed = 5585
        track_dl_dict = {}
        collate_fn = self.get_collate_fn()
        if num_rand_idx > 0:
            with np_temp_seed(track_seed):
                rand_idx = np.random.choice(
                    idx, size=num_rand_idx, replace=False)
            rand_dl = th_data.DataLoader(
                th_data.Subset(self, rand_idx),
                batch_size=1,
                collate_fn=collate_fn,
                num_workers=0,
                pin_memory=False,
                shuffle=False,
                drop_last=False
            )
            track_dl_dict["rand"] = rand_dl
        if not (topk_idx is None):
            topk_idx = idx[topk_idx]
            topk_dl = th_data.DataLoader(
                th_data.Subset(self, topk_idx),
                batch_size=1,
                collate_fn=collate_fn,
                num_workers=0,
                pin_memory=False,
                shuffle=False,
                drop_last=False
            )
            track_dl_dict["topk"] = topk_dl
        if not (bottomk_idx is None):
            bottomk_idx = idx[bottomk_idx]
            bottomk_dl = th_data.DataLoader(
                th_data.Subset(self, bottomk_idx),
                batch_size=1,
                collate_fn=collate_fn,
                num_workers=0,
                pin_memory=False,
                shuffle=False,
                drop_last=False
            )
            track_dl_dict["bottomk"] = bottomk_dl
        if not (other_idx is None):
            other_idx = idx[other_idx]
            other_dl = th_data.DataLoader(
                th_data.Subset(self, other_idx),
                batch_size=1,
                collate_fn=collate_fn,
                num_workers=0,
                pin_memory=False,
                shuffle=False,
                drop_last=False
            )
            track_dl_dict["other"] = other_dl
        if not (spec_ids is None):
            # preserves order
            spec_idx = []
            for spec_id in spec_ids:
                spec_idx.append(
                    int(self.spec_df[self.spec_df["spec_id"] == spec_id].index[0]))
            spec_idx = np.array(spec_idx)
            spec_dl = th_data.DataLoader(
                th_data.Subset(self, spec_idx),
                batch_size=1,
                collate_fn=collate_fn,
                num_workers=0,
                pin_memory=False,
                shuffle=False,
                drop_last=False
            )
            track_dl_dict["spec"] = spec_dl
        return track_dl_dict

    def compute_lda(self, run_d):
        
        if not run_d["lda_topic_loss"]:
            return

        def lda_transform(spec):
            spec_max = np.maximum(EPS,np.max(spec,axis=1).reshape(-1,1))
            # convert each peak to an integer percentage of maximum
            spec = (spec / spec_max) * 100.
            spec = np.round(spec).astype(int)
            return spec

        val_frac = run_d["val_frac"]
        test_frac = run_d["test_frac"]
        sec_frac = run_d["sec_frac"]
        split_key = run_d["split_key"]
        split_seed = run_d["split_seed"]
        ignore_casmi = run_d["ignore_casmi_in_split"]
        train_mask, _, _, _ = self.get_split_masks(
            val_frac, test_frac, sec_frac, split_key, split_seed, ignore_casmi)
        all_idx = np.arange(len(self))
        train_idx = all_idx[train_mask]
        train_spec_id = self.spec_df.iloc[train_idx]["spec_id"].to_numpy()
        # load all data
        all_data = self.load_all(["spec_id","spec"])
        all_spec_id = all_data["spec_id"].numpy()
        other_spec_id = np.setdiff1d(all_spec_id,train_spec_id)
        train_spec = all_data["spec"][np.isin(all_spec_id,train_spec_id)]
        # untransform
        train_spec = spec_utils.unprocess_spec(train_spec,self.transform).numpy()
        # lda transform
        train_spec = lda_transform(train_spec)
        # fit lda
        lda = LatentDirichletAllocation(
            n_components=100, # hardcoded
            verbose=3,
            n_jobs=min(10, len(os.sched_getaffinity(0))),
            random_state=run_d["train_seed"]
        )
        lda_topic = lda.fit_transform(train_spec)
        # create dummy entries for other_spec_id
        spec_id = np.concatenate([train_spec_id,other_spec_id],axis=0)
        lda_topic = np.concatenate([lda_topic,np.zeros((other_spec_id.shape[0],lda_topic.shape[1]))],axis=0)
        lda_topic = [row[0] for row in np.split(lda_topic,lda_topic.shape[0],axis=0)]
        # add them to the spec_df
        lda_df = pd.DataFrame(
            {
                "spec_id": spec_id,
                "lda_topic": lda_topic
            }
        )
        self.spec_df = self.spec_df.merge(
            lda_df,
            on=["spec_id"],
            how="inner"
        )

    def get_data_dims(self):

        data = self.__getitem__(0)
        dim_d = {}
        if self.is_fp_dset:
            fp_dim = data["fp"].shape[1]
        else:
            fp_dim = -1
        if self.is_graph_dset:
            # node
            if self.atom_feature_mode == "pretrain":
                n_dim = -1
            else:
                n_dim = data["graph"].ndata['h'].shape[1]
            # edge
            if self.bond_feature_mode == "none":
                e_dim = 0
            elif self.bond_feature_mode == "pretrain":
                e_dim = -1
            else:
                e_dim = data["graph"].edata['h'].shape[1]
        else:
            n_dim = e_dim = -1
        c_dim = l_dim = -1
        if self.spec_meta_global:
            g_dim = data["spec_meta"].shape[1]
        else:
            g_dim = 0  # -1
        o_dim = data["spec"].shape[1]
        dim_d = {
            "fp_dim": fp_dim,
            "n_dim": n_dim,
            "e_dim": e_dim,
            "c_dim": c_dim,
            "l_dim": l_dim,
            "g_dim": g_dim,
            "o_dim": o_dim
        }
        return dim_d

    def get_collate_fn(self):

        def _collate(data_ds):
            # check for rebatching
            if isinstance(data_ds[0], list):
                data_ds = flatten_lol(data_ds)
            assert isinstance(data_ds[0], dict)
            batch_data_d = {k: [] for k in data_ds[0].keys()}
            for data_d in data_ds:
                for k, v in data_d.items():
                    batch_data_d[k].append(v)
            for k, v in batch_data_d.items():
                if isinstance(data_ds[0][k], th.Tensor):
                    batch_data_d[k] = th.cat(v, dim=0)
                elif isinstance(data_ds[0][k], list):
                    batch_data_d[k] = flatten_lol(v)
                elif isinstance(data_ds[0][k], dgl.DGLGraph):
                    batch_data_d[k] = dgl.batch(v)
                elif isinstance(data_ds[0][k], torch_geometric.data.Data):
                    if k == "gf_v2_data":
                        batch_data_d[k] = gf_data_utils.collator(
                            v
                        )
                    else:
                        assert k == "esp_graph", k
                        batch_data_d[k] = esp_data_utils.collator(
                            v
                        )
                else:
                    raise ValueError(f"{type(data_ds[0][k])} is not supported")
            return batch_data_d

        return _collate

    def process_entry(self, spec_entry, mol):

        # initialize data with shared attributes
        spec_feats = self.get_spec_feats(spec_entry)
        data = {**spec_feats}
        data["smiles"] = [data_utils.mol_to_smiles(mol)]
        data["formula"] = [data_utils.mol_to_formula(mol)]
        # add dset_type specific attributes
        if self.is_fp_dset:
            assert len(self.fp_types) > 0, self.fp_types
            fps = []
            if "morgan" in self.fp_types:
                fp = data_utils.make_morgan_fingerprint(mol)
                fps.append(fp)
            if "rdkit" in self.fp_types:
                fp = data_utils.make_rdkit_fingerprint(mol)
                fps.append(fp)
            if "maccs" in self.fp_types:
                fp = data_utils.make_maccs_fingerprint(mol)
                fps.append(fp)
            fps = th.cat([th.as_tensor(fp, dtype=th.float32)
                         for fp in fps], dim=0).unsqueeze(0)
            data["fp"] = fps
        if self.is_graph_dset:
            atom_featurizer = self.get_atom_featurizer()
            bond_featurizer = self.get_bond_featurizer()
            graph = chemutils.mol_to_bigraph(
                mol,
                node_featurizer=atom_featurizer,
                edge_featurizer=bond_featurizer,
                add_self_loop=self.self_loop,
                num_virtual_nodes=0  # self.num_virtual_nodes
            )
            if self.spec_meta_node:
                assert self.atom_feature_mode not in [
                    "pretrain"], self.atom_feature_mode
                spec_meta_node = spec_feats["spec_meta"].repeat(
                    graph.num_nodes(), 1)
                graph.ndata['h'] = th.cat(
                    (graph.ndata['h'], spec_meta_node), dim=1)
            data["graph"] = graph
        if self.is_gf_v2_dset:
            gf_v2_data = gf_data_utils.gf_preprocess(
                mol, spec_entry["spec_id"], algos_v=self.gf_algos_v)
            data["gf_v2_data"] = gf_v2_data
        if self.is_esp_dset:
            esp_graph, esp_meta = esp_data_utils.esp_preprocess(
                mol,
                spec_entry["prec_type"],
                spec_entry[self.ce_key],
                [self.prec_type_i2c[i] for i in range(len(self.prec_type_i2c))],
            )
            data["esp_graph"] = esp_graph
            data["esp_meta"] = esp_meta
        if self.casmi_fp:
            fp = data_utils.make_maccs_fingerprint(mol)
            fp = th.as_tensor(fp, dtype=th.float32).unsqueeze(0)
            data["casmi_fp"] = fp
        return data

    def get_atom_featurizer(self):

        assert self.is_graph_dset
        if self.atom_feature_mode == "canonical":
            return chemutils.CanonicalAtomFeaturizer()
        elif self.atom_feature_mode == "pretrain":
            return chemutils.PretrainAtomFeaturizer()
        elif self.atom_feature_mode == 'light':
            atom_featurizer_funs = chemutils.ConcatFeaturizer([
                chemutils.atom_mass,
                data_utils.atom_type_one_hot
            ])
        elif self.atom_feature_mode == 'full':
            atom_featurizer_funs = chemutils.ConcatFeaturizer([
                chemutils.atom_mass,
                data_utils.atom_type_one_hot,
                data_utils.atom_bond_type_one_hot,
                chemutils.atom_degree_one_hot,
                chemutils.atom_total_degree_one_hot,
                chemutils.atom_explicit_valence_one_hot,
                chemutils.atom_implicit_valence_one_hot,
                chemutils.atom_hybridization_one_hot,
                chemutils.atom_total_num_H_one_hot,
                chemutils.atom_formal_charge_one_hot,
                chemutils.atom_num_radical_electrons_one_hot,
                chemutils.atom_is_aromatic_one_hot,
                chemutils.atom_is_in_ring_one_hot,
                chemutils.atom_chiral_tag_one_hot
            ])
        elif self.atom_feature_mode == 'medium':
            atom_featurizer_funs = chemutils.ConcatFeaturizer([
                chemutils.atom_mass,
                data_utils.atom_type_one_hot,
                data_utils.atom_bond_type_one_hot,
                chemutils.atom_total_degree_one_hot,
                chemutils.atom_total_num_H_one_hot,
                chemutils.atom_is_aromatic_one_hot,
                chemutils.atom_is_in_ring_one_hot,
            ])
        else:
            raise ValueError(
                f"Invalid atom_feature_mode: {self.atom_feature_mode}")

        return chemutils.BaseAtomFeaturizer(
            {"h": atom_featurizer_funs}
        )

    def get_bond_featurizer(self):

        assert self.is_graph_dset
        if self.bond_feature_mode == "canonical":
            return chemutils.CanonicalBondFeaturizer()
        elif self.bond_feature_mode == "pretrain":
            return chemutils.PretrainBondFeaturizer()
        elif self.bond_feature_mode == 'light':
            return chemutils.BaseBondFeaturizer(
                featurizer_funcs={'h': chemutils.ConcatFeaturizer([
                    chemutils.bond_type_one_hot
                ])}, self_loop=self.self_loop
            )
        elif self.bond_feature_mode == 'full':
            return chemutils.CanonicalBondFeaturizer(
                bond_data_field='h', self_loop=self.self_loop
            )
        else:
            assert self.bond_feature_mode == 'none'
            return None

    def batch_from_smiles(self, smiles_list, ref_spec_entry):

        data_list = []
        for smiles in smiles_list:
            mol = data_utils.mol_from_smiles(smiles, standardize=True)
            assert not none_or_nan(mol)
            data = self.process_entry(ref_spec_entry, mol)
            data_list.append(data)
        collate_fn = self.get_collate_fn()
        batch_data = collate_fn(data_list)
        return batch_data

    def update_casmi_info(self, casmi_ds):

        query_mol_df = casmi_ds.spec_df[["mol_id"]].merge(
            casmi_ds.mol_df[["mol_id", "inchikey_s", "scaffold"]], on=["mol_id"], how="inner")
        for k in list(self.casmi_info.keys()):
            self.casmi_info[k] = self.casmi_info[k] | set(query_mol_df[k])

    def load_all(self, keys):

        collate_fn = self.get_collate_fn()
        dl = th_data.DataLoader(
            self,
            batch_size=100,
            collate_fn=collate_fn,
            num_workers=min(10, len(os.sched_getaffinity(0))),
            pin_memory=False,
            shuffle=False,
            drop_last=False
        )
        all_ds = []
        for b_idx, b in tqdm(enumerate(dl), total=len(dl), desc="> load_all"):
            b_d = {}
            for k in keys:
                b_d[k] = b[k]
            all_ds.append(b_d)
        all_d = collate_fn(all_ds)
        return all_d


class CASMIDataset(BaseDataset):

    def __init__(self, ds, casmi_type, *dset_types, **kwargs):

        self.is_fp_dset = "fp" in dset_types
        self.is_graph_dset = "graph" in dset_types
        self.is_gf_v2_dset = "gf_v2" in dset_types
        self.is_esp_dset = "esp" in dset_types
        assert (self.is_fp_dset or self.is_graph_dset or self.is_gf_v2_dset)
        for k, v in kwargs.items():
            setattr(self, k, v)
        if casmi_type == "casmi":
            casmi_dp = self.casmi_dp
        elif casmi_type == "pcasmi":
            casmi_dp = self.pcasmi_dp
        else:
            assert casmi_type == "casmi22"
            casmi_dp = self.casmi22_dp
        self.spec_df = pd.read_pickle(
            os.path.join(
                self.proc_dp,
                casmi_dp,
                "spec_df.pkl"))
        self.mol_df = pd.read_pickle(
            os.path.join(
                self.proc_dp,
                casmi_dp,
                "mol_df.pkl"))
        self.cand_df = pd.read_pickle(
            os.path.join(
                self.proc_dp,
                casmi_dp,
                "cand_df.pkl"))
        if casmi_type == "pcasmi":
            self.s_df = pd.read_pickle(
                os.path.join(
                    self.proc_dp,
                    casmi_dp,
                    "s_df.pkl"))
        # select the spectra
        self.spec_df = self.spec_df[self.spec_df["ion_mode"] == "P"].reset_index(
            drop=True)
        if casmi_type in ["casmi","casmi22"] and getattr(self,f"{casmi_type}_num_entries") > -1:
            assert getattr(self,f"{casmi_type}_num_entries") <= self.spec_df.shape[0]
            self.spec_df = self.spec_df.sample(
                n=getattr(self,f"{casmi_type}_num_entries"), replace=False, random_state=6699)
        elif casmi_type == "pcasmi" and self.pcasmi_num_entries > -1:
            assert self.pcasmi_num_entries <= self.spec_df["group_id"].nunique(
            )
            # stratified random sampling
            sample_df = self.spec_df[["group_id", "spec_id_old"]].rename(
                columns={"spec_id_old": "spec_id"})
            sample_df = sample_df.merge(
                self.s_df[["spec_id", "mw_range"]], on="spec_id", how="inner")
            assert sample_df.shape[0] == self.spec_df.shape[0]
            sample_df = sample_df.drop(columns=["spec_id"]).drop_duplicates()
            assert sample_df["group_id"].nunique() == sample_df.shape[0]
            assert sample_df["group_id"].nunique(
            ) == self.spec_df["group_id"].nunique()
            group_ids = []
            k = sample_df["mw_range"].nunique()
            for mw_range_idx, mw_range in enumerate(
                    sample_df["mw_range"].unique().tolist()):
                n = self.pcasmi_num_entries // k
                if mw_range_idx < self.pcasmi_num_entries % k:
                    n += 1
                mw_sample_df = sample_df[sample_df["mw_range"] == mw_range]
                group_ids.extend(
                    mw_sample_df.sample(
                        n=n,
                        replace=False,
                        random_state=6699)["group_id"].tolist())
            assert len(set(group_ids)) == self.pcasmi_num_entries
            self.spec_df = self.spec_df[self.spec_df["group_id"].isin(
                group_ids)]
            # random sampling
            # sample_df = self.spec_df[["group_id"]].drop_duplicates().sample(n=self.pcasmi_num_entries,replace=False,random_state=6699)
            # self.spec_df = self.spec_df[self.spec_df["group_id"].isin(sample_df["group_id"])]
        # extract the query and candidate mol_ids
        query_mol_id = set(
            self.spec_df["mol_id"]) & set(
            self.mol_df["mol_id"]) & set(
            self.cand_df["query_mol_id"])
        cand_mol_id = set(self.cand_df[self.cand_df["query_mol_id"].isin(
            query_mol_id)]["candidate_mol_id"])
        # drop candidates with duplicate inchikey_s (i.e. stereoisomers)
        # in principle, this might drop the matching mol_id for a challenge spectrum!
        # so, we need to do it this way
        self.mol_df = self.mol_df[self.mol_df["mol_id"].isin(
            query_mol_id | cand_mol_id)]
        match_mol = self.mol_df[self.mol_df["mol_id"].isin(query_mol_id)]
        self.mol_df = self.mol_df[~self.mol_df["inchikey_s"].isin(
            match_mol["inchikey_s"])]
        self.mol_df = self.mol_df.drop_duplicates(subset=["inchikey_s"])
        self.mol_df = pd.concat([self.mol_df, match_mol])
        query_mol_id = query_mol_id & set(self.mol_df["mol_id"])
        cand_mol_id = cand_mol_id & set(self.mol_df["mol_id"])
        # remove dropped candidates
        self.spec_df = self.spec_df[self.spec_df["mol_id"].isin(
            query_mol_id)].reset_index(drop=True)
        self.mol_df = self.mol_df[self.mol_df["mol_id"].isin(
            query_mol_id | cand_mol_id)].reset_index(drop=True)
        self.cand_df = self.cand_df[self.cand_df["query_mol_id"].isin(
            query_mol_id) & self.cand_df["candidate_mol_id"].isin(cand_mol_id)].reset_index(drop=True)
        # sanity checks
        assert not self.mol_df["mol"].isna().any()
        if not (self.spec_df["prec_type"] == "[M+H]+").all():
            print("> warning: not all [M+H]+ prec_type")
        assert (self.spec_df["inst_type"] == "FT").all()
        assert (self.spec_df["frag_mode"] == "HCD").all()
        self._copy_from_ds(ds)
        if casmi_type in ["casmi","casmi22"]:
            # this is the classic CASMI challenge
            assert self.spec_df["nce"].isna().all()
            spec_dfs = []
            casmi_nces = getattr(self,f"{casmi_type}_nces")
            for nce_idx, nce in enumerate(casmi_nces):
                spec_df = self.spec_df.copy(deep=True)
                spec_df.loc[:, "nce"] = nce
                # grouping is by mol, one spec per mol
                # group_id is equal to original spec_id, assign new spec_id
                n_spec = spec_df.shape[0]
                spec_df.loc[:, "spec_id"] = np.arange(
                    nce_idx * n_spec, (nce_idx + 1) * n_spec)
                spec_dfs.append(spec_df)
            self.spec_df = pd.concat(spec_dfs, axis=0).reset_index(drop=True)
            del spec_dfs
        df = self.spec_df[["mol_id", "spec_id"]].rename(
            columns={"mol_id": "query_mol_id", "spec_id": "query_spec_id"})
        self.mol_spec_df = self.cand_df.merge(
            df, on="query_mol_id", how="inner")
        del df
        # remove spec with no candidates (this is possible due to filtering)
        self.spec_df = self.spec_df[self.spec_df["spec_id"].isin(
            self.mol_spec_df["query_spec_id"])]
        # set indices for quicker access
        self.spec_df = self.spec_df.set_index(
            "spec_id", drop=False).sort_index().rename_axis(None)
        self.mol_df = self.mol_df.set_index(
            "mol_id", drop=False).sort_index().rename_axis(None)

    def _copy_from_ds(self, ds):

        assert isinstance(ds, BaseDataset)
        attrs = [
            "inst_type_c2i",
            "inst_type_i2c",
            "prec_type_c2i",
            "prec_type_i2c",
            "frag_mode_c2i",
            "frag_mode_i2c",
            "num_inst_type",
            "num_prec_type",
            "num_frag_mode",
            "can_seeds",
            "max_ce", "mean_ce", "std_ce"
        ]
        for attr in attrs:
            if hasattr(ds, attr):
                setattr(self, attr, getattr(ds, attr))

    def get_dataloader(self, run_d, mode, group_id=None):

        if mode == "spec":
            batch_size = self.spec_df.shape[0]
            num_workers = 0
            pin_memory = False
            ds = CASMISpecDataset(self)
        else:
            assert mode == "group"
            batch_size = run_d["casmi_batch_size"]
            if batch_size == -1:
                assert run_d["batch_size"] % run_d["grad_acc_interval"] == 0
                batch_size = run_d["batch_size"] // run_d["grad_acc_interval"]
            num_workers = run_d["casmi_num_workers"]
            if num_workers == -1:
                num_workers = run_d["num_workers"]
            pin_memory = run_d["pin_memory"] if run_d["device"] != "cpu" else False
            ds = CASMIGroupDataset(self, group_id=group_id)
        dl = th_data.DataLoader(
            ds,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            shuffle=False,
            drop_last=False,
            collate_fn=self.get_collate_fn()
        )
        return dl

    def __getitem__(self, idx):

        raise NotImplementedError

    def __len__(self):

        raise NotImplementedError


class CASMISpecDataset(th_data.Dataset):

    def __init__(self, casmi_ds):

        self.spec_df = casmi_ds.spec_df
        self.mol_df = casmi_ds.mol_df
        self.process_entry = casmi_ds.process_entry

    def __getitem__(self, idx):

        spec_entry = self.spec_df.iloc[idx]
        mol_id = spec_entry["mol_id"]
        if mol_id in self.mol_df["mol_id"]:
            mol_entry = self.mol_df.loc[mol_id]
        else:
            raise ValueError(f"mol_id {mol_id} not found in mol_df")
            # just choose an artbitrary mol_entry
            mol_entry = self.mol_df.iloc[0]
        data = self.process_entry(spec_entry, mol_entry["mol"])
        return data

    def __len__(self):

        return self.spec_df.shape[0]


class CASMIGroupDataset(th_data.Dataset):

    def __init__(self, casmi_ds, group_id=None):

        self.spec_df = casmi_ds.spec_df
        self.mol_df = casmi_ds.mol_df
        self.mol_spec_df = casmi_ds.mol_spec_df
        self.process_entry = casmi_ds.process_entry
        if not (group_id is None):
            self.spec_df = self.spec_df[self.spec_df["group_id"] == group_id]
            self.mol_spec_df = self.mol_spec_df[self.mol_spec_df["query_spec_id"].isin(
                self.spec_df["spec_id"])]
            self.mol_df = self.mol_df[self.mol_df["mol_id"].isin(
                self.mol_spec_df["candidate_mol_id"])]

    def __getitem__(self, idx):

        mol_spec_entry = self.mol_spec_df.iloc[idx]
        mol_id = mol_spec_entry["candidate_mol_id"].item()
        spec_id = mol_spec_entry["query_spec_id"].item()
        mol_entry = self.mol_df.loc[mol_id]
        spec_entry = self.spec_df.loc[spec_id].copy()
        # don't use .loc[:,"mol_id"] since it's a single row
        spec_entry.loc["mol_id"] = mol_id
        data = self.process_entry(spec_entry, mol_entry["mol"])
        return data

    def __len__(self):

        return self.mol_spec_df.shape[0]


def get_dset_types(embed_types):

    dset_types = set()
    for embed_type in embed_types:
        if embed_type == "fp":
            dset_types.add("fp")
        elif embed_type in ["gat", "wln", "gin_pt"]:
            dset_types.add("graph")
        elif embed_type in ["gf"]:
            dset_types.add("gf")
        elif embed_type in ["gf_v2"]:
            dset_types.add("gf_v2")
        elif embed_type in ["esp"]:
            dset_types.add("esp")
        else:
            raise ValueError(f"invalid embed_type {embed_type}")
    dset_types = list(dset_types)
    return dset_types


def get_default_ds(data_d_ow=dict(),model_d_ow=dict(),run_d_ow=dict()):

    from massformer.runner import load_config
    template_fp = "config/template.yml"
    custom_fp = None
    device_id = None
    checkpoint_name = None
    _, _, _, data_d, model_d, run_d = load_config(
        template_fp, 
        custom_fp, 
        device_id,
        checkpoint_name
    )
    for k, v in data_d_ow.items():
        if k in data_d:
            data_d[k] = v
    for k,v in model_d_ow.items():
        if k in model_d:
            model_d[k] = v
    for k,v in run_d_ow.items():
        if k in run_d:
            run_d[k] = v
    return data_d, model_d, run_d


def get_dataloader(data_d_ow=dict(),model_d_ow=dict(),run_d_ow=dict()):

    data_d, model_d, run_d = get_default_ds(
        data_d_ow=data_d_ow,
        model_d_ow=model_d_ow,
        run_d_ow=run_d_ow
    )
    dset_types = get_dset_types(model_d["embed_types"])
    ds = BaseDataset(*dset_types, **data_d)
    ds.compute_lda(run_d)
    dl_dict, split_id_dict = ds.get_dataloaders(run_d)
    return ds, dl_dict, data_d, model_d, run_d


def get_esp_dicts():

    data_d_ow = dict(
        num_entries=1000,
        primary_dset=["nist"],
        secondary_dset=[],
        ce_key="nce",
        inst_type=["FT"],
        frag_mode=["HCD"],
        pos_prec_type=['[M+H]+', '[M+H-H2O]+', '[M+H-2H2O]+', '[M+2H]2+', '[M+H-NH3]+', "[M+Na]+"],
        fp_types=["morgan","rdkit","maccs"],
        preproc_ce="none",
        spectrum_normalization="l1",
        casmi_num_entries=-1,
        pcasmi_num_entries=-1,
    )

    model_d_ow = dict(
        embed_types=["esp"],
        model_seed=6666,
        esp_model=True,
    )

    run_d_ow = dict(
        loss="cos",
        sim="cos",
        lr=5e-4, ##
        batch_size=128, ##
        weight_decay=0.0, ##
        num_epochs=2, #100
        num_workers=0, #10,
        cuda_deterministic=False,
        do_test=True,
        do_casmi=False,
        do_pcasmi=False,
        save_state=True,
        save_media=True,
        casmi_save_sim=True,
        casmi_batch_size=400,
        casmi_num_workers=10,
        casmi_pred_all=True,
        pcasmi_pred_all=False,
        log_pcasmi_um=True,
        log_auxiliary=True,
        train_seed=5585,
        split_seed=420,
        split_key="inchikey_s",
        log_tqdm=False,
        test_sets=["train","val","test"],
        save_test_sims=True,
        lda_topic_loss=True,
    )

    return data_d_ow, model_d_ow, run_d_ow


if __name__ == "__main__":

    th.multiprocessing.set_sharing_strategy('file_system')

    data_d_ow, model_d_ow, run_d_ow = get_esp_dicts()

    ds, dl_dict, data_d, model_d, run_d = get_dataloader(
        data_d_ow=data_d_ow,
        model_d_ow=model_d_ow,
        run_d_ow=run_d_ow
    )

    item = ds[0]
    print(item.keys())
    print(item["esp_graph"])
    print(item["esp_meta"])
    print(
        th.mean(item["spec"],dim=1),
        th.std(item["spec"],dim=1),
        th.min(item["spec"],dim=1)[0],
        th.max(item["spec"],dim=1)[0]
    )
    # print(item["lda_topic"])
    
    train_dl = dl_dict["primary"]["train"]
    print(len(train_dl))

    # get one batch
    train_iter = iter(train_dl)
    batch = next(train_iter)
    print(batch.keys())

    # iterate through batches
    train_iter = iter(train_dl)
    for batch in train_iter:
        pass

    import pdb; pdb.set_trace()
