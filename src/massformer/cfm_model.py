import torch as th
import torch.nn as nn
import numpy as np
import pandas as pd
import os

import massformer.data_utils as data_utils


class CFMPredictor(nn.Module):

    def __init__(self, cfm_dp, ds, do_casmi, do_pcasmi, do_casmi22, use_rb=True):

        super().__init__()
        # load cfm spectra
        self.primary_df = pd.read_pickle(
            os.path.join(cfm_dp, "all_spec_df.pkl"))
        self.primary_mol_ids = set(self.primary_df["mol_id"].tolist())
        self.primary_df = self.primary_df.set_index(
            ["mol_id", "ace", "prec_type"])
        if use_rb:
            self.primary_rb_df = pd.read_pickle(
                os.path.join(cfm_dp, "all_rb_spec_df.pkl"))
            self.primary_rb_mol_ids = set(
                self.primary_rb_df["mol_id"].tolist())
            self.primary_rb_df = self.primary_rb_df.set_index(
                ["mol_id", "ace", "prec_type"])
        else:
            self.primary_rb_df = pd.DataFrame().reindex_like(self.primary_df)
            self.primary_rb_mol_ids = set()
        for casmi_type, casmi_flag in zip(["casmi","pcasmi","casmi22"],[do_casmi, do_pcasmi, do_casmi22]):
            if not casmi_flag:
                continue
            setattr(self, f"{casmi_type}_df", pd.read_pickle(os.path.join(cfm_dp, f"{casmi_type}_spec_df.pkl")))
            setattr(self, f"{casmi_type}_mol_ids", set(getattr(self, f"{casmi_type}_df")["mol_id"].tolist()))
            setattr(self, f"{casmi_type}_df", getattr(self, f"{casmi_type}_df").set_index(["mol_id", "ace", "prec_type"]))
            if use_rb:
                setattr(self, f"{casmi_type}_rb_df", pd.read_pickle(os.path.join(cfm_dp, f"{casmi_type}_rb_spec_df.pkl")))
                setattr(self, f"{casmi_type}_rb_mol_ids", set(getattr(self, f"{casmi_type}_rb_df")["mol_id"].tolist()))
                setattr(self, f"{casmi_type}_rb_df", getattr(self, f"{casmi_type}_rb_df").set_index(["mol_id", "ace", "prec_type"]))
            else:
                setattr(self, f"{casmi_type}_rb_df", pd.DataFrame().reindex_like(getattr(self, f"{casmi_type}_df")))
                setattr(self, f"{casmi_type}_rb_mol_ids", set())
        self.mode = "primary"
        self.supported_aces = np.array([10., 20., 40.])
        self.supported_prec_types = ["[M+H]+"]
        self.default_prec_type = "[M+H]+"
        self.spec_func = lambda mzs, ints: ds.transform_func(
            ds.bin_func(mzs, ints))
        self.dummy_weight = nn.Parameter(data=th.ones(1,), requires_grad=True)
        dummy_mzs, dummy_ints = list(zip(*self.primary_df.iloc[0]["peaks"]))
        self.dummy_spec = np.ones_like(self.spec_func(dummy_mzs, dummy_ints))
        self.dummy_spec = self.dummy_spec / self.dummy_spec.shape[0]

    def set_mode(self, mode):
        assert mode in ["primary", "casmi", "pcasmi", "casmi22"]
        self.mode = mode

    def _get_mode_data(self):
        spec_df = getattr(self,f"{self.mode}_df")
        mol_ids = getattr(self,f"{self.mode}_mol_ids")
        rb_spec_df = getattr(self,f"{self.mode}_rb_df")
        rb_mol_ids = getattr(self,f"{self.mode}_rb_mol_ids")
        return spec_df, mol_ids, rb_spec_df, rb_mol_ids

    def _map_nce(self, nce, charge, prec_mz):
        """ assumes nce is a float """

        # convert to ace
        ace = data_utils.nce_to_ace_helper(nce, charge, prec_mz)
        # map to nearest supported ce
        supported_dists = np.abs(ace - self.supported_aces)
        supported_ace = int(self.supported_aces[np.argmin(supported_dists)])
        return supported_ace

    def forward(self, data, perturb=None, amp=False):

        spec_df, mol_ids, rb_spec_df, rb_mol_ids = self._get_mode_data()
        b_nce = data["col_energy"]
        b_charge = data["charge"]
        b_prec_mz = data["prec_mz"]
        b_ace = [
            self._map_nce(
                nce,
                charge,
                prec_mz) for (
                nce,
                charge,
                prec_mz) in zip(
                b_nce,
                b_charge,
                b_prec_mz)]
        b_mol_id = [int(mol_id) for mol_id in data["mol_id"].tolist()]
        b_prec_type = data["prec_type"]
        # select spectra
        b_spec = []
        for ace, mol_id, prec_type in zip(b_ace, b_mol_id, b_prec_type):
            row = None
            if mol_id in rb_mol_ids:
                # check if it's in the thing
                try:
                    row = rb_spec_df.loc[(mol_id, ace, prec_type)]
                except KeyError as e:
                    pass
            if (row is None) and (mol_id in mol_ids):
                # check if the precursor type is supported
                if prec_type not in self.supported_prec_types:
                    prec_type = self.default_prec_type
                row = spec_df.loc[(mol_id, ace, prec_type)]
            if row is None:
                spec = np.copy(self.dummy_spec)
            else:
                mzs, ints = zip(*row["peaks"])
                spec = self.spec_func(mzs, ints)
            b_spec.append(spec)
        b_spec = th.as_tensor(np.vstack(b_spec), dtype=th.float32)
        b_spec = b_spec + 0. * self.dummy_weight  # silly trick for autograd
        output_d = {"pred": b_spec}
        return output_d

    def get_attn_mats(self, data):
        raise NotImplementedError

    def get_split_params(self):
        raise NotImplementedError

    def count_parameters(self):
        return 0, 0, 0
