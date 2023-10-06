import numpy as np
import torch as th
import pandas as pd
import os
import time
import argparse

from massformer.dataset import get_dataloader
import massformer.data_utils as data_utils
from massformer.data_utils import par_apply_series, mol_from_smiles, mol_to_smiles, mol_to_inchikey_s, mol_to_mol_weight, smiles_to_selfies, check_mol_props
from massformer.casmi_utils import common_filter, load_mw_cand, prepare_casmi_mol_df, prepare_casmi_cand_df, prepare_casmi_spec_df, proc_cand_smiles


def setup_splits():

    split_d = {}

    mw_range_d = {
        (0., 200.): 75,
        (200., 300.): 75,
        (300., 400.): 75,
        (400., 1e6): 75
    }

    prec_splits = ["[M+H]+", "other"]

    s_dfs = []
    group_count = 0

    primary = "nist"

    model_d_ow = {"embed_types": ["fp"]}
    data_d_ow = {
        "primary_dset": [primary],
        "transform": "none",
        "spectrum_normalization": "l1"}    

    print(f">>> primary dset = {primary}")
    ds, _, data_d, model_d, run_d = get_dataloader(
        data_d_ow=data_d_ow,
        model_d_ow=model_d_ow)
    all_d = ds.load_all(("spec_id", "mol_id", "spec"))

    spec_df = ds.spec_df
    spec_df = spec_df[spec_df["spec_id"].isin(all_d["spec_id"].numpy())]
    mol_df = ds.mol_df
    mol_df = mol_df[mol_df["mol_id"].isin(all_d["mol_id"].numpy())]

    scaffold_df = mol_df[["scaffold", "mol_id"]].groupby(
        "scaffold").size().reset_index(name="counts")
    # filter by scaffold uniqueness
    un_scaffold = scaffold_df[scaffold_df["counts"] == 1]["scaffold"]
    un_mol_df = mol_df[mol_df["scaffold"].isin(un_scaffold)]
    un_spec_df = spec_df[spec_df["mol_id"].isin(un_mol_df["mol_id"])]
    print("scaffold", un_mol_df.shape[0])

    split_d[primary] = {prec_split: None for prec_split in prec_splits}
    for prec_split in prec_splits:
        if prec_split == "[M+H]+":
            f_spec_df = un_spec_df[un_spec_df["prec_type"] == "[M+H]+"]
        else:
            f_spec_df = un_spec_df[un_spec_df["prec_type"] != "[M+H]+"]
        f_mol_df = un_mol_df[un_mol_df["mol_id"].isin(f_spec_df["mol_id"])]
        print(prec_split, f_mol_df.shape[0])

        # filter by multiple collision energy
        ce_df = f_spec_df[["mol_id", "spec_id"]].groupby(
            "mol_id").size().reset_index(name="counts")
        # print(ce_df["counts"].value_counts())
        multi_ce_df = ce_df[ce_df["counts"] >= 3]
        f_mol_df = f_mol_df[f_mol_df["mol_id"].isin(multi_ce_df["mol_id"])]
        f_spec_df = f_spec_df[f_spec_df["mol_id"].isin(f_mol_df["mol_id"])]
        print("multi_ce", f_mol_df.shape[0])

        # stratified sampling by mw range
        num_mols_d = {k: None for k in mw_range_d.keys()}
        for k, v in mw_range_d.items():
            # filter based on mw range
            mw_mol_df = f_mol_df[(f_mol_df["exact_mw"] >= k[0]) & (
                f_mol_df["exact_mw"] < k[1])]
            num_mols_d[k] = mw_mol_df.shape[0]
            # get dataframe of mol_id, prec_type pairs
            s_df = f_spec_df[f_spec_df["mol_id"].isin(mw_mol_df["mol_id"])][[
                "mol_id", "prec_type"]].drop_duplicates()
            # sample v pairs
            s_df = s_df.sample(n=v, replace=False, random_state=420)
            # fill out rest of split information
            s_df.loc[:, "group_id"] = np.arange(
                group_count, group_count + v)
            group_count += v
            s_df.loc[:, "dset"] = primary
            s_df.loc[:, "prec_split"] = prec_split
            s_df.loc[:, "mw_range"] = [k for i in range(s_df.shape[0])]
            s_df = f_spec_df[["mol_id", "prec_type", "spec_id"]].merge(
                s_df, on=["mol_id", "prec_type"], how="inner")
            s_df = s_df.drop(columns=["prec_type"])
            s_dfs.append(s_df)
        print("num_mols", num_mols_d)

    s_dfs = pd.concat(s_dfs, axis=0)

    return s_dfs


if __name__ == "__main__":

    """
    Note: this script will not exactly reproduce the NIST20 Outlier dataset in 
    the manuscript, since PubChem has been upated with more compounds.
    However, the exact NIST 20 Outlier dataset can be downloaded from the 
    Zenodo link in the README, for benchmarking and reproducibility.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("--proc_dp", type=str, default="data/proc")
    parser.add_argument("--raw_dp", type=str, default="data/raw")
    parser.add_argument("--pcasmi_output_dir", type=str, default="nist20_outlier")
    parser.add_argument("--cid_smiles_file", type=str, default="cid_smiles.tsv")
    args = parser.parse_args()

    th.multiprocessing.set_sharing_strategy("file_system")

    pcasmi_dp = os.path.join(args.proc_dp, args.pcasmi_output_dir)
    cid_smiles_fp = os.path.join(args.raw_dp, args.cid_smiles_file)
    os.makedirs(pcasmi_dp, exist_ok=True)
    
    cand_fp_1 = os.path.join(pcasmi_dp, "cand_df_1.pkl")
    s_fp = os.path.join(pcasmi_dp, "s_df.pkl")
    if not os.path.isfile(s_fp):
        s_df = setup_splits()
        s_df.to_pickle(s_fp)
    else:
        s_df = pd.read_pickle(s_fp)
    s_df = s_df[(s_df["dset"] == "nist") & (s_df["prec_split"]
                                            == "[M+H]+")][["spec_id", "mol_id", "group_id"]]
    spec_df = pd.read_pickle(os.path.join(args.proc_dp, "spec_df.pkl"))
    mol_df = pd.read_pickle(os.path.join(args.proc_dp, "mol_df.pkl"))
    spec_df = spec_df.merge(s_df, on=["spec_id", "mol_id"], how="inner")
    mol_df = mol_df[mol_df["mol_id"].isin(s_df["mol_id"])]
    cand_df = load_mw_cand(
        pcasmi_dp,
        mol_df,
        cid_smiles_fp,
        retmax=10000,
        weight_tol=5e-7)
    cand_df.to_pickle(cand_fp_1)

    # convert to casmi format
    # casmi_mol_df: dataframe of both candidate and query molecules
    # casmi_cand_df: dataframe that maps each query spec_id to the mol_id of
    # the candidates

    # process candidates
    cand_fp_2 = os.path.join(pcasmi_dp, "cand_df_2.pkl")
    if not os.path.isfile(cand_fp_2):
        cand_df = cand_df.drop(columns=["query_mol_id", "candidate_cid"])
        # check if smiles can be parsed correctly into different forms
        # still isomeric at this point
        cand_df.loc[:, "clean"] = par_apply_series(
            cand_df["candidate_smiles"], proc_cand_smiles)
        cand_df = cand_df.dropna().drop(columns=["clean"])
        cand_df.to_pickle(cand_fp_2)
    else:
        cand_df = pd.read_pickle(cand_fp_2)

    # remove query smiles candidates, then add them back
    cand_df = cand_df[cand_df["candidate_smiles"] != cand_df["query_smiles"]]
    query_smiles = cand_df["query_smiles"].unique()
    cand_df = pd.concat([cand_df, pd.DataFrame(
        {"query_smiles": query_smiles, "candidate_smiles": query_smiles})], axis=0)

    casmi_mol_df = prepare_casmi_mol_df(mol_df, cand_df, pcasmi_dp)

    casmi_spec_df = prepare_casmi_spec_df(
        spec_df, mol_df, casmi_mol_df, pcasmi_dp)

    casmi_cand_df = prepare_casmi_cand_df(cand_df, casmi_mol_df, pcasmi_dp)

    # make sure the datasets are internally consistent
    casmi_spec_df, casmi_mol_df, casmi_cand_df = common_filter(
        casmi_spec_df, casmi_mol_df, casmi_cand_df)

    # save dfs
    casmi_spec_df.to_pickle(os.path.join(pcasmi_dp, "spec_df.pkl"))
    casmi_mol_df.to_pickle(os.path.join(pcasmi_dp, "mol_df.pkl"))
    casmi_cand_df.to_pickle(os.path.join(pcasmi_dp, "cand_df.pkl"))

    # export smiles for CFM
    export_mol_df = casmi_mol_df[["mol_id", "smiles"]]
    export_mol_df.to_csv(
        os.path.join(
            pcasmi_dp,
            "all_smiles.txt"),
        sep=" ",
        header=False,
        index=False)
