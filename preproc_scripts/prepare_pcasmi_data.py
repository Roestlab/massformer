import numpy as np
import torch as th
import pandas as pd
import os
import time
import argparse
from tqdm import tqdm
import urllib
import json

from massformer.dataset import get_dataloader
from prepare_casmi_data import common_filter
import massformer.data_utils as data_utils
from massformer.data_utils import par_apply_series, mol_from_smiles, mol_to_smiles, mol_to_inchikey_s, mol_to_mol_weight, smiles_to_selfies, check_mol_props


def load_mw_cand(
        proc_dp,
        mol_df,
        cid_smiles_fp,
        retmax=5000,
        weight_tol=10e-6):

    cand_cid_fp = os.path.join(proc_dp, "cand_mw_cid_df.pkl")
    cand_fp = os.path.join(proc_dp, "cand_mw_df.pkl")

    if os.path.isfile(cand_fp):

        print(f"> loading cand_df: {cand_fp}")
        cand_df = pd.read_pickle(cand_fp)

    else:

        if os.path.isfile(cand_cid_fp):

            print(f"> loading cand_cid_df: {cand_cid_fp}")
            cand_cid_df = pd.read_pickle(cand_cid_fp)

        else:

            mol_ids = mol_df["mol_id"].tolist()
            mol_weights = mol_df["exact_mw"].tolist()
            smiles = mol_df["smiles"].tolist()

            # get the CIDs from Entrez
            assert retmax <= 100000, retmax
            base_path = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pccompound&api_key=effce203aca460a0fe9c4b372d8742995c08&retmode=JSON"
            retry_cooldown = 1

            valid_entries = []
            bad_ids = []

            with tqdm(total=len(mol_weights), desc="search_mw") as pbar:
                i = 0
                while i < len(mol_weights):
                    mol_weight = mol_weights[i]
                    upper_weight = (1. + weight_tol) * mol_weight
                    lower_weight = (1. - weight_tol) * mol_weight
                    full_path = f"{base_path}&retmax={retmax}&term={lower_weight:.9f}%3A{upper_weight:.9f}%5BExactMass%5D"
                    # print(i,mol_weight,full_path)
                    try:
                        with urllib.request.urlopen(full_path) as url:
                            data = json.loads(url.read().decode())
                        cur_cids = data["esearchresult"]["idlist"]
                        for cur_cid in cur_cids:
                            valid_entries.append(
                                (mol_ids[i], smiles[i], int(cur_cid)))
                        i += 1
                        pbar.update(1)
                    except urllib.error.HTTPError as http_err:
                        print(
                            f"> HTTPError {http_err} encountered, trying again in {retry_cooldown} min(s)")
                        time.sleep(retry_cooldown * 60)
                    except json.decoder.JSONDecodeError as json_err:
                        print(
                            f"> JSONDecodeError {json_err} encountered, trying again in {retry_cooldown} min(s)")
                        time.sleep(retry_cooldown * 60)
                    except OSError as os_err:
                        print(
                            f"> OSError {os_err} encountered, trying again in {retry_cooldown} min(s)")
                        time.sleep(retry_cooldown * 60)

            print(f"> number of bad mws: {len(bad_ids)}")

            cand_cid_df = pd.DataFrame(
                valid_entries,
                columns=[
                    "query_mol_id",
                    "query_smiles",
                    "candidate_cid"])
            cand_cid_df.to_pickle(cand_cid_fp)

        # map CID to SMILES
        print(f"> loading cid_to_smiles: {cid_smiles_fp}")
        cid_smiles_df = pd.read_csv(
            cid_smiles_fp, sep="\t", index_col=False, names=[
                "candidate_cid", "candidate_smiles"])
        cand_df = cand_cid_df.merge(
            cid_smiles_df, on=["candidate_cid"], how="inner")
        cand_df = cand_df.dropna(
            subset=["candidate_smiles"]).reset_index(
            drop=True)
        cand_df.to_pickle(cand_fp)

    cand_count = cand_df.groupby(
        by=["query_mol_id", "query_smiles"]).size().reset_index(name="count")["count"]
    print(
        f"> number of candidates: total = {cand_count.sum()}, mean = {cand_count.mean()}, min = {cand_count.min()}, max = {cand_count.max()}")

    return cand_df


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

    for primary in ["nist"]:

        print(f">>> primary dset = {primary}")
        ds, _ = get_dataloader(["fp"], primary_dset=[
                               primary], transform="none", spectrum_normalization="l1")  # ,num_entries=10000)
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


def proc_cand_smiles(smiles):

    orig_smiles = smiles
    if "." in smiles:
        return np.nan
    mol = mol_from_smiles(smiles)
    smiles = mol_to_smiles(mol)
    inchikey_s = mol_to_inchikey_s(mol)
    selfies = smiles_to_selfies(smiles)
    if mol is np.nan or smiles is np.nan or inchikey_s is np.nan or selfies is np.nan:
        return np.nan
    return orig_smiles


def prepare_casmi_mol_df(mol_df, cand_df, casmi_dp):

    print("> starting mol")
    casmi_mol_fp = os.path.join(casmi_dp, "mol_df.pkl")
    if os.path.isfile(casmi_mol_fp):
        casmi_mol_df = pd.read_pickle(casmi_mol_fp)
    else:
        casmi_mol_df = pd.DataFrame(
            {"smiles_iso": cand_df["candidate_smiles"].unique()})
        assert np.all(mol_df["smiles"].isin(casmi_mol_df["smiles_iso"]))
        casmi_mol_df.loc[:, "mol"] = par_apply_series(
            casmi_mol_df["smiles_iso"], data_utils.mol_from_smiles)
        casmi_mol_df = check_mol_props(casmi_mol_df)
        casmi_mol_df.loc[:, "smiles"] = par_apply_series(
            casmi_mol_df["mol"], data_utils.mol_to_smiles)
        casmi_mol_df.loc[:, "inchikey_s"] = par_apply_series(
            casmi_mol_df["mol"], data_utils.mol_to_inchikey_s)
        print(
            f"> # smiles_iso = {casmi_mol_df['smiles_iso'].nunique()}, # smiles = {casmi_mol_df['smiles'].nunique()}, # inchikey_s = {casmi_mol_df['inchikey_s'].nunique()}")
        casmi_mol_df.loc[:, "mol_id"] = np.arange(casmi_mol_df.shape[0])
        casmi_mol_df.loc[:, "scaffold"] = par_apply_series(
            casmi_mol_df["mol"], data_utils.get_murcko_scaffold)
        casmi_mol_df.loc[:, "formula"] = par_apply_series(
            casmi_mol_df["mol"], data_utils.mol_to_formula)
        casmi_mol_df.loc[:, "inchi"] = par_apply_series(
            casmi_mol_df["mol"], data_utils.mol_to_inchi)
        casmi_mol_df.loc[:, "mw"] = par_apply_series(
            casmi_mol_df["mol"], lambda mol: data_utils.mol_to_mol_weight(mol, exact=False))
        casmi_mol_df.loc[:, "exact_mw"] = par_apply_series(
            casmi_mol_df["mol"], lambda mol: data_utils.mol_to_mol_weight(mol, exact=True))
        casmi_mol_df.to_pickle(casmi_mol_fp)
    return casmi_mol_df


def prepare_casmi_spec_df(spec_df, mol_df, casmi_mol_df, casmi_dp):

    # remap mol_id in casmi_spec_df
    print("> starting spec")
    casmi_spec_fp = os.path.join(casmi_dp, "spec_df.pkl")
    if os.path.isfile(casmi_spec_fp):
        casmi_spec_df = pd.read_pickle(casmi_spec_fp)
    else:
        casmi_spec_df = spec_df.copy()
        casmi_spec_df.loc[:, "spec_id_old"] = casmi_spec_df["spec_id"].copy()
        casmi_spec_df.loc[:, "mol_id_old"] = casmi_spec_df["mol_id"].copy()
        casmi_spec_df = casmi_spec_df.drop(columns=["mol_id"])
        spec_id_to_smiles = spec_df[["spec_id", "mol_id"]].merge(mol_df[["mol_id", "smiles"]], on="mol_id", how="inner").drop(
            columns=["mol_id"]).rename(columns={"smiles": "smiles_iso"})
        spec_id_to_mol_id = spec_id_to_smiles.merge(casmi_mol_df[[
                                                    "mol_id", "smiles_iso"]], on="smiles_iso", how="inner").drop(columns=["smiles_iso"])
        casmi_spec_df = casmi_spec_df.merge(
            spec_id_to_mol_id, on="spec_id", how="inner")
        casmi_spec_df.loc[:, "spec_id"] = np.arange(casmi_spec_df.shape[0])
        casmi_spec_df.to_pickle(casmi_spec_fp)
    return casmi_spec_df


def prepare_casmi_cand_df(cand_df, casmi_mol_df, casmi_dp):

    # create casmi_cand_df (maps query_mol_id to candidate_mol_id)
    print("> starting cand")
    casmi_cand_fp = os.path.join(casmi_dp, "cand_df.pkl")
    if os.path.isfile(casmi_cand_fp):
        casmi_cand_df = pd.read_pickle(casmi_cand_fp)
    else:
        casmi_cand_df = cand_df.copy()
        casmi_cand_df = casmi_cand_df.merge(casmi_mol_df[["smiles_iso", "mol_id"]].rename(
            columns={"smiles_iso": "query_smiles"}), on="query_smiles", how="inner")
        casmi_cand_df = casmi_cand_df.drop(
            columns=["query_smiles"]).rename(
            columns={
                "mol_id": "query_mol_id"})
        casmi_cand_df = casmi_cand_df.merge(casmi_mol_df[["smiles_iso", "mol_id"]].rename(
            columns={"smiles_iso": "candidate_smiles"}), on="candidate_smiles", how="inner")
        casmi_cand_df = casmi_cand_df.drop(
            columns=["candidate_smiles"]).rename(
            columns={
                "mol_id": "candidate_mol_id"})
        casmi_cand_df.to_pickle(os.path.join(casmi_dp, "cand_df.pkl"))
    return casmi_cand_df


if __name__ == "__main__":

    """
    Note: this script will not exactly reproduce the pCASMI data because PubChem has been upated with more compounds.
    However, the exact pcasmi data can be downloaded from the Zenodo link in the README.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("--proc_dp", type=str, default="data/proc")
    parser.add_argument(
        "--cid_smiles_fp",
        type=str,
        default="data/raw/cid_smiles.tsv")
    args = parser.parse_args()

    th.multiprocessing.set_sharing_strategy("file_system")

    pcasmi_dp = os.path.join(args.proc_dp, "pcasmi_1")
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
        args.cid_smiles_fp,
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
            "pcasmi_smiles.txt"),
        sep=" ",
        header=False,
        index=False)
