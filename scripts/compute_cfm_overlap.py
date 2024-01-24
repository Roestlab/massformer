import pandas as pd
import numpy as np

from massformer.data_utils import mol_from_smiles, mol_to_inchikey_s, par_apply_series, get_murcko_scaffold, mol_to_charge
from massformer.runner import load_config, get_ds_model
from massformer.dataset import CASMIDataset


if __name__ == "__main__":

    cfm_fp = "data/raw/cfm/metab_mol_list.txt"
    data_dir = "data"

    # load the datasets
    template_fp = "config/template.yml"
    custom_fp = "config/train_both/train_both_all_MF.yml"
    device_id = -1
    _, _, _, data_d, model_d, run_d = load_config(
        template_fp,
        custom_fp,
        device_id,
        None
    )
    ds, model, _, _, _ = get_ds_model(data_d, model_d, run_d)
    all_spec_df = ds.spec_df
    all_mol_df = ds.mol_df
    casmi_ds = CASMIDataset(ds, "casmi", *["gf_v2"], **data_d)
    casmi_spec_df = casmi_ds.spec_df
    casmi_mol_df = casmi_ds.mol_df
    pcasmi_ds = CASMIDataset(ds, "pcasmi", *["gf_v2"], **data_d)
    pcasmi_spec_df = pcasmi_ds.spec_df
    pcasmi_mol_df = pcasmi_ds.mol_df
    casmi22_ds = CASMIDataset(ds, "casmi22", *["gf_v2"], **data_d)
    casmi22_spec_df = casmi22_ds.spec_df
    casmi22_mol_df = casmi22_ds.mol_df

    # read in the CFM compounds, compute inchikey_s
    cfm_df = pd.read_csv(cfm_fp,delimiter=" ",skiprows=1,names=["metlin_id","smiles","count"])
    cfm_df.loc[:,"mol"] = par_apply_series(cfm_df["smiles"],mol_from_smiles)
    print(cfm_df["mol"].isna().sum())
    cfm_df = cfm_df.dropna(subset=["mol"])
    cfm_df.loc[:,"inchikey_s"] = par_apply_series(cfm_df["mol"],mol_to_inchikey_s)
    assert not cfm_df["inchikey_s"].isna().any()
    cfm_df.loc[:,"scaffold"] = par_apply_series(cfm_df["mol"],get_murcko_scaffold)
    assert not cfm_df["scaffold"].isna().any()
    cfm_inchikey_ses = set(cfm_df["inchikey_s"])
    cfm_scaffolds = set(cfm_df["scaffold"])
    # cfm_df[["metlin_id","smiles","inchikey_s","scaffold"]].to_csv(f"{data_dir}/cfm/cfm_mol_df.csv",index=False)

    # read in NIST/MoNA compounds
    nist_spec_mask = all_spec_df["dset"] == "nist"
    nist_all_spec_df = all_spec_df[nist_spec_mask]
    nist_all_mol_df = all_mol_df[all_mol_df["mol_id"].isin(nist_all_spec_df["mol_id"])]
    nist_all_inchikey_ses = set(nist_all_mol_df["inchikey_s"])
    nist_all_scaffolds = set(nist_all_mol_df["scaffold"])
    nist_all_mol_ids = set(nist_all_mol_df["mol_id"])
    nist_mh_spec_mask = nist_all_spec_df["prec_type"] == "[M+H]+"
    nist_mh_spec_df = nist_all_spec_df[nist_mh_spec_mask]
    nist_mh_mol_df = nist_all_mol_df[nist_all_mol_df["mol_id"].isin(nist_mh_spec_df["mol_id"])]
    nist_mh_inchikey_ses = set(nist_mh_mol_df["inchikey_s"])
    nist_mh_scaffolds = set(nist_mh_mol_df["scaffold"])
    nist_mh_mol_ids = set(nist_mh_mol_df["mol_id"])

    mona_all_spec_mask = all_spec_df["dset"] == "mb_na"
    mona_all_spec_df = all_spec_df[mona_all_spec_mask]
    mona_all_mol_df = all_mol_df[all_mol_df["mol_id"].isin(mona_all_spec_df["mol_id"])]
    mona_all_inchikey_ses = set(mona_all_mol_df["inchikey_s"])
    mona_all_scaffolds = set(mona_all_mol_df["scaffold"])
    mona_all_mol_ids = set(mona_all_mol_df["mol_id"])
    mona_mh_spec_mask = mona_all_spec_df["prec_type"] == "[M+H]+"
    mona_mh_spec_df = mona_all_spec_df[mona_mh_spec_mask]
    mona_mh_mol_df = mona_all_mol_df[mona_all_mol_df["mol_id"].isin(mona_mh_spec_df["mol_id"])]
    mona_mh_inchikey_ses = set(mona_mh_mol_df["inchikey_s"])
    mona_mh_scaffolds = set(mona_mh_mol_df["scaffold"])
    mona_mh_mol_ids = set(mona_mh_mol_df["mol_id"])

    # read in the CASMI 2016 compounds
    casmi_cfm_df = casmi_mol_df[casmi_mol_df["inchikey_s"].isin(cfm_df["inchikey_s"])]
    casmi_query_spec_df = casmi_spec_df.copy() #[casmi_spec_df["ion_mode"]=="P"]
    casmi_query_mol_df = casmi_mol_df[casmi_mol_df["mol_id"].isin(casmi_spec_df["mol_id"])]
    casmi_query_inchikey_ses = set(casmi_query_mol_df["inchikey_s"])
    casmi_query_scaffolds = set(casmi_query_mol_df["scaffold"])
    casmi_cand_mol_df = casmi_mol_df[~casmi_mol_df["mol_id"].isin(casmi_query_spec_df["mol_id"])]
    casmi_cand_inchikey_ses = set(casmi_cand_mol_df["inchikey_s"])
    casmi_cand_scaffolds = set(casmi_cand_mol_df["scaffold"])

    # read in the CASMI 2022 compounds
    casmi22_cfm_df = casmi22_mol_df[casmi22_mol_df["inchikey_s"].isin(cfm_df["inchikey_s"])]
    casmi22_query_spec_df = casmi22_spec_df.copy() #[casmi22_spec_df["ion_mode"]=="P"]
    casmi22_query_mol_df = casmi22_mol_df[casmi22_mol_df["mol_id"].isin(casmi22_spec_df["mol_id"])]
    casmi22_query_inchikey_ses = set(casmi22_query_mol_df["inchikey_s"])
    casmi22_query_scaffolds = set(casmi22_query_mol_df["scaffold"])
    casmi22_cand_mol_df = casmi22_mol_df[~casmi22_mol_df["mol_id"].isin(casmi22_query_spec_df["mol_id"])]
    casmi22_cand_inchikey_ses = set(casmi22_cand_mol_df["inchikey_s"])
    casmi22_cand_scaffolds = set(casmi22_cand_mol_df["scaffold"])

    # read in the pCASMI compounds
    pcasmi_cfm_df = pcasmi_mol_df[pcasmi_mol_df["inchikey_s"].isin(cfm_df["inchikey_s"])]
    pcasmi_query_spec_df = pcasmi_spec_df.copy() #[pcasmi_spec_df["ion_mode"]=="P"]
    pcasmi_query_mol_df = pcasmi_mol_df[pcasmi_mol_df["mol_id"].isin(pcasmi_spec_df["mol_id"])]
    pcasmi_query_inchikey_ses = set(pcasmi_query_mol_df["inchikey_s"])
    pcasmi_query_scaffolds = set(pcasmi_query_mol_df["scaffold"])
    pcasmi_cand_mol_df = pcasmi_mol_df[~pcasmi_mol_df["mol_id"].isin(pcasmi_query_spec_df["mol_id"])]
    pcasmi_cand_inchikey_ses = set(pcasmi_cand_mol_df["inchikey_s"])
    pcasmi_cand_scaffolds = set(pcasmi_cand_mol_df["scaffold"])

    other_dsets = [
        ("NIST [M+H]+", nist_mh_inchikey_ses, nist_mh_scaffolds),
        ("MoNA [M+H]+", mona_mh_inchikey_ses, mona_mh_scaffolds),
        ("NIST All", nist_all_inchikey_ses, nist_all_scaffolds),
        ("MoNA All", mona_all_inchikey_ses, mona_all_scaffolds),
        ("CASMI 2016 (queries)", casmi_query_inchikey_ses, casmi_query_scaffolds),
        ("CASMI 2016 (candidates)", casmi_cand_inchikey_ses, casmi_cand_scaffolds),
        ("CASMI 2022 (queries)", casmi22_query_inchikey_ses, casmi22_query_scaffolds),
        ("CASMI 2022 (candidates)", casmi22_cand_inchikey_ses, casmi22_cand_scaffolds),
        ("NIST-Outlier (queries)", pcasmi_query_inchikey_ses, pcasmi_query_scaffolds),
        ("NIST-Outlier (candidates)", pcasmi_cand_inchikey_ses, pcasmi_cand_scaffolds),
    ]
    counts_df_entries = []
    counts_df_entries.append(
        {
            "dset": "CFM",
            "num_mols": len(cfm_inchikey_ses),
            "num_scaffolds": len(cfm_scaffolds),
        }
    )
    for other_dset in other_dsets:
        counts_df_entry = {
            "dset": other_dset[0],
            "num_mols": len(other_dset[1]),
            "num_scaffolds": len(other_dset[2]),
        }
        counts_df_entries.append(counts_df_entry)
    counts_df = pd.DataFrame(counts_df_entries)
    print(counts_df)
    counts_df.to_csv("figs/dset/cfm_counts.csv",index=False)

    overlap_df_entries = []
    for other_dset in other_dsets:
        overlap_df_entry = {
            "dset": other_dset[0],
            "num_mols": len(cfm_inchikey_ses&other_dset[1]),
            "num_scaffolds": len(cfm_scaffolds&other_dset[2]),
        }
        overlap_df_entries.append(overlap_df_entry)
    overlap_df = pd.DataFrame(overlap_df_entries)
    print(overlap_df)
    overlap_df.to_csv("figs/dset/cfm_overlap.csv",index=False)
