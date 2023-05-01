import pandas as pd
import argparse
import os
import glob
import numpy as np

import massformer.data_utils as data_utils
from massformer.data_utils import par_apply_series, check_mol_props
from massformer.misc_utils import none_or_nan


def common_filter(spec_df, mol_df, cand_df):

    query_mol_id = set(
        spec_df["mol_id"]) & set(
        mol_df["mol_id"]) & set(
            cand_df["query_mol_id"])
    cand_mol_id = set(cand_df[cand_df["query_mol_id"].isin(
        query_mol_id)]["candidate_mol_id"])
    spec_df = spec_df[spec_df["mol_id"].isin(
        query_mol_id)].reset_index(drop=True)
    mol_df = mol_df[mol_df["mol_id"].isin(
        query_mol_id | cand_mol_id)].reset_index(drop=True)
    cand_df = cand_df[cand_df["query_mol_id"].isin(
        query_mol_id) & cand_df["candidate_mol_id"].isin(cand_mol_id)].reset_index(drop=True)
    return spec_df, mol_df, cand_df


def main(args):

    casmi_dp = os.path.join(args.raw_dp, args.casmi_input_dir)
    spec_dp = os.path.join(casmi_dp, "spectra")
    cand_dp = os.path.join(casmi_dp, "candidates")
    sol_fp = os.path.join(casmi_dp, "challenge_solutions.csv")

    id_offset = len("Challenge-")

    spec_peakses = []
    spec_ids = []
    for spec_file in sorted(glob.glob(os.path.join(spec_dp, "*.txt"))):
        spec_file = os.path.basename(spec_file)
        spec_fp = os.path.join(spec_dp, spec_file)
        spec_id = int(spec_file[id_offset:id_offset + 3])
        spec_df = pd.read_csv(spec_fp, sep="\t", names=["mz", "ints"])
        spec_peaks = list(spec_df.to_records(index=False))
        spec_peakses.append(spec_peaks)
        spec_ids.append(spec_id)
    peak_df = pd.DataFrame({"spec_id": spec_ids, "peaks": spec_peakses})

    sol_df = pd.read_csv(sol_fp)
    sol_dict = {
        "ChallengeName": "spec_id",
        "PRECURSOR_MZ": "prec_mz",
        "ION_MODE": "ion_mode",
        "SMILES": "smiles"
    }
    ion_mode_dict = {
        " POSITIVE": "P",
        " NEGATIVE": "N"
    }
    sol_df = sol_df[list(sol_dict.keys())]
    sol_df = sol_df.rename(columns=sol_dict)
    sol_df.loc[:, "spec_id"] = sol_df["spec_id"].str[id_offset:id_offset +
                                                     3].astype(int)
    sol_df.loc[:, "ion_mode"] = sol_df["ion_mode"].map(ion_mode_dict)
    assert set(spec_ids) == set(sol_df["spec_id"])

    cand_dict = {
        "SMILES": "smiles"
    }
    cand_fps = [
        os.path.join(
            cand_dp,
            f"Challenge-{spec_id:03d}.csv") for spec_id in spec_ids]
    if args.num_entries > -1:
        cand_fps = cand_fps[:args.num_entries]
    cand_smileses = []
    query_spec_ids = []
    for cand_idx, cand_fp in enumerate(cand_fps):
        cand_df = pd.read_csv(cand_fp)
        cand_df = cand_df[list(cand_dict.keys())]
        cand_df = cand_df.rename(columns=cand_dict)
        # add the candidates
        cand_smileses.extend(cand_df["smiles"].tolist())
        query_spec_ids.extend([spec_ids[cand_idx]
                              for i in range(cand_df.shape[0])])
        # add the actual match
        cand_smileses.append(
            sol_df[sol_df["spec_id"] == spec_ids[cand_idx]]["smiles"].item())
        query_spec_ids.append(spec_ids[cand_idx])
    cand_df = pd.DataFrame(
        {"candidate_smiles": cand_smileses, "query_spec_id": query_spec_ids})
    # make sure there are no duplicates
    cand_df = cand_df.drop_duplicates()
    un_smileses = sorted(cand_df["candidate_smiles"].unique().tolist())
    smiles_to_id = {smiles: idx for idx, smiles in enumerate(un_smileses)}
    cand_df.loc[:, "candidate_mol_id"] = cand_df["candidate_smiles"].map(
        smiles_to_id)
    cand_df = cand_df.drop(columns=["candidate_smiles"])

    mol_df = pd.DataFrame({"smiles": un_smileses})
    mol_df.loc[:, "mol_id"] = mol_df["smiles"].map(smiles_to_id)
    mol_df.loc[:, "mol"] = par_apply_series(
        mol_df["smiles"], data_utils.mol_from_smiles)
    print(mol_df.isna().sum())  # hopefully none
    mol_df = mol_df.dropna()
    mol_df = check_mol_props(mol_df)  # this might drop stuff

    # add all the mol stuff
    mol_df.loc[:, "smiles_iso"] = mol_df["smiles"].copy(deep=True)
    mol_df.loc[:, "smiles"] = par_apply_series(
        mol_df["mol"], data_utils.mol_to_smiles)
    mol_df.loc[:, "inchikey_s"] = par_apply_series(
        mol_df["mol"], data_utils.mol_to_inchikey_s)
    mol_df.loc[:, "scaffold"] = par_apply_series(
        mol_df["mol"], data_utils.get_murcko_scaffold)
    mol_df.loc[:, "formula"] = par_apply_series(
        mol_df["mol"], data_utils.mol_to_formula)
    mol_df.loc[:, "inchi"] = par_apply_series(
        mol_df["mol"], data_utils.mol_to_inchi)
    mol_df.loc[:, "mw"] = par_apply_series(
        mol_df["mol"], lambda mol: data_utils.mol_to_mol_weight(mol, exact=False))
    mol_df.loc[:, "exact_mw"] = par_apply_series(
        mol_df["mol"], lambda mol: data_utils.mol_to_mol_weight(mol, exact=True))
    print(
        "mol_df nunique smiles_iso/smiles/inchikey_stuffs:",
        mol_df["smiles_iso"].nunique(),
        mol_df["smiles"].nunique(),
        mol_df["inchikey_s"].nunique())
    mol_df = mol_df.astype({"mol_id": int})

    # this fails if num_entries != -1, or if we filter any of the solution
    # molecules
    print("sol_df subset mol_df (when including isomer info):",
          (sol_df["smiles"].isin(mol_df["smiles_iso"])).all())
    # spec_df = sol_df[sol_df["smiles"].isin(mol_df["smiles"])]
    spec_df = sol_df.copy(deep=True)
    spec_df.loc[:, "mol_id"] = spec_df["smiles"].map(smiles_to_id)
    spec_df = spec_df.drop(columns=["smiles"])
    spec_df = spec_df.merge(peak_df, how="inner", on="spec_id")
    # add stuff that applies to all CASMI spectra
    prec_type_dict = {
        "P": "[M+H]+",
        "N": "[M-H]-"
    }
    spec_df.loc[:, "prec_type"] = spec_df["ion_mode"].map(prec_type_dict)
    spec_df.loc[:, "ace"] = np.nan  # this is determined at test time
    spec_df.loc[:, "nce"] = np.nan  # this is determined at test time
    spec_df.loc[:, "inst_type"] = "FT"
    spec_df.loc[:, "frag_mode"] = "HCD"
    spec_df.loc[:, "res"] = 4
    spec_df.loc[:, "spec_type"] = "MS2"
    spec_df.loc[:, "group_id"] = spec_df["spec_id"].copy()
    spec_df = spec_df.astype({"spec_id": int, "mol_id": int, "group_id": int})

    # remap cand_df based on query mol_id
    cand_df = cand_df.merge(spec_df[["spec_id", "mol_id"]].rename(columns={
                            "spec_id": "query_spec_id", "mol_id": "query_mol_id"}), on="query_spec_id", how="inner")
    cand_df = cand_df.drop(columns=["query_spec_id"])

    # make sure the datasets are internally consistent
    spec_df, mol_df, cand_df = common_filter(spec_df, mol_df, cand_df)

    # save the data
    os.makedirs(
        os.path.join(
            args.proc_dp,
            args.casmi_output_dir),
        exist_ok=True)
    spec_df_fp = os.path.join(
        args.proc_dp,
        args.casmi_output_dir,
        "spec_df.pkl")
    mol_df_fp = os.path.join(args.proc_dp, args.casmi_output_dir, "mol_df.pkl")
    cand_df_fp = os.path.join(
        args.proc_dp,
        args.casmi_output_dir,
        "cand_df.pkl")
    spec_df.to_pickle(spec_df_fp)
    mol_df.to_pickle(mol_df_fp)
    cand_df.to_pickle(cand_df_fp)

    # export smiles for CFM
    export_mol_df = mol_df[["mol_id", "smiles",]]
    export_mol_df.to_csv(
        os.path.join(
            args.proc_dp,
            args.casmi_output_dir,
            "casmi_smiles.txt"),
        sep=" ",
        header=False,
        index=False)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_dp", type=str, default="data/raw")
    parser.add_argument("--proc_dp", type=str, default="data/proc")
    parser.add_argument("--casmi_input_dir", type=str, default="casmi_2016")
    parser.add_argument("--casmi_output_dir", type=str, default="casmi_2016")
    parser.add_argument("--num_entries", type=int, default=-1)
    args = parser.parse_args()
    main(args)
