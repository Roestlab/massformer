from glob import glob
import os
import numpy as np
import pandas as pd
import re
from tqdm import tqdm
import argparse

from massformer.casmi_utils import common_filter, load_mw_cand, prepare_casmi_mol_df, prepare_casmi_cand_df, prepare_casmi_spec_df, proc_cand_smiles
from massformer.data_utils import par_apply_series, mol_from_smiles, mol_to_smiles, mol_to_mol_weight, check_mol_props, get_res, H_MASS, O_MASS, NA_MASS, N_MASS, C_MASS


def calculate_total_spec_ints(peaks):
    """ this is just used as a heuristic to select the spectrum to identify """

    total_ints = 0
    for peak in peaks:
        total_ints += peak[1]
    return total_ints


def setup_casmi_spec(casmi_input_dp):

    all_dp = os.path.join(casmi_input_dp,"")
    assert os.path.isdir(all_dp), all_dp

    results_fp = os.path.join(casmi_input_dp,"all_meta_results.csv")
    results_df = pd.read_csv(results_fp,skiprows=1)

    casmi_fps = list(glob(os.path.join(all_dp,"all","*.msp")))

    # keys in this dict are (file,rt,prec_mz)
    # rt is to two decimals, prec_mz is to four 
    casmi_ms_entries = {}

    for casmi_fp in tqdm(casmi_fps,total=len(casmi_fps)):
        with open(casmi_fp,"r") as file:
            lines = file.readlines()
        cur_name = None
        cur_rt = None
        cur_prec_mz = None
        cur_peaks = []
        cur_res = -1
        cur_spec_type = None
        for line in lines:
            line = line.strip()
            if "Name" in line:
                cur_name = line.split(" ")[1].split(".")[0]
            elif line == "":
                assert not (cur_name is None)
                assert not (cur_rt is None)
                assert not (cur_prec_mz is None)
                assert len(cur_peaks) > 0
                assert cur_spec_type == "MS2"
                entry = {
                    "rt": cur_rt,
                    "prec_mz": cur_prec_mz,
                    "spec_type": cur_spec_type,
                    "peaks": cur_peaks,
                    "res": cur_res
                }
                if cur_name in casmi_ms_entries:
                    casmi_ms_entries[cur_name].append(entry)
                else:
                    casmi_ms_entries[cur_name] = [entry]
                cur_name = cur_rt = cur_prec_mz = cur_spec_type = None
                cur_peaks = []
                cur_res = -1
            elif "PRECURSORMZ" in line:
                cur_prec_mz = float(line.split(" ")[1])
            elif "RETENTIONTIME" in line:
                cur_rt = float(line.split(" ")[1])
            elif "Spectrum_type" in line:
                cur_spec_type = line.split(" ")[1]
            elif len(re.findall("\d+\.\d+\t\d+\.\d+",line)) > 0:
                peak = re.findall("\d+\.\d+\t\d+\.\d+",line)[0].split("\t")
                res = get_res([peak])
                mz = float(peak[0])
                ints = float(peak[1])
                cur_peaks.append((mz,ints))
                cur_res = max(cur_res,res)
            else:
                pass

    casmi_rows = []
    for row_idx,row in tqdm(results_df.iterrows(),total=results_df.shape[0]):
        casmi_row = {}
        # get meta information
        casmi_row["casmi_id"] = row["Compound Number"]
        if "pos" in row["File"]:
            casmi_row["ion_mode"] = "P"
        else:
            casmi_row["ion_mode"] = "N"
        casmi_row["priority"] = (row["Priority/Bonus"] == "Priority")
        casmi_row["smiles"] = row["SMILES"]
        casmi_row["prec_type"] = row["Adduct"]
        # get ms information
        name = row["File"]
        prec_mz = row["Precursor m/z (Da)"]
        rt = row["RT [min]"]
        # get the corresponding files
        name_entries = casmi_ms_entries[name]
        # prec_mz must be within 0.005 Da
        # rt must be within 0.1 min
        prec_mz_diff = np.array([np.abs(entry["prec_mz"]-prec_mz) for entry in name_entries])
        rt_diff = np.array([np.abs(entry["rt"]-rt) for entry in name_entries])
        total_ints = np.array([calculate_total_spec_ints(entry["peaks"]) for entry in name_entries])
        # name_entries = [entry for entry in name_entries if np.abs(entry["prec_mz"]-prec_mz) <= 0.005]
        # name_entries = [entry for entry in name_entries if np.abs(entry["rt"]-rt) <= 0.1]
        both_mask = (prec_mz_diff <= 0.005) & (rt_diff <= 0.1)
        if len(both_mask) == 0:
            print("> warning: no spectrum meets criterion")
            # select the one with the minimum RT diff
            rt_diff = (np.max(rt_diff)+1)*(prec_mz_diff > 0.005) + rt_diff*(prec_mz_diff <= 0.005)
            idx = np.argmin(rt_diff)
        else:
            # select the spectrum with maximum total intensity
            total_ints = total_ints*both_mask
            idx = np.argmax(total_ints)
        casmi_entry = name_entries[idx]
        for k,v in casmi_entry.items():
            assert not (k in casmi_row)
            casmi_row[k] = v
        casmi_rows.append(casmi_row)
    casmi_df = pd.DataFrame(casmi_rows)

    # rename formic acid precursor adduct
    casmi_df.loc[:,"prec_type"] = casmi_df["prec_type"].replace("[M+FA-H]-","[M+CH2O2-H]-")

    # add a bunch of dummy information
    casmi_df.loc[:,"ace"] = np.nan # this is determined at test time
    casmi_df.loc[:,"nce"] = np.nan # this is determined at test time
    casmi_df.loc[:,"inst_type"] = "FT"
    casmi_df.loc[:,"frag_mode"] = "HCD"
    casmi_df.loc[:,"spec_id"] = np.arange(casmi_df.shape[0]) # not sure what this is used for
    casmi_df.loc[:,"mol_id"] = np.arange(casmi_df.shape[0]) # this is only used for candidate sampling
    casmi_df.loc[:,"group_id"] = np.arange(casmi_df.shape[0]) # this will be used for ce grouping

    print(casmi_df.columns)
    print(casmi_df.isna().sum())

    return casmi_df


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--proc_dp", type=str, default="data/proc")
    parser.add_argument("--raw_dp", type=str, default="data/raw")
    parser.add_argument("--casmi_input_dir", type=str, default="casmi_2022")
    parser.add_argument("--casmi_output_dir", type=str, default="casmi_2022")
    parser.add_argument("--cid_smiles_file", type=str, default="cid_smiles.tsv")
    args = parser.parse_args()

    tqdm.pandas()

    casmi_input_dp = os.path.join(args.raw_dp, args.casmi_input_dir)
    casmi_output_dp = os.path.join(args.proc_dp, args.casmi_output_dir)
    cid_smiles_fp = os.path.join(args.raw_dp, args.cid_smiles_file)
    os.makedirs(casmi_output_dp,exist_ok=True)
    
    spec_fp = os.path.join(casmi_output_dp,"spec_df_1.pkl")
    if not os.path.isfile(spec_fp):
        spec_df = setup_casmi_spec(casmi_input_dp)
        spec_df.to_pickle(spec_fp)
    else:
        spec_df = pd.read_pickle(spec_fp)

    # filtering for precursor adduct here
    mw_diff = {
        "[M+H]+": H_MASS,
        '[M+Na]+': NA_MASS,
    }
    prec_type_mask = spec_df["prec_type"].isin(list(mw_diff.keys()))
    print(f"> number with modelled precursor types: {prec_type_mask.sum()} / {prec_type_mask.shape[0]}")
    spec_df = spec_df[prec_type_mask]

    # reformat smiles
    def smiles_to_smiles(smiles):
        mol = mol_from_smiles(smiles)
        smiles = mol_to_smiles(mol)
        return smiles
    spec_df.loc[:,"smiles"] = spec_df["smiles"].progress_apply(smiles_to_smiles)

    # infer mw from prec_mz and prec_type
    def compute_mw_from_prec_mz(row):
        prec_type = row["prec_type"]
        prec_mz = row["prec_mz"]
        mw = prec_mz - mw_diff[prec_type]
        return mw
    def compute_mw_from_smiles(smiles):
        mol = mol_from_smiles(smiles)
        mw = mol_to_mol_weight(mol,exact=True)
        return mw
    pred_mw = spec_df.progress_apply(compute_mw_from_prec_mz,axis=1)
    exact_mw = spec_df["smiles"].progress_apply(compute_mw_from_smiles)
    print("> mw calculation difference")
    print((pred_mw-exact_mw).abs().describe())

    # create mol_df
    mol_df = pd.DataFrame(
        {
            "mol_id": spec_df["mol_id"].copy(),
            "smiles": spec_df["smiles"].copy(),
            "exact_mw": exact_mw,
        }
    )

    cand_fp_1 = os.path.join(casmi_output_dp,"cand_df_1.pkl")
    if not os.path.isfile(cand_fp_1):
        cand_df = load_mw_cand(
            casmi_output_dp,
            mol_df,
            cid_smiles_fp,
            retmax=10000,
            weight_tol=10e-6
        )
        cand_df.to_pickle(cand_fp_1)
    else:
        cand_df = pd.read_pickle(cand_fp_1)

    # convert to casmi format
    # casmi_mol_df: dataframe of both candidate and query molecules
    # casmi_cand_df: dataframe that maps each query spec_id to the mol_id of the candidates

    # process candidates
    cand_fp_2 = os.path.join(casmi_output_dp,"cand_df_2.pkl")
    if not os.path.isfile(cand_fp_2):
        cand_df = cand_df.drop(columns=["query_mol_id","candidate_cid"])
        # check if smiles can be parsed correctly into different forms
        # still isomeric at this point
        cand_df.loc[:,"clean"] = par_apply_series(cand_df["candidate_smiles"],proc_cand_smiles)
        cand_df = cand_df.dropna().drop(columns=["clean"])
        cand_df.to_pickle(cand_fp_2)
    else:
        cand_df = pd.read_pickle(cand_fp_2)

    # remove query smiles candidates, then add them back
    cand_df = cand_df[cand_df["candidate_smiles"]!=cand_df["query_smiles"]]
    query_smiles = cand_df["query_smiles"].unique()
    cand_df = pd.concat([cand_df,pd.DataFrame({"query_smiles":query_smiles,"candidate_smiles":query_smiles})],axis=0)

    casmi_mol_df = prepare_casmi_mol_df(mol_df,cand_df,casmi_output_dp)

    casmi_spec_df = prepare_casmi_spec_df(spec_df,mol_df,casmi_mol_df,casmi_output_dp)

    casmi_cand_df = prepare_casmi_cand_df(cand_df,casmi_mol_df,casmi_output_dp)

    # make sure the datasets are internally consistent
    casmi_spec_df, casmi_mol_df, casmi_cand_df = common_filter(casmi_spec_df,casmi_mol_df,casmi_cand_df)

    # save dfs
    casmi_spec_df.to_pickle(os.path.join(casmi_output_dp,"spec_df.pkl"))
    casmi_mol_df.to_pickle(os.path.join(casmi_output_dp,"mol_df.pkl"))
    casmi_cand_df.to_pickle(os.path.join(casmi_output_dp,"cand_df.pkl"))

    # export smiles for CFM
    export_mol_df = casmi_mol_df[["mol_id","smiles"]]
    export_mol_df.to_csv(os.path.join(casmi_output_dp,"casmi_smiles.txt"),sep=" ",header=False,index=False)
