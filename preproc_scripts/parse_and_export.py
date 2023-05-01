from collections import Counter
import numpy as np
import os
import pandas as pd
import time
from tqdm import tqdm
import pickle
import argparse
import ast
import glob
import importlib
import re

from massformer.data_utils import rdkit_import, seq_apply, par_apply, seq_apply_series, par_apply_series, par_apply_df_rows, ELEMENT_LIST
from massformer.misc_utils import booltype

key_dict = {
    "Precursor_type": "prec_type",
    "Spectrum_type": "spec_type",
    "PrecursorMZ": "prec_mz",
    "Instrument_type": "inst_type",
    "Collision_energy": "col_energy",
    "Ion_mode": "ion_mode",
    "Ionization": "ion_type",
    "ID": "spec_id",
    "Collision_gas": "col_gas",
    "Pressure": "pressure",
    "Num peaks": "num_peaks",
    "MW": "mw",
    "ExactMass": "exact_mass",
    "CASNO": "cas_num",
    "NISTNO": "nist_num",
    "Name": "name",
    "MS": "peaks",
    "SMILES": "smiles",
    "Rating": "rating",
    "Frag_mode": "frag_mode",
    "Instrument": "inst",
    "RI": "ri"
}


def extract_info_from_comments(comments, key):

    start_idx = comments.find(key)
    if start_idx == -1:
        return None
    start_idx += len(key) + 1  # +1 is for =
    end_idx = start_idx + 1
    cur_char = comments[end_idx]
    while cur_char != "\"":
        end_idx += 1
        cur_char = comments[end_idx]
    value = comments[start_idx:end_idx]
    return value


"""
Convert data from MSMS database format to pandas dataframe with JSON
No type conversions or filtering: all of that is done downstream
"""


def preproc_msp(msp_fp, keys, num_entries):
    """ """

    with open(msp_fp) as f:
        raw_data_lines = f.readlines()
    # raw_data_lines = raw_data_lines[:1000]
    raw_data_list = []
    raw_data_item = {key: None for key in keys}
    read_ms = False
    for raw_l in tqdm(raw_data_lines):
        if num_entries > -1 and len(raw_data_list) == num_entries:
            break
        raw_l = raw_l.replace('\n', '')
        if raw_l == '':
            raw_data_list.append(raw_data_item.copy())
            raw_data_item = {key: None for key in keys}
            read_ms = False
        elif read_ms:
            raw_data_item['MS'] = raw_data_item['MS'] + raw_l + '\n'
        else:
            if "RI:" in raw_l:
                raw_l_split = raw_l.split(':')
            else:
                raw_l_split = raw_l.split(': ')
            assert len(raw_l_split) >= 2
            key = raw_l_split[0]
            if key == "Num peaks" or key == "Num Peaks":
                assert len(raw_l_split) == 2, raw_l_split
                value = raw_l_split[1]
                raw_data_item['Num peaks'] = int(value)
                raw_data_item['MS'] = ''
                read_ms = True
            elif key == "Comments":
                comments = ": ".join(raw_l_split[1:])
                smiles = extract_info_from_comments(
                    comments, "computed SMILES")
                rating = extract_info_from_comments(comments, "MoNA Rating")
                frag_mode = extract_info_from_comments(
                    comments, "fragmentation mode")
                if not (smiles is None):
                    raw_data_item["SMILES"] = smiles
                if not (rating is None):
                    raw_data_item["Rating"] = rating
                if not (frag_mode is None):
                    raw_data_item["Frag_mode"] = frag_mode
            elif key in keys:
                value = raw_l_split[1]
                raw_data_item[key] = value
    msp_df = pd.DataFrame(raw_data_list)
    # drop all-NaN rows
    msp_df = msp_df.dropna(axis=0, how="all")
    return msp_df


def preproc_nist_mol(mol_dp):
    """ read in all .MOL files and return a df """

    mol_fp_list = glob.glob(os.path.join(mol_dp, "*.MOL"))

    def proc_mol_file(mol_fp):
        modules = rdkit_import(
            "rdkit.Chem",
            "rdkit.Chem.rdinchi",
            "rdkit.Chem.AllChem")
        Chem = modules[0]
        rdinchi = modules[1]
        AllChem = modules[2]
        mol_fn = os.path.basename(os.path.normpath(mol_fp))
        spec_id = mol_fn.lstrip("ID").rstrip(".MOL")
        mol = Chem.MolFromMolFile(mol_fp, sanitize=True)
        if not (mol is None):
            smiles = Chem.MolToSmiles(mol)
        else:
            smiles = None
        entry = dict(
            spec_id=spec_id,
            smiles=smiles
        )
        return entry
    mol_df_entries = par_apply(mol_fp_list, proc_mol_file)
    mol_df = pd.DataFrame(mol_df_entries)
    return mol_df


def merge_and_check(msp_df, mol_df, rename_dict):

    # get rid of the columns that you don't care about
    msp_bad_cols = set(msp_df.columns) - set(rename_dict.keys())
    msp_df = msp_df.drop(columns=msp_bad_cols)
    # rename to be consistent
    msp_df = msp_df.rename(columns=rename_dict)
    if mol_df is None:
        assert not msp_df["smiles"].isna().all()
        assert msp_df["spec_id"].isna().all()
        msp_df.loc[:, "spec_id"] = np.arange(msp_df.shape[0])
        spec_df = msp_df
    else:
        assert msp_df["smiles"].isna().all()
        assert not msp_df["spec_id"].isna().all()
        # merge with mol on spec_id
        msp_df = msp_df.drop(columns=["smiles"])
        spec_df = pd.merge(msp_df, mol_df, how="inner", on="spec_id")
    print(spec_df.isna().sum())
    spec_df = spec_df.reset_index(drop=True)
    return spec_df


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--msp_file', type=str, required=True)
    parser.add_argument('--mol_dir', type=str, required=False)
    parser.add_argument('--output_name', type=str, required=True)
    parser.add_argument('--raw_data_dp', type=str, default='data/raw')
    parser.add_argument('--output_dp', type=str, default='data/df')
    parser.add_argument('--num_entries', type=int, default=-1)
    parser.add_argument(
        '--output_type',
        type=str,
        default="json",
        choices=[
            "json",
            "csv"])
    args = parser.parse_args()

    msp_fp = os.path.join(args.raw_data_dp, args.msp_file)
    assert os.path.isfile(msp_fp)
    if args.mol_dir:
        mol_dp = os.path.join(args.raw_data_dp, args.mol_dir)
        assert os.path.isdir(mol_dp)
    else:
        mol_dp = None

    os.makedirs(args.output_dp, exist_ok=True)

    msp_df = preproc_msp(msp_fp, key_dict.keys(), args.num_entries)

    if mol_dp:
        mol_df = preproc_nist_mol(mol_dp)
    else:
        mol_df = None
    spec_df = merge_and_check(msp_df, mol_df, key_dict)
    # save files
    spec_df_fp = os.path.join(args.output_dp,
                              f"{args.output_name}.{args.output_type}")
    if args.output_type == "json":
        spec_df.to_json(spec_df_fp)
    else:
        assert args.output_type == "csv"
        spec_df.to_csv(spec_df_fp, index=False)
