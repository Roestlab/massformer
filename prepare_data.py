import numpy as np
import os
import pandas as pd
import selfies as sf
import json

import data_utils
from data_utils import par_apply_series, par_apply_df_rows, seq_apply_series, seq_apply_df_rows
from misc_utils import timeout, list_str2float, booltype, tqdm_joblib, TimeoutError

# load the nist, mb na data

def load_df(df_dp,num_entries):
	
	nist_fp = os.path.join(df_dp,"nist_df.json")
	mb_na_fp = os.path.join(df_dp,"mb_na_df.json")
	dfs = []
	nist_df = pd.read_json(nist_fp)
	nist_df["dset"] = "nist"
	dfs.append(nist_df)
	mb_na_df = pd.read_json(mb_na_fp)
	mb_na_df["dset"] = "mb_na"
	dfs.append(mb_na_df)
	if num_entries > 0:
		dfs = [df.sample(n=num_entries,replace=False,random_state=420) for df in dfs]
	if len(dfs) > 1:
		all_df = pd.concat(dfs,ignore_index=True)
	else:
		all_df = dfs[0]
	all_df = all_df.reset_index(drop=True)

	return all_df

# preprocesses spectra and molecules
# remove compounds/spectra with invalid smiles
# remove compounds/spectra with no bonds or invalid atoms
# this filtering is done here because it would be expensive to do at the beginning of training

def preprocess_spec(spec_df):

	# convert smiles to mol and back (removing stereochemical information)
	spec_df.loc[:,"mol"] = par_apply_series(spec_df["smiles"],data_utils.mol_from_smiles)
	spec_df.loc[:,"smiles"] = par_apply_series(spec_df["mol"],data_utils.mol_to_smiles)
	spec_df = spec_df.dropna(subset=["mol","smiles"])
	# check atom types and number of bonds
	valid_atoms = par_apply_series(spec_df["mol"],data_utils.check_atoms)
	valid_num_bonds = par_apply_series(spec_df["mol"],data_utils.check_num_bonds)
	spec_df = spec_df[valid_atoms & valid_num_bonds]

	# enumerate smiles to create molecule ids
	smiles_set = set(spec_df["smiles"])
	print("num_smiles", len(smiles_set))
	smiles_to_mid = {smiles:i for i,smiles in enumerate(sorted(smiles_set))}
	spec_df.loc[:,"mol_id"] = spec_df["smiles"].replace(smiles_to_mid)

	# extract peak info (still represented as str)
	spec_df.loc[:,"peaks"] = par_apply_series(spec_df["peaks"],data_utils.parse_peaks_str)
	# get mz resolution
	spec_df.loc[:,"res"] = par_apply_series(spec_df["peaks"],data_utils.get_res)
	# standardize the instrument type and frag_mode
	inst_type, frag_mode = seq_apply_df_rows(spec_df,data_utils.parse_inst_info)
	spec_df.loc[:,"inst_type"] = inst_type
	spec_df.loc[:,"frag_mode"] = frag_mode
	# standardize ce
	spec_df.loc[:,"ace"] = par_apply_series(spec_df["col_energy"],data_utils.parse_ace_str)
	spec_df.loc[:,"nce"] = par_apply_series(spec_df["col_energy"],data_utils.parse_nce_str)
	spec_df = spec_df.drop(columns=["col_energy"])
	# standardise prec_type
	spec_df.loc[:,"prec_type"] = par_apply_series(spec_df["prec_type"],data_utils.parse_prec_type_str)
	# convert prec_mz
	spec_df.loc[:,"prec_mz"] = pd.to_numeric(spec_df["prec_mz"],errors="coerce")
	# convert ion_mode
	spec_df.loc[:,"ion_mode"] = par_apply_series(spec_df["ion_mode"],data_utils.parse_ion_mode_str)
	# convert peaks to float
	spec_df.loc[:,"peaks"] = par_apply_series(spec_df["peaks"],data_utils.convert_peaks_to_float)

	# remove columns from spec_df
	spec_df = spec_df[["spectrum_id","mol_id","prec_type","inst_type","frag_mode","spec_type","ion_mode","dset","col_gas","res","ace","nce","prec_mz","peaks"]]

	# get mol df
	mol_df = pd.DataFrame(zip(sorted(smiles_set),list(range(len(smiles_set)))),columns=["smiles","mol_id"])
	mol_df.loc[:,"mol"] = par_apply_series(mol_df["smiles"],data_utils.mol_from_smiles)
	mol_df.loc[:,"inchikey_s"] = par_apply_series(mol_df["mol"],data_utils.mol_to_inchikey_s)
	mol_df.loc[:,"scaffold"] = par_apply_series(mol_df["mol"],data_utils.get_murcko_scaffold)
	mol_df.loc[:,"formula"] = par_apply_series(mol_df["mol"],data_utils.mol_to_formula)
	mol_df.loc[:,"inchi"] = par_apply_series(mol_df["mol"],data_utils.mol_to_inchi)
	mol_df.loc[:,"mw"] = par_apply_series(mol_df["mol"],lambda mol: data_utils.mol_to_mol_weight(mol,exact=False))
	mol_df.loc[:,"exact_mw"] = par_apply_series(mol_df["mol"],lambda mol: data_utils.mol_to_mol_weight(mol,exact=True))

	# reset indices
	spec_df = spec_df.reset_index(drop=True)
	mol_df = mol_df.reset_index(drop=True)

	return spec_df, mol_df

# get the sets of smiles and selfies tokens

def compute_tokens(mol_df):

	# get the canonical smiles and selfies
	can_smiles = mol_df["smiles"]
	can_selfies = par_apply_series(can_smiles,sf.encoder)

	# get the character sets
	smiles_chars = par_apply_series(can_smiles,data_utils.split_smiles)
	smiles_char_set = set(smiles_chars.sum())
	smiles_c2i = {char: i for i,char in enumerate(smiles_char_set)}
	smiles_i2c = {i: char for i,char in enumerate(smiles_char_set)}
	selfies_chars = par_apply_series(can_selfies,data_utils.split_selfies)
	selfies_char_set = set(selfies_chars.sum())
	selfies_c2i = {char: i for i,char in enumerate(selfies_char_set)}
	selfies_i2c = {i: char for i,char in enumerate(selfies_char_set)}

	token_dict = {
		"smiles_c2i": smiles_c2i,
		"smiles_i2c": smiles_i2c,
		"selfies_c2i": selfies_c2i,
		"selfies_i2c": selfies_i2c,	
	}

	return token_dict

if __name__ == "__main__":

	import argparse
	parser = argparse.ArgumentParser()
	parser.add_argument("--df_dp", type=str, default="data/df")
	parser.add_argument("--proc_dp", type=str, default="data/proc")
	parser.add_argument("--num_entries", type=int, default=-1)
	parser.add_argument("--ow_spec_mol", type=booltype, default=False)
	parser.add_argument("--frag_depth", type=int, default=2)
	parser.add_argument("--frag_timeout", type=int, default=5)
	flags = parser.parse_args()

	assert os.path.isdir(flags.proc_dp), flags.proc_dp
	data_dp = flags.proc_dp

	spec_df_fp = os.path.join(data_dp,"spec_df.pkl")
	mol_df_fp = os.path.join(data_dp,"mol_df.pkl")

	if not flags.ow_spec_mol and os.path.isfile(spec_df_fp) and os.path.isfile(mol_df_fp):

		print("> loading previous spec_df, mol_df")
		spec_df = pd.read_pickle(spec_df_fp)
		mol_df = pd.read_pickle(mol_df_fp)

	else:

		print("> creating new spec_df, mol_df")
		assert os.path.isdir(flags.df_dp), flags.df_dp
		all_df = load_df(flags.df_dp,flags.num_entries)

		spec_df, mol_df = preprocess_spec(all_df)

		# get dictionary of tokens
		token_dict = compute_tokens(mol_df)

		# save everything to file
		spec_df.to_pickle(spec_df_fp)
		mol_df.to_pickle(mol_df_fp)
		for k,v in token_dict.items():
			v_fp = os.path.join(data_dp,f"{k}.json")
			with open(v_fp,"w") as v_file:
				json.dump(v,v_file)

	print(spec_df.shape)
	print(spec_df.isna().sum())
	print(mol_df.shape)
	print(mol_df.isna().sum())
