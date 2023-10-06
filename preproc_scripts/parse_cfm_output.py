import pandas as pd
import numpy as np
import os
import glob
from tqdm import tqdm
import argparse

from massformer.misc_utils import booltype
from massformer.data_utils import mol_from_smiles,mol_to_mol_weight,H_MASS,NA_MASS,N_MASS


ENERGY_TO_CE = {
	0: 10,
	1: 20,
	2: 40
}

PREC_TYPE_TO_MASS_DIFF = {
	"[M+H]+": H_MASS,
	"[M-H]-": -H_MASS,
	"[M+Na]+": NA_MASS,
	"[M+NH4]+": N_MASS+4*H_MASS
}


def parse_cfm_split(split_fp,rb):
	cfm_spec_ds = []
	with open(split_fp,"r") as split_file:
		split_lines = split_file.readlines()
	if len(split_lines) == 0:
		return cfm_spec_ds
	line_idx = 0
	in_spec = in_peaks = False
	# import pdb; pdb.set_trace()
	while line_idx < len(split_lines):
		line = split_lines[line_idx].strip()
		if in_spec:
			if ">>> END SPEC" in line:
				in_spec = in_peaks = False
				# cleanup
				_cfm_spec_ds = []
				for ce in ENERGY_TO_CE.values():
					_cfm_spec_d = cfm_spec_d.copy()
					_cfm_spec_d["peaks"] = [(peak[1],peak[2]) for peak in cfm_spec_d["peaks"] if peak[0] == ce]
					_cfm_spec_d["ace"] = ce
					_cfm_spec_ds.append(_cfm_spec_d)
				cfm_spec_ds.extend(_cfm_spec_ds)
			elif not in_peaks:
				assert rb
				# keep skipping until END SPEC
				pass
			elif len(line) > 0 and line[0] == "#":
				assert in_peaks
				# comment line
				if line.startswith("#In-silico"):
					if "[M+H]+" in line:
						cfm_spec_d["prec_type"] = "[M+H]+"
						cfm_spec_d["ion_mode"] = "P"
					elif "[M-H]-" in line:
						cfm_spec_d["prec_type"] = "[M-H]-"
						cfm_spec_d["ion_mode"] = "N"
					elif "[M+Na]+" in line:
						assert rb
						cfm_spec_d["prec_type"] = "[M+Na]+"
						cfm_spec_d["ion_mode"] = "P"
					elif "[M+NH4]+" in line:
						assert rb
						cfm_spec_d["prec_type"] = "[M+NH4]+"
						cfm_spec_d["ion_mode"] = "P"
					else:
						raise ValueError("invalid prec_type")
				elif line.startswith("#Formula"):
					assert not rb
					cfm_spec_d["formula"] = line[len("#Formula="):]
				elif line.startswith("#PMass"):
					assert not rb
					cfm_spec_d["prec_mz"] = float(line[len("#PMass="):])
			else:
				assert in_peaks
				if line.startswith("energy"):
					energy_level = int(line[-1])
					ce = ENERGY_TO_CE[energy_level]
				elif line != "":
					assert ce is not None
					# this is a peak entry
					mz, ints = line.split(" ")[:2]
					mz = float(mz)
					ints = float(ints)
					cfm_spec_d["peaks"].append((ce,mz,ints))
				else:
					assert rb
					# just skip this line
					in_peaks = False
		elif ">>> START SPEC" in line:
			in_spec = in_peaks = True
			# skip the next line (metadata)
			line_idx += 1
			# init
			cfm_spec_d = {}
			cfm_spec_d["peaks"] = []
			line_elems = line.split(" ")
			cfm_spec_d["mol_id"] = int(line_elems[3])
			ce = None
			if rb:
				# formulae and prec_mz not provided
				cfm_spec_d["formula"] = None
				cfm_spec_d["prec_mz"] = None
		else:
			# something went wrong!
			import pdb; pdb.set_trace()
		line_idx += 1
	return cfm_spec_ds

def spec_normalize(peaks):
	if len(peaks) == 0:
		return peaks
	ints_max = max(peak[1] for peak in peaks)
	norm_peaks = [(peak[0],1000.*(peak[1]/ints_max)) for peak in peaks]
	return norm_peaks

if __name__ == "__main__":

	parser = argparse.ArgumentParser()
	parser.add_argument("--cfm_dp",type=str,default="data/cfm")
	parser.add_argument("--spec_dir",type=str,default="spec_splits/all") # spec_splits_rb/all
	parser.add_argument("--smiles_file",type=str,default="all_smiles.txt")
	parser.add_argument("--output_file",type=str,default="all_spec_df.pkl") # all_rb_spec_df.pkl
	parser.add_argument("--rb",type=booltype,default=False)
	args = parser.parse_args()

	cfm_spec_dp = os.path.join(args.cfm_dp,args.spec_dir)
	cfm_smiles_fp = os.path.join(args.cfm_dp,args.smiles_file)

	cfm_smiles_df = pd.read_csv(cfm_smiles_fp,sep=" ",names=["mol_id","smiles"])
	cfm_spec_dps = sorted(glob.glob(os.path.join(cfm_spec_dp,"spec_*")))

	spec_entries = []
	for spec_idx, spec_dp in tqdm(enumerate(cfm_spec_dps),total=len(cfm_spec_dps),desc="> split"):
		spec_dp = os.path.abspath(spec_dp)
		split_fp = os.path.join(spec_dp,"split.txt")
		split_spec_ds = parse_cfm_split(split_fp,args.rb)
		spec_entries.extend(split_spec_ds)

	cfm_spec_df = pd.DataFrame(spec_entries)
	# import pdb; pdb.set_trace()

	# drop entries with no peaks
	cfm_spec_df = cfm_spec_df[cfm_spec_df["peaks"].apply(len)>0]

	# some mols might be missing spectra
	print(cfm_smiles_df["mol_id"].nunique()-cfm_spec_df["mol_id"].nunique())
	assert cfm_spec_df["mol_id"].nunique() <= cfm_smiles_df["mol_id"].nunique()
	assert cfm_spec_df["mol_id"].isin(cfm_smiles_df["mol_id"]).all()
	assert cfm_spec_df.drop_duplicates(subset=["mol_id","ace","prec_type"]).shape[0] == cfm_spec_df.shape[0]

	# normalize cfm_spec_df to have max intensity of 1000
	cfm_spec_df.loc[:,"peaks"] = cfm_spec_df["peaks"].apply(spec_normalize)
	assert not cfm_spec_df["prec_type"].isna().any()

	# infer prec_mz (could also do formula, but not really important)
	no_prec_mz_df = cfm_spec_df[cfm_spec_df["prec_mz"].isna()][["mol_id","prec_type"]].drop_duplicates()
	# no_formula_df = cfm_spec_df[cfm_spec_df["formula"].isna()][["mol_id"]].drop_duplicates()
	print(f"> no prec_mz: {no_prec_mz_df.shape[0]}")
	if no_prec_mz_df.shape[0] > 0:
		no_prec_mz_df = no_prec_mz_df.merge(cfm_smiles_df,on="mol_id",how="inner")
		no_prec_mz_df.loc[:,"mol"] = no_prec_mz_df["smiles"].apply(mol_from_smiles)
		assert not no_prec_mz_df["mol"].isna().any()
		no_prec_mz_df = no_prec_mz_df.drop(columns=["smiles"])
		def compute_prec_mz(mol,prec_type):
			prec_mz = mol_to_mol_weight(mol,exact=True)
			prec_mz += PREC_TYPE_TO_MASS_DIFF[prec_type]
			return prec_mz
		no_prec_mz_df.loc[:,"prec_mz"] = no_prec_mz_df.apply(lambda row: compute_prec_mz(row["mol"],row["prec_type"]),axis=1)
		cfm_spec_df = cfm_spec_df.merge(no_prec_mz_df,on=["mol_id","prec_type"],how="left")
		cfm_spec_df.loc[:,"prec_mz"] = cfm_spec_df.apply(lambda row: row["prec_mz_y"] if (row["prec_mz_x"] is None) else row["prec_mz_x"],axis=1)
		cfm_spec_df = cfm_spec_df.drop(columns=["prec_mz_x","prec_mz_y","mol"])
		assert not cfm_spec_df["prec_mz"].isna().any()

	# save data
	print(cfm_spec_df)
	cfm_spec_df.to_pickle(os.path.join(args.cfm_dp,args.output_file))
