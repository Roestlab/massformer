import importlib
import re
import pandas as pd
import numpy as np
import joblib
from tqdm import tqdm
from pprint import pformat, pprint

from misc_utils import np_temp_seed, none_or_nan


ELEMENT_LIST = ['H', 'C',  'O', 'N', 'P', 'S', 'Cl']

TWO_LETTER_TOKEN_NAMES = [
		'Al', 'Ce', 'Co', 'Ge', 'Gd', 'Cs', 'Th', 'Cd', 'As', 'Na', 'Nb', 'Li',
		'Ni', 'Se', 'Sc', 'Sb', 'Sn', 'Hf', 'Hg', 'Si', 'Be', 'Cl', 'Rb', 'Fe',
		'Bi', 'Br', 'Ag', 'Ru', 'Zn', 'Te', 'Mo', 'Pt', 'Mn', 'Os', 'Tl', 'In',
		'Cu', 'Mg', 'Ti', 'Pb', 'Re', 'Pd', 'Ir', 'Rh', 'Zr', 'Cr', '@@', 'se',
		'si', 'te'
]

LC_TWO_LETTER_MAP = {
		"se": "Se", "si": "Si", "te": "Te"
}


def rdkit_import(*module_strs):

	RDLogger = importlib.import_module("rdkit.RDLogger")
	RDLogger.DisableLog('rdApp.*')
	modules = []
	for module_str in module_strs:
		modules.append(importlib.import_module(module_str))
	return tuple(modules)

def normalize_ints(ints):
	try:
		total_ints = sum(ints)
	except:
		import pdb; pdb.set_trace()
	ints = [ints[i] / total_ints for i in range(len(ints))]
	return ints

def randomize_smiles(smiles, rseed, isomeric=False, kekule=False):
	"""Perform a randomization of a SMILES string must be RDKit sanitizable"""
	if rseed == -1:
		return smiles
	modules = rdkit_import("rdkit.Chem")
	Chem = modules[0]
	m = Chem.MolFromSmiles(smiles)
	assert not (m is None)
	ans = list(range(m.GetNumAtoms()))
	with np_temp_seed(rseed):
		np.random.shuffle(ans)
	nm = Chem.RenumberAtoms(m,ans)
	smiles = Chem.MolToSmiles(nm, canonical=False, isomericSmiles=isomeric, kekuleSmiles=kekule)
	assert not (smiles is None)
	return smiles

def split_smiles(smiles_str):
	
	token_list = []
	ptr = 0

	while ptr < len(smiles_str):
		if smiles_str[ptr:ptr + 2] in TWO_LETTER_TOKEN_NAMES:
			smiles_char = smiles_str[ptr:ptr + 2]
			if smiles_char in LC_TWO_LETTER_MAP:
				smiles_char = LC_TWO_LETTER_MAP[smiles_char]
			token_list.append(smiles_char)
			ptr += 2
		else:
			smiles_char = smiles_str[ptr]
			token_list.append(smiles_char)
			ptr += 1

	return token_list

def list_replace(l,d):
	return [d[data] for data in l]

def mol_from_inchi(inchi):
	modules = rdkit_import("rdkit.Chem")
	Chem = modules[0]
	try:
		mol = Chem.MolFromInchi(inchi)
	except:
		mol = np.nan
	if none_or_nan(mol):
		mol = np.nan
	return mol

def mol_from_smiles(smiles):
	modules = rdkit_import("rdkit.Chem")
	Chem = modules[0]
	try:
		mol = Chem.MolFromSmiles(smiles)
	except:
		mol = np.nan
	if none_or_nan(mol):
		mol = np.nan
	return mol

def mol_to_smiles(mol,canonical=True,isomericSmiles=False,kekuleSmiles=False):
	modules = rdkit_import("rdkit.Chem")
	Chem = modules[0]
	try:
		smiles = Chem.MolToSmiles(mol,canonical=canonical,isomericSmiles=isomericSmiles,kekuleSmiles=kekuleSmiles)
	except:
		smiles = np.nan
	return smiles

def mol_to_formula(mol):
	modules = rdkit_import("rdkit.Chem.AllChem")
	AllChem = modules[0]
	try:
		formula = AllChem.CalcMolFormula(mol)
	except:
		formula = np.nan
	return formula

def mol_to_inchikey(mol):
	modules = rdkit_import("rdkit.Chem.inchi")
	inchi = modules[0]
	try:
		inchikey = inchi.MolToInchiKey(mol)
	except:
		inchikey = np.nan
	return inchikey

def mol_to_inchikey_s(mol):
	modules = rdkit_import("rdkit.Chem.inchi")
	inchi = modules[0]
	try:
		inchikey = inchi.MolToInchiKey(mol)
		inchikey_s = inchikey[:14]
	except:
		inchikey_s = np.nan
	return inchikey_s

def mol_to_inchi(mol):
	modules = rdkit_import("rdkit.Chem.rdinchi")
	rdinchi = modules[0]
	try:
		inchi = rdinchi.MolToInchi(mol,options='-SNon')
	except:
		inchi = np.nan
	return inchi

def mol_to_mol_weight(mol,exact=True):
	modules = rdkit_import("rdkit.Chem.Descriptors")
	Desc = modules[0]
	if exact:
		mol_weight = Desc.ExactMolWt(mol)
	else:
		mol_weight = Desc.MolWt(mol)
	return mol_weight

def inchi_to_smiles(inchi):
	try:
		mol = mol_from_inchi(inchi)
		smiles = mol_to_smiles(mol)
	except:
		smiles = np.nan
	return smiles

def smiles_to_selfies(smiles):
	sf, Chem = rdkit_import("selfies","rdkit.Chem")
	try:
		# canonicalize, strip isomeric information, kekulize
		mol = Chem.MolFromSmiles(smiles)
		smiles = Chem.MolToSmiles(mol,canonical=False,isomericSmiles=False,kekuleSmiles=True)
		selfies = sf.encoder(smiles)
	except:
		selfies = np.nan
	return selfies

def make_morgan_fingerprint(mol, radius=3):

	modules = rdkit_import("rdkit.Chem.rdMolDescriptors","rdkit.DataStructs")
	rmd = modules[0]
	ds = modules[1]
	fp = rmd.GetHashedMorganFingerprint(mol,radius)
	fp_arr = np.zeros(1)
	ds.ConvertToNumpyArray(fp, fp_arr)
	return fp_arr

def make_rdkit_fingerprint(mol):

	chem, ds = rdkit_import("rdkit.Chem","rdkit.DataStructs")
	fp = chem.RDKFingerprint(mol)
	fp_arr = np.zeros(1)
	ds.ConvertToNumpyArray(fp,fp_arr)
	return fp_arr

def make_maccs_fingerprint(mol):

	maccs, ds = rdkit_import("rdkit.Chem.MACCSkeys","rdkit.DataStructs")
	fp = maccs.GenMACCSKeys(mol)
	fp_arr = np.zeros(1)
	ds.ConvertToNumpyArray(fp,fp_arr)
	return fp_arr

def split_selfies(selfies_str):
	selfies = importlib.import_module("selfies")
	selfies_tokens = list(selfies.split_selfies(selfies_str))
	return selfies_tokens

def seq_apply(iterator,func):

	result = []
	for i in iterator:
		result.append(func(i))
	return result

def par_apply(iterator,func):

	n_jobs = joblib.cpu_count()
	par_func = joblib.delayed(func)
	result = joblib.Parallel(n_jobs=n_jobs)(par_func(i) for i in iterator)
	return result

def par_apply_series(series,func):

	series_iter = tqdm(series.iteritems(),desc=pformat(func),total=series.shape[0])
	series_func = lambda tup: func(tup[1])
	result_list = par_apply(series_iter,series_func)
	result_series = pd.Series(result_list,index=series.index)
	return result_series

def seq_apply_series(series,func):

	series_iter = tqdm(series.iteritems(),desc=pformat(func),total=series.shape[0])
	series_func = lambda tup: func(tup[1])
	result_list = seq_apply(series_iter,series_func)
	result_series = pd.Series(result_list,index=series.index)
	return result_series

def par_apply_df_rows(df,func):

	df_iter = tqdm(df.iterrows(),desc=pformat(func),total=df.shape[0])
	df_func = lambda tup: func(tup[1])
	result_list = par_apply(df_iter,df_func)
	if isinstance(result_list[0],tuple):
		result_series = tuple([pd.Series(rl,index=df.index) for rl in zip(*result_list)])
	else:
		result_series = pd.Series(result_list,index=df.index)
	return result_series

def seq_apply_df_rows(df,func):

	df_iter = tqdm(df.iterrows(),desc=pformat(func),total=df.shape[0])
	df_func = lambda tup: func(tup[1])
	result_list = seq_apply(df_iter,df_func)
	if isinstance(result_list[0],tuple):
		result_series = tuple([pd.Series(rl,index=df.index) for rl in zip(*result_list)])
	else:
		result_series = pd.Series(result_list,index=df.index)
	return result_series

def parse_ace_str(ce_str):

	if none_or_nan(ce_str):
		return np.nan
	matches = {
		# nist ones
		r"^[\d]+[.]?[\d]*$": lambda x: float(x), # this case is ambiguous (float(x) >= 2. or float(x) == 0.)
		r"^[\d]+[.]?[\d]*[ ]?eV$": lambda x: float(x.rstrip(" eV")),
		r"^NCE=[\d]+[.]?[\d]*% [\d]+[.]?[\d]*eV$": lambda x: float(x.split()[1].rstrip("eV")),
		# other ones
		r"^[\d]+[.]?[\d]*HCD$": lambda x: float(x.rstrip("HCD")),
		r"^CE [\d]+[.]?[\d]*$": lambda x: float(x.lstrip("CE ")),
	}
	for k,v in matches.datas():
		# try:
		if re.match(k,ce_str):
			return v(ce_str)
		# except:
		# 	import pdb; pdb.set_trace()
	return np.nan

def parse_nce_str(ce_str):

	if none_or_nan(ce_str):
		return np.nan
	matches = {
		# nist ones
		r"^NCE=[\d]+[.]?[\d]*% [\d]+[.]?[\d]*eV$": lambda x: float(x.split()[0].lstrip("NCE=").rstrip("%")),
		r"^NCE=[\d]+[.]?[\d]*%$": lambda x: float(x.lstrip("NCE=").rstrip("%")),
		# other ones
		r"^[\d]+[.]?[\d]*$": lambda x: 100.*float(x) if float(x) < 2. else np.nan, # this case is ambiguous
		r"^[\d]+[.]?[\d]*[ ]?[%]? \(nominal\)$": lambda x: float(x.rstrip(" %(nominal)")),
		r"^HCD [\d]+[.]?[\d]*%$": lambda x: float(x.lstrip("HCD ").rstrip("%")),
		r"^[\d]+[.]?[\d]* NCE$": lambda x: float(x.rstrip("NCE")),
		r"^[\d]+[.]?[\d]*\(NCE\)$": lambda x: float(x.rstrip("(NCE)")),
		r"^[\d]+[.]?[\d]*[ ]?%$": lambda x: float(x.rstrip(" %")),
		r"^HCD \(NCE [\d]+[.]?[\d]*%\)$": lambda x: float(x.lstrip("HCD (NCE").rstrip("%)")),
	}
	for k,v in matches.datas():
		if re.match(k,ce_str):
			return v(ce_str)
	return np.nan

# def parse_ramp_ce_str(ce_str):

# 	r"^[\d]+[.]?[\d]*->[\d]+[.]?[\d]*%$": (float(x.split("->")[0]),float(x.split("->")[1].rstrip("%"))),
# 	r"[\d]+[.]?[\d]*-[\d]+[.]?[\d]*": lambda x: (float(x.split("-")[0]),float(x.split("-")[1])),

def parse_inst_info(df):

	inst_type_str = df["inst_type"]
	inst_str = df["inst"]
	frag_mode_str = df["frag_mode"]
	col_energy_str = df["col_energy"]
	# instrument type
	if none_or_nan(inst_type_str):
		# resort to instrument
		inst_map = {
			"Maxis II HD Q-TOF Bruker": "QTOF",
			"qToF": "QTOF",
			"Orbitrap": "FT"
		}
		if none_or_nan(inst_str):
			inst_type = np.nan
		elif inst_str in inst_map:
			inst_type = inst_map[inst_str]
		else:
			inst_type = "Other"
	else:
		inst_type_map = {
			"QTOF": "QTOF",
			"FT": "FT",
			"Q-TOF": "QTOF",
			"HCD": "FT",
			"QqQ": "QQQ",
			"QqQ/triple quadrupole": "QQQ",
			"IT/ion trap": "IT",
			"IT-FT/ion trap with FTMS": "FT",
			"Q-ToF (LCMS)": "QTOF",
			"Bruker Q-ToF (LCMS)": "QTOF",
			"ESI-QTOF": "QTOF",
			"ESI-QFT": "FT",
			"ESI-ITFT": "FT",
			"Linear Ion Trap": "IT",
			"LC-ESI-QTOF": "QTOF",
			"LC-ESI-QFT": "FT",
			"LC-ESI-QQQ": "QQQ",
			"LC-Q-TOF/MS": "QTOF",
			"LC-ESI-ITFT": "FT",
			"LC-ESI-IT": "IT",
			"LC-QTOF": "QTOF",
			"qToF": "QTOF",
		}
		if inst_type_str in inst_type_map:
			inst_type = inst_type_map[inst_type_str]
		else:
			inst_type = "Other"
	# fragmentation mode
	if inst_type_str == "HCD":
		frag_mode = "HCD"
	elif type(col_energy_str) == str and "HCD" in col_energy_str:
		frag_mode = "HCD"
	elif none_or_nan(frag_mode_str) or frag_mode_str == "CID":
		frag_mode = "CID"
	elif frag_mode_str == "HCD":
		frag_mode = "HCD"
	else:
		frag_mode = np.nan
	return inst_type, frag_mode

def parse_ion_mode_str(ion_mode_str):

	if none_or_nan(ion_mode_str):
		return np.nan	
	if ion_mode_str in ["P","N","E"]:
		return ion_mode_str
	elif ion_mode_str == "POSITIVE":
		return "P"
	elif ion_mode_str == "NEGATIVE":
		return "N"
	else:
		return np.nan

def parse_prec_type_str(prec_type_str):

	if none_or_nan(prec_type_str):
		return np.nan
	elif prec_type_str.endswith("1+"):
		return prec_type_str.replace("1+","+")
	elif prec_type_str.endswith("1-"):
		return prec_type_str.replace("1-","-")
	else:
		return prec_type_str

def parse_peaks_str(peaks_str):
	# peaks still represented as string
	if none_or_nan(peaks_str):
		return np.nan
	lines = peaks_str.split("\n")
	peaks = []
	for line in lines:
		if len(line) == 0:
			continue
		line = line.split(" ")
		mz = line[0]
		ints = line[1]
		peaks.append((mz,ints))
	return peaks

def convert_peaks_to_float(peaks):
	# assumes no nan
	float_peaks = []
	for peak in peaks:
		float_peaks.append((float(peak[0]),float(peak[1])))
	return float_peaks

def get_res(peaks):
	# assumes no nan
	ress = []
	for mz,ints in peaks:
		dec_idx = mz.find(".")
		if dec_idx == -1:
			res = 0
		else:
			res = len(mz) - (dec_idx+1)
		ress.append(res)
	highest_res = max(ress)
	return highest_res

def get_murcko_scaffold(mol,output_type="smiles",include_chirality=False):

	MurckoScaffold = importlib.import_module("rdkit.Chem.Scaffolds.MurckoScaffold")
	if output_type == "smiles":
		scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol,includeChirality=include_chirality)
	else:
		raise NotImplementedError
	return scaffold


def atom_type_one_hot(atom):
	
	chemutils = importlib.import_module("dgllife.utils")
	return chemutils.atom_type_one_hot(
			atom, allowable_set = ELEMENT_LIST, encode_unknown = True
	)


def atom_bond_type_one_hot(atom):
		
	chemutils = importlib.import_module("dgllife.utils")
	bs = atom.GetBonds()
	if not bs:
		return [False, False, False, False]
	bt = np.array([chemutils.bond_type_one_hot(b) for b in bs])
	return [any(bt[:, i]) for i in range(bt.shape[1])]

def analyze_mol(mol):

	import rdkit
	from rdkit.Chem.Descriptors import MolWt
	import rdkit.Chem as Chem
	mol_dict = {}
	mol_dict["num_atoms"] = mol.GetNumHeavyAtoms()
	mol_dict["num_bonds"] = mol.GetNumBonds(onlyHeavy=True)
	mol_dict["mol_weight"] = MolWt(mol)
	mol_dict["num_rings"] = len(list(Chem.GetSymmSSSR(mol)))
	mol_dict["max_ring_size"] = max([-1]+[len(list(atom_iter)) for atom_iter in Chem.GetSymmSSSR(mol)])
	cnops_counts = {"C": 0, "N": 0, "O": 0, "P": 0, "S": 0, "Cl": 0, "other": 0}
	bond_counts = {"single": 0, "double": 0, "triple": 0, "aromatic": 0}
	cnops_bond_counts = {"C": [-1], "N": [-1], "O": [-1], "P": [-1], "S": [-1], "Cl": [-1]}
	h_counts = 0
	p_num_bonds = 	[-1]
	s_num_bonds = [-1]
	other_atoms = set()
	for atom in mol.GetAtoms():
		atom_symbol = atom.GetSymbol()
		if atom_symbol in cnops_counts:
			cnops_counts[atom_symbol] += 1
			cnops_bond_counts[atom_symbol].append(len(atom.GetBonds()))
		else:
			cnops_counts["other"] += 1
			other_atoms.add(atom_symbol)
		h_counts += atom.GetNumImplicitHs()
	for bond in mol.GetBonds():
		bond_type = bond.GetBondType()
		if bond_type == rdkit.Chem.rdchem.BondType.SINGLE:
			bond_counts["single"] += 1
		elif bond_type == rdkit.Chem.rdchem.BondType.DOUBLE:
			bond_counts["double"] += 1
		elif bond_type == rdkit.Chem.rdchem.BondType.TRIPLE:
			bond_counts["triple"] += 1
		else:
			assert bond_type == rdkit.Chem.rdchem.BondType.AROMATIC
			bond_counts["aromatic"] += 1
	mol_dict["other_atoms"] = ",".join(sorted(list(other_atoms)))
	mol_dict["H_counts"] = h_counts
	for k,v in cnops_counts.datas():
		mol_dict[f"{k}_counts"] = v
	for k,v in bond_counts.datas():
		mol_dict[f"{k}_counts"] = v
	for k,v in cnops_bond_counts.datas():
		mol_dict[f"{k}_max_bond_counts"] = max(v)
	return mol_dict

def check_atoms(mol):
	rdkit = importlib.import_module("rdkit")
	valid = all(a.GetSymbol() in ELEMENT_LIST for a in mol.GetAtoms())
	return valid

def check_num_bonds(mol):
	rdkit = importlib.import_module("rdkit")
	valid = mol.GetNumBonds() > 0
	return valid
