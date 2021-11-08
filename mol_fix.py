import rdkit
import rdkit.Chem as Chem
import rdkit.RDLogger as RDLogger
import os
import glob
from tqdm import tqdm
import argparse

from misc_utils import booltype

parser = argparse.ArgumentParser()
parser.add_argument("nist_mol_dp",type=str,required=True)
parser.add_argument("--overwrite",type=booltype,default=True)
args = parser.parse_args()

RDLogger.DisableLog('rdApp.*')

mol_dp = args.nist_mol_dp
assert os.path.isdir(mol_dp), mol_dp

m_end = "M  END\n"
fns = glob.glob(f"{mol_dp}/*")
bad_fps = []
m_end_lt = []
m_end_gt = []
m_end_eq = []
for fn in tqdm(fns):
	fp = os.path.join(mol_dp,fn)
	mol = Chem.MolFromMolFile(fp)
	if mol is None:
		bad_fps.append(fp)
		with open(fp,"r") as file:
			lines = file.readlines()
		m_end_count = sum(line == m_end for line in lines)
		if m_end_count < 1:
			m_end_lt.append(fp)
			lines.insert(-1,m_end)
		elif m_end_count > 1:
			m_end_gt.append(fp)
			for i in range(m_end_count-1):
				assert lines[-2] == m_end
				lines.pop(-2)
		else:
			m_end_eq.append(fp)
		if args.overwrite:
			with open(fp,"w") as file:
				file.writelines(lines)
print(len(bad_fps))
print(len(m_end_lt))
print(len(m_end_gt))
print(len(m_end_eq))
print(m_end_eq)
