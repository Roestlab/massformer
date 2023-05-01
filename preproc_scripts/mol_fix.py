import rdkit
import rdkit.Chem as Chem
import rdkit.RDLogger as RDLogger
import os
import glob
from tqdm import tqdm
import argparse

from massformer.misc_utils import booltype


def main(args):

    RDLogger.DisableLog('rdApp.*')

    assert os.path.isdir(args.nist_raw_dp), args.nist_raw_dp
    mol_dp = os.path.join(args.nist_raw_dp, "hr_nist_msms.MOL")
    assert os.path.isdir(mol_dp), mol_dp

    m_end = "M  END\n"
    fns = glob.glob(f"{mol_dp}/*")
    bad_fps = []
    m_end_lt = []
    m_end_gt = []
    m_end_eq = []
    for fn in tqdm(fns, total=len(fns), desc="> mol_fix"):
        fp = os.path.abspath(fn)
        mol = Chem.MolFromMolFile(fp)
        if mol is None:
            bad_fps.append(fp)
            with open(fp, "r") as file:
                lines = file.readlines()
            m_end_count = sum(line == m_end for line in lines)
            if m_end_count < 1:
                m_end_lt.append(fp)
                lines.insert(-1, m_end)
            elif m_end_count > 1:
                m_end_gt.append(fp)
                for i in range(m_end_count - 1):
                    assert lines[-2] == m_end
                    lines.pop(-2)
            else:
                m_end_eq.append(fp)
            if args.overwrite:
                with open(fp, "w") as file:
                    file.writelines(lines)
    print(len(bad_fps))
    print(len(m_end_lt))
    print(len(m_end_gt))
    print(len(m_end_eq))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--overwrite", type=booltype, default=False)
    parser.add_argument("--nist_raw_dp", type=str, default="data/raw/nist_20")
    args = parser.parse_args()
    main(args)
