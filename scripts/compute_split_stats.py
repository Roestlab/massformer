import pandas as pd
import numpy as np

from massformer.data_utils import mol_to_charge
from massformer.runner import load_config, get_ds_model


if __name__ == "__main__":

    ps_to_config = {
        "mh_nist-inchikey": "config/mh_prec_type/nist_inchikey_mh_FP.yml",
        "mh_nist-scaffold": "config/mh_prec_type/nist_scaffold_mh_FP.yml",
        "mh_mona-inchikey": "config/mh_prec_type/mona_inchikey_mh_FP.yml",
        "mh_mona-scaffold": "config/mh_prec_type/mona_scaffold_mh_FP.yml",
        "all_nist-inchikey": "config/all_prec_type/nist_inchikey_all_FP.yml",
        "all_nist-scaffold": "config/all_prec_type/nist_scaffold_all_FP.yml",
        "all_mona-inchikey": "config/all_prec_type/mona_inchikey_all_FP.yml",
        "all_mona-scaffold": "config/all_prec_type/mona_scaffold_all_FP.yml",
    }

    # load the datasets
    template_fp = "config/template.yml"
    device_id = -1
    split_entries = []
    for k,v in ps_to_config.items():
        adduct, split = k.split("_")
        custom_fp = v
        _, _, _, data_d, model_d, run_d = load_config(
            template_fp,
            custom_fp,
            device_id,
            None
        )
        ds, model, _, _, _ = get_ds_model(data_d, model_d, run_d)
        spec_df = ds.spec_df
        mol_df = ds.mol_df
        val_frac = run_d["val_frac"]
        test_frac = run_d["test_frac"]
        sec_frac = run_d["sec_frac"]
        split_key = run_d["split_key"]
        split_seed = run_d["split_seed"]
        ignore_casmi = run_d["ignore_casmi_in_split"]
        train_mask, val_mask, test_mask, sec_masks = ds.get_split_masks(
            val_frac, test_frac, sec_frac, split_key, split_seed, ignore_casmi)
        train_spec_df = spec_df[train_mask]
        train_mol_df = mol_df[mol_df["mol_id"].isin(train_spec_df["mol_id"])]
        train_num_mols = train_mol_df["inchikey_s"].nunique()
        val_spec_df = spec_df[val_mask]
        val_mol_df = mol_df[mol_df["mol_id"].isin(val_spec_df["mol_id"])]
        val_num_mols = val_mol_df["inchikey_s"].nunique()
        test_spec_df = spec_df[test_mask]
        test_mol_df = mol_df[mol_df["mol_id"].isin(test_spec_df["mol_id"])]
        test_num_mols = test_mol_df["inchikey_s"].nunique()
        if len(sec_masks) > 0:
            assert len(sec_masks) == 1, sec_masks
            sec_spec_df = spec_df[sec_masks[0]]
            sec_mol_df = mol_df[mol_df["mol_id"].isin(sec_spec_df["mol_id"])]
            sec_num_mols = sec_mol_df["inchikey_s"].nunique()
        else:
            sec_num_mols = 0
        split_entry = {
            "adduct": adduct,
            "split": split,
            "nist_train": train_num_mols,
            "nist_val": val_num_mols,
            "nist_test": test_num_mols,
            "mona_test": sec_num_mols,
        }
        split_entries.append(split_entry)
    split_df = pd.DataFrame(split_entries)
    split_df.to_csv("figs/dset/split_stats.csv",index=False)
    print(split_df)
