import pandas as pd
import numpy as np

from massformer.data_utils import mol_to_charge
from massformer.runner import load_config, get_ds_model


if __name__ == "__main__":

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
    adducts = data_d["pos_prec_type"]
    ds, model, _, _, _ = get_ds_model(data_d, model_d, run_d)
    spec_df = ds.spec_df
    mol_df = ds.mol_df

    # read in NIST/MoNA compounds
    adduct_entries = []
    for dsets in [["nist"],["mb_na"],["nist","mb_na"]]:
        dsets_str = "+".join(dsets)
        print(f">> {dsets_str}")
        spec_mask = spec_df["dset"].isin(dsets)
        # import pdb; pdb.set_trace()
        dset_spec_df = spec_df[spec_mask]
        dset_mol_df = mol_df[mol_df["mol_id"].isin(dset_spec_df["mol_id"])]
        num_mols = dset_mol_df["inchikey_s"].nunique()
        num_specs = len(dset_spec_df)
        num_ces_per_adduct = dset_spec_df[["spec_id","mol_id","prec_type"]].groupby(["mol_id","prec_type"]).count().reset_index()["spec_id"].mean()
        adduct_entry = {
            "dsets": dsets_str,
            "adduct": "all",
            "num_mols": num_mols,
            "num_specs": num_specs,
            "num_ces_per_adduct": num_ces_per_adduct,
        }
        adduct_entries.append(adduct_entry)
        for adduct in adducts:
            print(f"> {adduct}")
            adduct_entry = {}
            adduct_spec_df = dset_spec_df[dset_spec_df["prec_type"] == adduct]
            adduct_mol_df = dset_mol_df[dset_mol_df["mol_id"].isin(adduct_spec_df["mol_id"])]
            num_mols = adduct_mol_df["inchikey_s"].nunique()
            num_specs = len(adduct_spec_df)
            num_ces_per_adduct = adduct_spec_df[["spec_id","mol_id","prec_type"]].groupby(["mol_id","prec_type"]).count().reset_index()["spec_id"].mean()
            adduct_entry = {
                "dsets": dsets_str,
                "adduct": adduct,
                "num_mols": num_mols,
                "num_specs": num_specs,
                "num_ces_per_adduct": num_ces_per_adduct,
            }
            adduct_entries.append(adduct_entry)
    stats_df = pd.DataFrame(adduct_entries)
    stats_df = stats_df[["dsets","adduct","num_specs","num_mols","num_ces_per_adduct"]]
    stats_df.to_csv("figs/dset/adduct_stats.csv",index=False)
    print(stats_df)
