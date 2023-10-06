import argparse
import numpy as np
import torch as th
import pandas as pd
import copy
import os

import massformer.data_utils as data_utils
from massformer.data_utils import H_MASS, O_MASS, NA_MASS, N_MASS, C_MASS, par_apply_series
from massformer.runner import init_wandb_run, load_config, train_and_eval, get_ds_model, get_pbar
from massformer.dataset import data_to_device
from massformer.spec_utils import unprocess_spec, process_spec

def init_from_smiles(dataset, smiles_df, prec_types, nces, run_d):

    ds = copy.deepcopy(dataset)
    MW_DIFF = {
        '[M+H]+': H_MASS, 
        '[M+H-H2O]+': -H_MASS-O_MASS, 
        '[M+H-2H2O]+': -3*H_MASS+2*O_MASS, 
        '[M+2H]2+': 2*H_MASS, 
        '[M+H-NH3]+': -2*H_MASS-N_MASS, 
        "[M+Na]+": NA_MASS
    }
    ds.is_fp_dset = dataset.is_fp_dset
    ds.is_graph_dset = dataset.is_graph_dset
    ds.is_gf_v2_dset = dataset.is_gf_v2_dset
    ds.is_esp_dset = dataset.is_esp_dset
    # preprocess molecules
    mol_df = smiles_df.copy()
    mol_df.loc[:, "mol"] = par_apply_series(
        mol_df["smiles"], data_utils.mol_from_smiles)
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
    mol_df = mol_df.dropna(subset=["mol"])
    # generate fake spectra (for compatibility with dataset)
    def calculate_prec_mz(prec_type, exact_mw):
        prec_mz = exact_mw + MW_DIFF[prec_type]
        charge = data_utils.get_charge(prec_type)
        prec_mz = prec_mz / charge
        return prec_mz
    num_mol = mol_df.shape[0]
    num_nces = len(nces)
    num_prec_types = len(prec_types)
    rows = []
    for i in range(num_mol):
        for j in range(num_prec_types):
            for k in range(num_nces):
                row_d = {}
                row_d["spec_id"] = i*(num_nces+num_prec_types) + j*(num_nces) + k
                row_d["mol_id"] = mol_df["mol_id"].iloc[i]
                row_d["prec_type"] = prec_types[j]
                row_d["prec_mz"] = calculate_prec_mz(prec_types[j],mol_df["exact_mw"].iloc[i])
                row_d["ace"] = np.nan
                row_d["nce"] = nces[k]
                row_d["inst_type"] = "FT"
                row_d["frag_mode"] = "HCD"
                row_d["spec_type"] = "MS2"
                row_d["ion_mode"] = "P"
                row_d["dset"] = "new"
                row_d["peaks"] = [(100.,0.2),(200.,0.3),(500.,0.5)]
                row_d["res"] = 7
                row_d["col_gas"] = np.nan # ignored
                row_d["ri"] = np.nan # ignored
                rows.append(row_d)
    spec_df = pd.DataFrame(rows)
    ds.spec_df = spec_df
    ds.mol_df = mol_df
    # modify flags
    ds.num_entries = -1
    ds.subsample_size = -1
    ds.primary_dset = ["new"]
    ds.secondary_dset = []
    ds._select_spec()
    # set mol_id as index
    ds.mol_df = ds.mol_df.set_index("mol_id", drop=False).sort_index().rename_axis(None)
    # copy over the metadata
    for k in ["inst_type_c2i", "inst_type_i2c", "prec_type_c2i", "prec_type_i2c", "frag_mode_c2i", "frag_mode_i2c"]:
        setattr(ds, k, getattr(dataset, k))
    for k in ["num_inst_type", "num_prec_type", "num_frag_mode", "max_ce", "mean_ce", "std_ce"]:
        setattr(ds, k, getattr(dataset, k))        
    ds.casmi_info = {"inchikey_s": set(), "scaffold": set()}
    # dataloader stuff
    batch_size = run_d["batch_size"] // run_d["grad_acc_interval"]
    num_workers = run_d["num_workers"]
    pin_memory = run_d["pin_memory"] if run_d["device"] != "cpu" else False
    collate_fn = ds.get_collate_fn()
    dl = th.utils.data.DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        collate_fn=collate_fn)
    return ds, dl

def convert_to_peaks(spec,mz_max,mz_bin_res):

    nz_idx = np.nonzero(spec)[0]
    all_mzs = np.arange(0.,mz_max+mz_bin_res,mz_bin_res)
    mzs = all_mzs[nz_idx]
    ints = spec[nz_idx]
    peaks = [(mz,int_) for mz,int_ in zip(mzs,ints)]
    return peaks

def run_inference(
        model,
        dl,
        data_d,
        model_d,
        run_d,
        transform,
        normalization):

    # stuff related to device
    dev = th.device(run_d["device"])
    nb = run_d["non_blocking"]
    # model setup
    model.to(dev)
    model.eval()
    pred, spec_id, mol_id, group_id = [], [], [], []
    with th.no_grad():
        for b_idx, b in get_pbar(
                enumerate(dl), run_d, desc=f"> inference", total=len(dl)):
            b = data_to_device(b, dev, nb)
            b_pred = model(data=b, amp=run_d["amp"])["pred"]
            b_spec_id = b["spec_id"]
            b_mol_id = b["mol_id"]
            b_group_id = b["group_id"]
            pred.append(b_pred.detach().to("cpu", non_blocking=nb))
            spec_id.append(
                b_spec_id.detach().to(
                    "cpu", non_blocking=nb))
            mol_id.append(
                b_mol_id.detach().to(
                    "cpu", non_blocking=nb))
            group_id.append(
                b_group_id.detach().to(
                    "cpu", non_blocking=nb))
    pred = th.cat(pred, dim=0)
    spec_id = th.cat(spec_id, dim=0)
    mol_id = th.cat(mol_id, dim=0)
    group_id = th.cat(group_id, dim=0)
    # untransform
    pred = unprocess_spec(pred, data_d["transform"])
    # transform
    pred = process_spec(pred, transform, normalization)
    # split
    specs = [spec[0] for spec in np.vsplit(pred.numpy(),pred.shape[0])]
    # convert to peaks
    peakses = [convert_to_peaks(spec,data_d["mz_max"],data_d["mz_bin_res"]) for spec in specs]
    # stick everything into a dictionary
    out_d = {
        "peaks": peakses,
        "spec_id": spec_id.numpy(),
        "mol_id": mol_id.numpy(),
        "group_id": group_id.numpy()
    }
    return out_d

def main(flags):

    np.random.seed(flags.rseed)

    entity_name, project_name, run_name, data_d, model_d, run_d = load_config(
        flags.template_fp,
        flags.custom_fp,
        flags.device_id,
        None
    )

    ds, model, _, _, _ = get_ds_model(data_d, model_d, run_d)

    # load saved model from checkpoint
    if model_d["checkpoint_name"] is not None:
        chkpt_fp = os.path.join(
            data_d["checkpoint_dp"],
            model_d["checkpoint_name"] + ".pkl")
        chkpt_d = th.load(chkpt_fp,map_location="cpu")
        model.load_state_dict(chkpt_d["best_model_sd"])

    smiles_df = pd.read_csv(flags.smiles_fp,sep=",")

    new_ds, new_dl = init_from_smiles(ds, smiles_df, flags.prec_types, flags.nces, run_d)

    out_d = run_inference(model, new_dl, data_d, model_d, run_d, flags.transform, flags.normalization)

    out_df = pd.DataFrame(out_d)
    meta_cols = ["spec_id","mol_id","group_id","prec_type","prec_mz","nce","inst_type","frag_mode","spec_type","ion_mode"]
    spec_df = new_ds.spec_df[meta_cols]
    out_df = out_df.merge(
        spec_df,
        on=["spec_id","mol_id","group_id"],
        how="inner")
    out_df = out_df[meta_cols+["peaks"]]
    
    out_df.to_csv(flags.output_fp,index=False)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-t",
        "--template_fp",
        type=str,
        default="config/template.yml",
        help="path to template config file")
    parser.add_argument(
        "-d",
        "--device_id",
        type=int,
        required=False,
        help="device id (-1 for cpu)")
    parser.add_argument(
        "-c",
        "--custom_fp",
        type=str,
        default="config/demo.yml",
        help="path to custom config file")
    parser.add_argument(
        "-s",
        "--smiles_fp",
        type=str,
        required=True,
        help="path to smiles file (CSV)")
    parser.add_argument(
        "--nces",
        type=float,
        nargs="+",
        default=[20.,40.,60.,80.,100.],
        help="list of nces")
    parser.add_argument(
        "--prec_types",
        type=str,
        nargs="+",
        default=["[M+H]+"],
        help="list of prec_types")
    parser.add_argument(
        "-o",
        "--output_fp",
        type=str,
        required=True,
        help="path to output file (CSV)"
    )
    parser.add_argument(
        "--rseed",
        type=int,
        default=123456,
        help="random seed"
    )
    parser.add_argument(
        "--transform",
        type=str,
        default="none",
        choices=["none","log10","log10over3","loge","sqrt"],
        help="Transformation applied to predicted intensities"
    )
    parser.add_argument(
        "--normalization",
        type=str,
        default="l1",
        choices=["l1","l2","none"],
        help="Normalization applied to predicted intensities"
    )
    flags = parser.parse_args()
    main(flags)
