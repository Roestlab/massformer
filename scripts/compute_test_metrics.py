import torch as th
import torch_scatter as th_s
import os
import wandb
from pprint import pprint
import tempfile
import pandas as pd

from massformer.losses import get_sim_func
from massformer.spec_utils import merge_spec, unprocess_spec, process_spec
from massformer.model import cfm_postprocess
from massformer.dataset import data_to_device
from massformer.runner import load_config, get_ds_model


def init_model_from_chkpt(chkpt, chkpt_type, model):

    if chkpt_type == "local":
        print(f"> loading local checkpoint {chkpt}")
        chkpt_fp = chkpt
        chkpt_d = th.load(chkpt_fp,map_location="cpu")
        model.load_state_dict(chkpt_d["best_model_sd"])
    else:
        print(f"> downloading checkpoint {chkpt} from wandb")
        assert chkpt_type == "wandb", chkpt_type
        # download usin wandb api
        with tempfile.TemporaryDirectory(dir=os.getcwd()) as temp_dp:
            api = wandb.Api(timeout=20)
            wandb_path = f"adamoyoung/msms/{chkpt}"
            run = api.run(path=wandb_path)
            chkpt_fp = run.file("chkpt.pkl").download(root=temp_dp,replace=False)
            chkpt_d = th.load(os.path.join(temp_dp,"chkpt.pkl"),map_location="cpu")
            model.load_state_dict(chkpt_d["best_model_sd"])
    return model


def calculate_sim(
    pred, 
    targ, 
    prec_mz_idx, 
    group_id, 
    mol_id,
    transform, 
    prec_peak,
    merge,
    sim_func_name,
    mz_bin_res):

    # translate tranform
    if transform == "none":
        t = "none"
        n = "l1"
    elif transform == "log10":
        t = "log10"
        n = "l1"
    elif transform == "log10over3":
        t = "log10over3"
        n = "l1"
    elif transform == "sqrt":
        t = "sqrt"
        n = "l1"
    else:
        raise ValueError
    if prec_peak:
        def pre_proc(x,pm): 
            x_copy = x.clone()
            x_copy[th.arange(x.shape[0],device=x.device),pm] = 0.
            return x_copy
    else:
        def pre_proc(x,pm): return x
    # preprocess
    pred = unprocess_spec(pred, "log10over3")
    targ = unprocess_spec(targ, "log10over3")
    pred = pre_proc(pred, prec_mz_idx)
    targ = pre_proc(targ, prec_mz_idx)
    pred = process_spec(pred, t, n)
    targ = process_spec(targ, t, n)
    if merge:
        pred, m_group_id, m_mol_id = merge_spec(
            pred, group_id, t, n, mol_id)
        targ, _ = merge_spec(targ, group_id, t, n)
        group_id, mol_id = m_group_id, m_mol_id
    # compute sim
    sim_func = get_sim_func(sim_func_name, mz_bin_res)
    sim = sim_func(pred, targ)
    return sim, group_id, mol_id
    

def compute_metrics(
        pred,
        targ,
        prec_mz_idx,
        mol_id,
        group_id,
        sim_func_names,
        transforms,
        mz_bin_res):

    # do unmerged first
    sim_d = {}
    for sim_func_name in sim_func_names:
        for transform in transforms:
            if sim_func_name == "jacc" and transform != "none":
                continue
            for prec_peak in [False, True]:
                for merge in [False, True]:
                    key = f"{sim_func_name}_{transform}"
                    if prec_peak:
                        key += "_drop"
                    else:
                        key += "_keep"
                    if merge:
                        key += "_merge"
                    else:
                        key += "_unmerge"
                    a_sim, a_group_id, a_mol_id = calculate_sim(
                        pred=pred, 
                        targ=targ, 
                        prec_mz_idx=prec_mz_idx, 
                        group_id=group_id, 
                        mol_id=mol_id,
                        transform=transform, 
                        prec_peak=prec_peak, 
                        merge=merge, 
                        sim_func_name=sim_func_name, 
                        mz_bin_res=mz_bin_res
                    )
                    if th.any(th.isnan(a_sim)):
                        print(f"> Warning: NaN sim value ({key})")
                    sim_d[key] = (a_sim, a_group_id, a_mol_id)
    agg_sim_d = {}
    for k,v in sim_d.items():
        a_sim, a_group_id, a_mol_id = v
        for agg_mol in [False, True]:
            key = k
            if agg_mol:
                key += "_mol"
                un_mol_id, un_mol_idx = th.unique(
                    a_mol_id, dim=0, return_inverse=True)
                a_agg_sim = th_s.scatter_mean(
                    a_sim,
                    un_mol_idx,
                    dim=0,
                    dim_size=un_mol_id.shape[0])
                a_agg_sim = th.mean(a_agg_sim,dim=0)
            else: # agg_spec
                key += "_spec"
                a_agg_sim = th.mean(a_sim,dim=0)
            agg_sim_d[key] = a_agg_sim
    return sim_d, agg_sim_d


def get_wandb_chkpts(group):

    api = wandb.Api(timeout=20)
    wandb_path = f"adamoyoung/msms"
    runs = api.runs(path=wandb_path,filters={"group":group})
    chkpts = []
    for run_idx,run in enumerate(runs):
        if "outlier" in run.tags:
            continue
        assert run.state == "finished", run.state
        chkpts.append(run.id)
    return chkpts


def get_model_metrics(custom_fp, chkpts):

    template_fp = "config/template.yml"
    device_id = 0

    # load config
    entity_name, project_name, run_name, data_d, model_d, run_d = load_config(
        template_fp,
        custom_fp,
        device_id,
        None
    )
    # ### DEBUG
    # data_d["num_entries"] = 1000
    # ### DEBUG

    # load model, datasets, dataloaders
    ds, model, _, _, _ = get_ds_model(data_d, model_d, run_d)
    dl_d, _ = ds.get_dataloaders(run_d)
    if "mb_na" in data_d["secondary_dset"]:
        dl = dl_d["secondary"]["mb_na"]
    else:
        dl = dl_d["primary"]["test"]

    # checkpoints
    chkpt_types = ["wandb"] * len(chkpts)

    # setup device things
    dev = th.device(run_d["device"])
    nb = run_d["non_blocking"]

    chkpt_d = {}
    for chkpt_idx, (chkpt, chkpt_type) in enumerate(zip(chkpts, chkpt_types)):
        print(f"> Starting checkpoint {chkpt_idx} - {chkpt}")
        model = init_model_from_chkpt(chkpt, chkpt_type, model)
        model.to(dev)
        model.eval()
        with th.no_grad():
            pred, targ, prec_mz_idx, mol_id, group_id = [], [], [], [], []
            for b_idx, b in enumerate(dl):
                b = data_to_device(b, dev, nb)
                b_pred = model(data=b, amp=run_d["amp"])["pred"]
                b_targ = b["spec"]
                b_prec_mz_idx = b["prec_mz_idx"]
                b_mol_id = b["mol_id"]
                b_group_id = b["group_id"]
                pred.append(b_pred.detach().to("cpu", non_blocking=nb))
                targ.append(b_targ.detach().to("cpu", non_blocking=nb))
                prec_mz_idx.append(
                b_prec_mz_idx.detach().to(
                    "cpu", non_blocking=nb))
                mol_id.append(
                    b_mol_id.detach().to(
                        "cpu", non_blocking=nb))
                group_id.append(
                    b_group_id.detach().to(
                        "cpu", non_blocking=nb))
            pred = th.cat(pred, dim=0)
            targ = th.cat(targ, dim=0)
            prec_mz_idx = th.cat(prec_mz_idx, dim=0)
            mol_id = th.cat(mol_id, dim=0)
            group_id = th.cat(group_id, dim=0)
        sim_func_names = ["cos", "js", "jacc"]
        transforms = ["none", "log10over3", "sqrt"]
        sim_d, agg_sim_d = compute_metrics(
            pred=pred,
            targ=targ,
            prec_mz_idx=prec_mz_idx,
            mol_id=mol_id,
            group_id=group_id,
            sim_func_names=sim_func_names,
            transforms=transforms,
            mz_bin_res=data_d["mz_bin_res"]
        )
        # pprint(agg_sim_d)
        for k,v in agg_sim_d.items():
            if k not in chkpt_d:
                chkpt_d[k] = []
            chkpt_d[k].append(v)

    # mean and std
    agg_chkpt_d = {}
    for k in chkpt_d.keys():
        agg_chkpt_metric = th.stack(chkpt_d[k])
        agg_chkpt_d[k] = (th.mean(agg_chkpt_metric,dim=0), th.std(agg_chkpt_metric,dim=0))
    
    return agg_chkpt_d


if __name__ == "__main__":

    dsets = ["nist","mona"]
    splits = ["inchikey","scaffold"]
    prec_types = ["mh","all"]
    model_names = ["FP","WLN","MF","CFM"]
    # ### DEBUG
    # dsets = ["mona"]
    # splits = ["inchikey"]
    # prec_types = ["all"]
    # model_names = ["FP"]
    # ### DEBUG

    groups, config_fps = [], []
    for dset in dsets:
        for split in splits:
            for prec_type in prec_types:
                for model_name in model_names:
                    key = f"{dset}_{split}_{prec_type}_{model_name}"
                    group_name = f"{key}_rand"
                    groups.append(group_name)
                    config_fp = f"config/{prec_type}_prec_type/{key}.yml"
                    config_fps.append(config_fp)
    print(f">>> Total num groups = {len(groups)}")

    df_rows = []
    for group_idx, (group, config_fp) in enumerate(zip(groups, config_fps)):
        print(f">> Staring group {group_idx} - {group}")
        chkpts = get_wandb_chkpts(group)
        # ### DEBUG
        # chkpts = chkpts[:2]
        # ### DEBUG
        model_agg_chkpt_d = get_model_metrics(config_fp, chkpts)
        for k,v in model_agg_chkpt_d.items():
            dset, split, prec_type, model_name, _ = group.split("_")
            sim_func, transform, prec_peak, merge, aggregation = k.split("_")
            df_rows.append({
                "dset": dset,
                "split": split,
                "prec_type": prec_type,
                "model": model_name,
                "sim_func": sim_func,
                "transform": transform,
                "prec_peak": prec_peak,
                "merge": merge,
                "aggregation": aggregation,
                "mean": v[0].item(),
                "std": v[1].item()
            })
    df = pd.DataFrame(df_rows)
    print(df)
    df_fp = "figs/test_metrics/all_table.csv"
    os.makedirs(os.path.dirname(df_fp),exist_ok=True)
    df.to_csv(df_fp,index=False)

