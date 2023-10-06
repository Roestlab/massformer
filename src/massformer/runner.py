import torch as th
import torch.nn.functional as F
import numpy as np
import wandb
import argparse
import yaml
import os
import torch_scatter as th_s
from tqdm import tqdm
import copy
import math
import pandas as pd
import tempfile
from pprint import pprint

from massformer.dataset import BaseDataset, CASMIDataset, data_to_device, get_dset_types
from massformer.model import Predictor, apply_postprocessing, cfm_postprocess
from massformer.gf_model import GFv2Embedder, flag_bounded
from massformer.cfm_model import CFMPredictor
from massformer.esp_model import ESPPredictor
from massformer.misc_utils import get_nograd_param_names, np_temp_seed, th_temp_seed, DummyContext, DummyScaler, sparse_to_dense, count_parameters
from massformer.plot_utils import *
from massformer.losses import get_loss_func, get_sim_func
from massformer.lr import PolynomialDecayLR
from massformer.metric_table import MetricTable
from massformer.spec_utils import process_spec, unprocess_spec, merge_spec
from massformer.data_utils import ELEMENT_LIST


def get_scaler(amp):
    if amp:
        scaler = th.cuda.amp.GradScaler()
    else:
        scaler = DummyScaler()
    return scaler


def get_pbar(iter, run_d, **pbar_kwargs):
    if run_d["log_tqdm"]:
        return tqdm(iter, **pbar_kwargs)
    else:
        if "desc" in pbar_kwargs:
            print(pbar_kwargs["desc"])
        return iter


def run_train_epoch(
        step,
        epoch,
        model,
        dl_d,
        data_d,
        model_d,
        run_d,
        use_wandb,
        optimizer,
        scheduler):

    # stuff related to device
    dev = th.device(run_d["device"])
    nb = run_d["non_blocking"]
    # set up loss func
    loss_func = get_loss_func(
        run_d["loss"],
        data_d["mz_bin_res"],
        agg=run_d["batch_loss_agg"])
    b_losses = []
    if run_d["lda_topic_loss"]:
        lda_loss_func = get_loss_func(
            "forw_kl",
            100,
            agg=run_d["batch_loss_agg"])
    # set up scaler
    scaler = get_scaler(run_d["amp"])
    # train
    model.train()
    # get embed dim for flag
    if run_d["flag"]:
        if isinstance(model, th.nn.DataParallel):
            embedders = model.module.embedders
        else:
            embedders = model.embedders
        gfv2_idx = [isinstance(embedder, GFv2Embedder)
                    for embedder in embedders].index(True)
        embed_dim = embedders[gfv2_idx].args.encoder_embed_dim
    # iterate
    for b_idx, b in get_pbar(
        enumerate(
            dl_d["train"]), run_d, desc="> train", total=len(
            dl_d["train"])):
        optimizer.zero_grad()
        b = data_to_device(b, dev, nb)
        b_output = model(
            data=b, 
            amp=run_d["amp"], 
            return_lda_pred=run_d["lda_topic_loss"])
        b_pred = b_output["pred"]
        b_targ = b["spec"]
        if run_d["flag"]:
            def forward(perturb):
                b_pred = model(data=b, perturb=perturb, amp=run_d["amp"])["pred"]
                return b_pred
            def backward(loss):
                # this backward is only meant for generating perturbations
                scaler.scale(loss).backward()
            n_graph, n_node = b["gf_v2_data"]["x"].shape[:2]
            b_perturb_shape = (n_graph, n_node, embed_dim)
            b_loss_agg, b_pred = flag_bounded(
                (model, forward, backward),
                b_perturb_shape,
                b_targ,
                optimizer,
                dev,
                loss_func,
                scaler,
                m=run_d["flag_m"],
                step_size=run_d["flag_step_size"],
                mag=run_d["flag_mag"],
                mask=None
            )
        else:
            b_loss_agg = loss_func(b_pred, b_targ)
        if run_d["lda_topic_loss"]:
            b_lda_pred = b_output["lda_pred"]
            b_lda_targ = b["lda_topic"]
            b_lda_loss_agg = lda_loss_func(b_lda_pred, b_lda_targ)
            b_loss_agg = b_loss_agg + run_d["lda_topic_loss_weight"] * b_lda_loss_agg
        # backpropagate loss
        scaler.scale(b_loss_agg / run_d["grad_acc_interval"]).backward()
        # take a gradient step if finished accumulating
        if step % run_d["grad_acc_interval"] == 0:
            # unscale then gradient clip
            scaler.unscale_(optimizer)
            th.nn.utils.clip_grad_norm_(
                model.parameters(), run_d["clip_grad_norm"])
            scaler.step(optimizer)  # calls optimizer.step()
            scaler.update()
            if run_d["scheduler"] == "polynomial":
                # polynomial updates per-step
                scheduler.step()
        # increment step counter
        step += 1
        # update losses
        b_losses.append(b_loss_agg.detach().to("cpu").item())
    optimizer.zero_grad()
    train_spec_loss = np.mean(b_losses)
    if use_wandb:
        log_d = {
            "train_spec_loss_obj_mean": train_spec_loss,
            "epoch": epoch,
            "Epoch": epoch,
        }
        wandb.log(log_d, commit=False)
    return step, epoch, {}


def compute_metric_tables(
        pred,
        targ,
        mol_id,
        group_id,
        prefix,
        data_d,
        run_d,
        auxiliary=False,
        merge_group=False,
        compute_agg=False,
        compute_hist=False,
        groupby_mol=False,
        um_batch_size=10000,
        m_batch_size=1000):

    def merge_group_func(_pred, _targ, _group_id, _mol_id, _transform):
        # merge spectra by mol
        assert group_id is not None and mol_id is not None
        # translate tranform
        if _transform == "obj":
            t = data_d["transform"]
            def pp(x): return x
            n = data_d["spectrum_normalization"]
        elif _transform == "std":
            t = "none"
            def pp(x): return x
            n = "l1"
        elif _transform == "log":
            t = "log10over3"
            def pp(x): return x
            n = "l1"
        elif _transform == "cfm":
            t = "none"
            def pp(x): return cfm_postprocess(x, "l1")
            n = "l1"
        else:
            raise ValueError
        m_pred, m_group_id, m_mol_id = merge_spec(
            _pred, _group_id, t, n, _mol_id)
        m_pred = pp(m_pred)
        m_targ, _ = merge_spec(_targ, _group_id, t, n)
        return m_pred, m_targ, m_mol_id, m_group_id
    # batching
    um_num_batches = len(pred) // um_batch_size + int(len(pred) % um_batch_size != 0)
    # functions
    obj_sim_func = get_sim_func(run_d["sim"], data_d["mz_bin_res"])
    obj_loss_func = get_loss_func(run_d["loss"], data_d["mz_bin_res"])
    cos_sim_func = get_sim_func("cos", data_d["mz_bin_res"])
    # do unmerged first
    sim_obj, loss_obj, sim_cos_std = [], [], []
    for b in get_pbar(range(um_num_batches),run_d,desc="> unmerged metrics"):
        b_pred = pred[b*um_batch_size:(b+1)*um_batch_size]
        b_targ = targ[b*um_batch_size:(b+1)*um_batch_size]
        # basic loss and sim
        b_sim_obj = obj_sim_func(b_pred, b_targ)
        b_loss_obj = obj_loss_func(b_pred, b_targ)
        sim_obj.append(b_sim_obj)
        loss_obj.append(b_loss_obj)
        if auxiliary:
            # just doing cos, forget about the other ones
            b_pred = process_spec(unprocess_spec(b_pred, data_d["transform"]),"none","l2")
            b_targ = process_spec(unprocess_spec(b_targ, data_d["transform"]),"none","l2")
            b_sim_cos_std = cos_sim_func(b_pred, b_targ)
            sim_cos_std.append(b_sim_cos_std)
    sim_d = {
        "sim_obj": th.cat(sim_obj,dim=0),
        "loss_obj": th.cat(loss_obj,dim=0)
    }
    if auxiliary:
        sim_d["sim_cos_std"] = th.cat(sim_cos_std,dim=0)
    # do merged second
    if merge_group:
        un_group_id = th.unique(group_id)
        # batching
        m_num_batches = len(un_group_id) // m_batch_size + int(len(un_group_id) % m_batch_size != 0)
        m_sim_obj, m_loss_obj, m_sim_cos_std, m_group_id, m_mol_id = [], [], [], [], []
        for b in get_pbar(range(m_num_batches),run_d,desc="> merged metrics"):
            b_group_id = un_group_id[b*m_batch_size:(b+1)*m_batch_size]
            b_mask = th.isin(group_id,b_group_id)
            b_group_id = group_id[b_mask]
            b_mol_id = mol_id[b_mask]
            b_pred = pred[b_mask]
            b_targ = targ[b_mask]
            b_m_pred, b_m_targ, b_m_mol_id, b_m_group_id = merge_group_func(
                b_pred, b_targ, b_group_id, b_mol_id, "obj"
            )
            b_m_sim_obj = obj_sim_func(b_m_pred, b_m_targ)
            b_m_loss_obj = obj_loss_func(b_m_pred, b_m_targ)
            m_sim_obj.append(b_m_sim_obj)
            m_loss_obj.append(b_m_loss_obj)
            m_group_id.append(b_m_group_id)
            m_mol_id.append(b_m_mol_id)
            if auxiliary:
                # just doing cos, forget about the other ones
                b_pred = process_spec(unprocess_spec(b_pred, data_d["transform"]),"none","l2")
                b_targ = process_spec(unprocess_spec(b_targ, data_d["transform"]),"none","l2")
                b_m_pred, b_m_targ, b_m_mol_id, b_m_group_id = merge_group_func(
                    b_pred, b_targ, b_group_id, b_mol_id, "std"
                )
                b_m_sim_cos_std = cos_sim_func(b_m_pred, b_m_targ)
                m_sim_cos_std.append(b_m_sim_cos_std)
        m_group_id = th.cat(m_group_id,dim=0)
        m_mol_id = th.cat(m_mol_id,dim=0)
        sim_d["m_sim_obj"] = th.cat(m_sim_obj,dim=0)
        sim_d["m_loss_obj"] = th.cat(m_loss_obj,dim=0)
        sim_d["m_group_id"] = m_group_id
        sim_d["m_mol_id"] = m_mol_id
        if auxiliary:
            sim_d["m_sim_cos_std"] = th.cat(m_sim_cos_std,dim=0)
    # construct tables and compute metrics
    merged_flags = [False]
    if merge_group:
        merged_flags.append(True)
    groupby_mol_flags = [False]
    if groupby_mol:
        groupby_mol_flags.append(True)
    tables = []
    for sl in ["sim", "loss"]:
        for merged in merged_flags:
            keys, vals = [], []
            if merged:
                _mol_id = m_mol_id
                _group_id = m_group_id
                for k, v in sim_d.items():
                    if k.startswith(f"m_{sl}"):
                        keys.append(k[len(f"m_{sl}_"):])
                        vals.append(v)
            else:
                _mol_id = mol_id
                _group_id = group_id
                for k, v in sim_d.items():
                    if k.startswith(sl):
                        keys.append(k[len(f"{sl}_"):])
                        vals.append(v)
            # print(sl,keys)
            table = MetricTable(
                keys, vals, _mol_id, _group_id, prefix, loss=(
                    sl == "loss"), merged=merged)
            # compute all of the metrics
            for gm in groupby_mol_flags:
                table.compute(
                    compute_agg=compute_agg,
                    compute_hist=compute_hist,
                    groupby_mol=gm)
            tables.append(table)
    return tables


def run_val(
        step,
        epoch,
        model,
        dl_d,
        data_d,
        model_d,
        run_d,
        use_wandb):

    if not (dl_d["primary"]["val"] is None):
        # stuff related to device
        dev = th.device(run_d["device"])
        nb = run_d["non_blocking"]
        # validation
        model.eval()
        pred, targ, mol_id, group_id = [], [], [], []
        with th.no_grad():
            for b_idx, b in get_pbar(
                enumerate(
                    dl_d["primary"]["val"]), run_d, desc="> val", total=len(
                    dl_d["primary"]["val"])):
                b = data_to_device(b, dev, nb)
                b_pred = model(data=b, amp=run_d["amp"])["pred"]
                b_targ = b["spec"]
                b_mol_id = b["mol_id"]
                b_group_id = b["group_id"]
                pred.append(b_pred.detach().to("cpu", non_blocking=nb))
                targ.append(b_targ.detach().to("cpu", non_blocking=nb))
                mol_id.append(b_mol_id.detach().to("cpu", non_blocking=nb))
                group_id.append(b_group_id.detach().to("cpu", non_blocking=nb))
        pred = th.cat(pred, dim=0)
        targ = th.cat(targ, dim=0)
        mol_id = th.cat(mol_id, dim=0)
        group_id = th.cat(group_id, dim=0)
        tables = compute_metric_tables(
            pred, targ, mol_id, group_id, "val", 
            data_d, run_d,
            auxiliary=run_d["log_auxiliary"],
            merge_group=True,
            compute_agg=True,
            compute_hist=False,
            groupby_mol=True
        )
        # print("val",lsh_d.keys())
        out_d = {}
        for table in tables:
            out_d = dict(
                **out_d,
                **table.unload_cache(prefix=False, agg=True, hist=False),
                **table.export_val("obj")
            )
        stop_key = run_d["stop_key"]
        spec_loss_obj_mean = out_d["spec_loss_obj_mean"]
        mol_loss_obj_mean = out_d["mol_loss_obj_mean"]
        loss_mean = out_d[stop_key]
        print(f"> step {step}, epoch {epoch}: val, {stop_key}: {loss_mean:.4f}")
        log_d = {"epoch": epoch, "Epoch": epoch}
        for table in tables:
            for k, v in table.unload_cache(
                    agg=True, hist=run_d["save_media"]).items():
                log_d[k] = v
        if use_wandb:
            wandb.log(log_d, commit=False)
            wandb.log({}, commit=True)
        if run_d["print_stats"]:
            pprint(log_d)
    else:
        out_d = {run_d["stop_key"]: np.nan}
    return step, epoch, out_d


def run_track(
        step,
        epoch,
        model,
        dl_d,
        data_d,
        model_d,
        run_d,
        use_wandb,
        ds,
        score_d):

    # stuff related to device
    dev = th.device(run_d["device"])
    nb = run_d["non_blocking"]
    # set up loss func
    loss_func = get_loss_func(run_d["loss"], data_d["mz_bin_res"])
    sim_func = get_sim_func(run_d["sim"], data_d["mz_bin_res"])
    # tracking
    if run_d["save_media"] and run_d["num_track"] > 0:
        assert not (dl_d["primary"]["val"] is None)
        model.to(dev)
        model.eval()
        # get top/bottom k similarity
        topk, argtopk = th.topk(
            score_d["spec_sim_obj"], run_d["num_track"], largest=True)
        bottomk, argbottomk = th.topk(
            score_d["spec_sim_obj"], run_d["num_track"], largest=False)
        topk_str = "[" + \
            ",".join([f"{val:.2f}" for val in topk.tolist()]) + "]"
        bottomk_str = "[" + \
            ",".join([f"{val:.2f}" for val in bottomk.tolist()]) + "]"
        print(f"> tracking: topk = {topk_str} , bottomk = {bottomk_str}")
        val_idx = dl_d["primary"]["val"].dataset.indices
        track_dl_d = ds.get_track_dl(
            val_idx,
            num_rand_idx=run_d["num_track"],
            topk_idx=argtopk.numpy(),
            bottomk_idx=argbottomk.numpy()
        )
        for dl_type, dl in track_dl_d.items():
            for d_idx, d in enumerate(dl):
                d = data_to_device(d, dev, nb)
                pred = model(data=d, amp=run_d["amp"])["pred"]
                targ = d["spec"]
                loss = loss_func(pred, targ)
                sim = sim_func(pred, targ)
                smiles = d["smiles"][0]
                prec_mz_bin = d["prec_mz_bin"][0]
                if "cfm_mbs" in d:
                    cfm_mbs = d["cfm_mbs"][0]
                else:
                    cfm_mbs = None
                if "simple_mbs" in d:
                    simple_mbs = d["simple_mbs"][0]
                else:
                    simple_mbs = None
                targ = targ.cpu().detach().numpy()
                if run_d["pred_viz"]:
                    pred = pred.cpu().detach().numpy()
                else:
                    pred = np.zeros_like(targ)
                loss = loss.item()
                sim = sim.item()
                spec_id = d["spec_id"].item()
                title = f"spec_id = {spec_id}, {run_d['sim']}_sim = {sim:.4g}"
                plot_data = plot_spec(
                    targ, pred,
                    data_d["mz_max"],
                    data_d["mz_bin_res"],
                    loss_type=run_d["loss"],
                    loss=loss,
                    sim_type=run_d["sim"],
                    sim=sim,
                    prec_mz_bin=prec_mz_bin,
                    smiles=smiles,
                    cfm_mbs=cfm_mbs,
                    simple_mbs=simple_mbs,
                    plot_title=run_d["track_plot_title"],
                    custom_title=title,
                )
                if use_wandb:
                    dl_type = dl_type.rstrip("")
                    log_dict = {
                        "epoch": epoch,
                        "Epoch": epoch,
                        f"{dl_type}_{d_idx}": wandb.Image(plot_data)
                    }
                    wandb.log(log_dict, commit=False)
    return step, epoch, {}


def compute_cross_sims(data_d, run_d, dl_d, use_wandb):

    if not run_d["do_test"] or not use_wandb:
        return None
    test_dl = dl_d["primary"]["test"]
    sec_dl_d = dl_d["secondary"]
    ce_key = data_d["ce_key"]
    if len(sec_dl_d) == 0:
        return None
    # get all test spectra
    test_group_id, test_mol_id, test_prec_type, test_ce, test_spec = [], [], [], [], []
    for b_idx, b in enumerate(test_dl):
        test_group_id.extend(b["group_id"])
        test_mol_id.extend(b["mol_id"])
        test_prec_type.extend(b["prec_type"])
        test_ce.extend(b["col_energy"])
        test_spec.append(b["spec"])
    test_df = pd.DataFrame({
        "group_id": th.stack(test_group_id, dim=0).numpy(),
        "mol_id": th.stack(test_mol_id, dim=0).numpy(),
        "prec_type": test_prec_type,
        "col_energy": test_ce
    })
    test_df.loc[:, "idx"] = np.arange(test_df.shape[0])
    test_m_df = test_df.drop(
        columns=[
            "idx", "col_energy"]).drop_duplicates(
        subset=["group_id"]).sort_values("group_id")
    test_m_df.loc[:, "idx"] = np.arange(test_m_df.shape[0])
    test_spec = th.cat(test_spec, dim=0)
    test_spec = unprocess_spec(test_spec, data_d["transform"])
    test_spec = process_spec(test_spec, "none", "l1")
    test_m_spec, _ = merge_spec(test_spec, th.as_tensor(
        test_df["group_id"].to_numpy()), "none", "l1")
    sim_func = get_sim_func("cos", data_d["mz_bin_res"])
    log_d = {}
    for sec_key, sec_dl in sec_dl_d.items():
        # get all spectra
        sec_group_id, sec_mol_id, sec_prec_type, sec_ce, sec_spec = [], [], [], [], []
        for b_idx, b in enumerate(sec_dl):
            sec_group_id.extend(b["group_id"])
            sec_mol_id.extend(b["mol_id"])
            sec_prec_type.extend(b["prec_type"])
            sec_ce.extend(b["col_energy"])
            sec_spec.append(b["spec"])
        sec_df = pd.DataFrame({
            "group_id": th.stack(sec_group_id, dim=0).numpy(),
            "mol_id": th.stack(sec_mol_id, dim=0).numpy(),
            "prec_type": sec_prec_type,
            "col_energy": sec_ce
        })
        sec_df.loc[:, "idx"] = np.arange(sec_df.shape[0])
        sec_m_df = sec_df.drop(
            columns=[
                "idx", "col_energy"]).drop_duplicates(
            subset=["group_id"]).sort_values("group_id")
        sec_m_df.loc[:, "idx"] = np.arange(sec_m_df.shape[0])
        sec_spec = th.cat(sec_spec, dim=0)
        sec_spec = unprocess_spec(sec_spec, data_d["transform"])
        sec_spec = process_spec(sec_spec, "none", "l1")
        sec_m_spec, _ = merge_spec(sec_spec, th.as_tensor(
            sec_df["group_id"].to_numpy()), "none", "l1")
        # find the identical spectra
        both_df = test_df.merge(
            sec_df, on=[
                "mol_id", "prec_type", "col_energy"], how="inner")
        both_test_spec = test_spec[th.as_tensor(both_df["idx_x"].to_numpy())]
        both_sec_spec = sec_spec[th.as_tensor(both_df["idx_y"].to_numpy())]
        both_m_df = test_m_df.merge(
            sec_m_df, on=[
                "mol_id", "prec_type"], how="inner")
        both_test_m_spec = test_m_spec[th.as_tensor(
            both_m_df["idx_x"].to_numpy())]
        both_sec_m_spec = sec_m_spec[th.as_tensor(
            both_m_df["idx_y"].to_numpy())]
        # compute similarity
        sim = sim_func(both_test_spec, both_sec_spec)
        m_sim = sim_func(both_test_m_spec, both_sec_m_spec)
        log_d[f"test_{sec_key}_cross_sim"] = th.mean(sim)
        log_d[f"test_{sec_key}_cross_m_sim"] = th.mean(m_sim)
    if use_wandb:
        wandb.log(log_d, commit=False)
    if run_d["print_stats"]:
        pprint(log_d)
    return None


def run_test(
        step,
        epoch,
        model,
        dl_d,
        data_d,
        model_d,
        run_d,
        use_wandb,
        run_dir,
        test_sets=None):

    if test_sets is None:
        test_sets = ["test"]
    if run_d["do_test"]:
        # stuff related to device
        dev = th.device(run_d["device"])
        nb = run_d["non_blocking"]
        print(">> test")
        # model setup
        model.to(dev)
        model.eval()
        out_d, save_tables = {}, []
        for order in ["primary", "secondary"]:
            out_d[order] = {}
            for dl_key, dl in dl_d[order].items():
                if not (dl_key in test_sets) or dl is None:
                    continue
                pred, targ, mol_id, group_id = [], [], [], []
                with th.no_grad():
                    for b_idx, b in get_pbar(
                            enumerate(dl), run_d, desc=f"> {dl_key}", total=len(dl)):
                        b = data_to_device(b, dev, nb)
                        b_pred = model(data=b, amp=run_d["amp"])["pred"]
                        b_targ = b["spec"]
                        b_mol_id = b["mol_id"]
                        b_group_id = b["group_id"]
                        pred.append(b_pred.detach().to("cpu", non_blocking=nb))
                        targ.append(b_targ.detach().to("cpu", non_blocking=nb))
                        mol_id.append(
                            b_mol_id.detach().to(
                                "cpu", non_blocking=nb))
                        group_id.append(
                            b_group_id.detach().to(
                                "cpu", non_blocking=nb))
                pred = th.cat(pred, dim=0)
                targ = th.cat(targ, dim=0)
                mol_id = th.cat(mol_id, dim=0)
                group_id = th.cat(group_id, dim=0)
                tables = compute_metric_tables(
                    pred, targ, mol_id, group_id, dl_key,
                    data_d, run_d,
                    auxiliary=run_d["log_auxiliary"],
                    merge_group=True,
                    compute_agg=True,
                    compute_hist=run_d["save_media"],
                    groupby_mol=True
                )
                _out_d = {}
                for table in tables:
                    _out_d = dict(
                        **_out_d, **table.unload_cache(prefix=False, agg=True, hist=False))
                stop_key = run_d["stop_key"]
                spec_loss_obj_mean = _out_d["spec_loss_obj_mean"]
                mol_loss_obj_mean = _out_d["mol_loss_obj_mean"]
                loss_mean = _out_d[stop_key]
                print(
                    f"> {dl_key}, {stop_key} = {loss_mean:.4}")
                out_d[order] = _out_d
                log_d = {"epoch": epoch, "Epoch": epoch}
                for table in tables:
                    for k, v in table.unload_cache(
                            hist=run_d["save_media"]).items():
                        log_d[k] = v
                if use_wandb:
                    wandb.log(log_d, commit=False)
                if run_d["print_stats"]:
                    pprint(log_d)
                if run_d["save_test_sims"]:
                    # save the tables
                    save_tables.extend(tables)
        if run_d["save_test_sims"]:
            save_dp = os.path.join(run_dir, "save_tables")
            os.makedirs(save_dp, exist_ok=True)
            for table in save_tables:
                save_str = table.get_table_str()
                save_fp = os.path.join(save_dp, save_str)
                table.save(save_fp)
                if use_wandb:
                    wandb.save(save_fp, base_path=run_dir)
        if use_wandb:
            wandb.log({}, commit=True)
    else:
        out_d = {}
    return step, epoch, out_d


def rank_metrics(rank, total):

    d = {}
    d["rank"] = float(rank)
    d["top01"] = float(rank == 1)
    d["top05"] = float(rank <= 5)
    d["top10"] = float(rank <= 10)
    # d["ndcg"] = 1. / np.log2(float(rank) + 1.)
    norm_rank = float((rank - 1) / total)
    d["norm_rank"] = norm_rank
    d["top01%"] = float(norm_rank <= 0.01)
    d["top05%"] = float(norm_rank <= 0.05)
    d["top10%"] = float(norm_rank <= 0.10)
    d["total"] = total
    return d


def sims_to_rank_metrics(sim, sim2, key_prefix, cand_match_mask):
    # note: with binary relevance, and a single relevant item, NDCG is
    # equivalent to 1/log2(rank)

    rm_d = {}
    key = f"{key_prefix}"
    cand_match_idx = th.argmax(cand_match_mask.float())
    rm_d[f"{key}_sim"] = sim[cand_match_idx].item()
    noisey_sim = sim + 0.00001 * th.rand_like(sim)
    sim_argsorted = th.argsort(-noisey_sim, dim=0)
    rank = th.argmax(cand_match_mask.float()[sim_argsorted]) + 1
    _rm_d = rank_metrics(rank, cand_match_mask.shape[0])
    rm_d.update({f"{key}_{k}":v for k,v in _rm_d.items()})
    # molecule sims
    rm_d[f"{key}_sim2"] = sim2[cand_match_idx].item() # this should be ~1.0
    noisey_sim2 = sim2 + 0.00001 * th.rand_like(sim2)
    sim2_argsorted = th.argsort(-noisey_sim2, dim=0)
    # top 20%
    num_20p = int(np.round(0.2*sim2_argsorted.shape[0]))
    sim2_t20p = sim2_argsorted[:num_20p]
    if cand_match_idx not in sim2_t20p:
        sim2_t20p = th.cat([cand_match_idx.reshape(1,),sim2_t20p[1:]],dim=0)
    sim_t20p_argsorted = th.argsort(-noisey_sim[sim2_t20p], dim=0)
    rank_t20p = th.argmax(cand_match_mask.float()[sim2_t20p][sim_t20p_argsorted]) + 1
    total_t20p = sim_t20p_argsorted.shape[0]
    _rm_d = rank_metrics(rank_t20p, total_t20p)
    rm_d.update({f"{key}_t20p_{k}":v for k,v in _rm_d.items()})
    # bottom 20%
    sim2_b20p = sim2_argsorted[-num_20p:]
    if cand_match_idx not in sim2_b20p:
        sim2_b20p = th.cat([cand_match_idx.reshape(1,),sim2_b20p[1:]],dim=0)
    sim_b20p_argsorted = th.argsort(-noisey_sim[sim2_b20p], dim=0)
    rank_b20p = th.argmax(cand_match_mask.float()[sim2_b20p][sim_b20p_argsorted]) + 1
    total_b20p = sim_b20p_argsorted.shape[0]
    _rm_d = rank_metrics(rank_b20p, total_b20p)
    rm_d.update({f"{key}_b20p_{k}":v for k,v in _rm_d.items()})
    return rm_d


def run_casmi(
        step,
        epoch,
        model,
        casmi_ds,
        casmi_type,
        data_d,
        model_d,
        run_d,
        use_wandb,
        run_dir,
        mr_d,
        update_mr_d):

    assert not model_d["cfm_postprocessing"]
    assert casmi_type in ["casmi", "pcasmi","casmi22"]
    if run_d[f"do_{casmi_type}"]:
        pred_all = run_d[f"{casmi_type}_pred_all"]
        casmi_d = mr_d[f"{casmi_type}_d"]
        print(f">> {casmi_type}")
        # stuff related to device
        dev = th.device(run_d["device"])
        nb = run_d["non_blocking"]
        # model setup
        model.to(dev)
        model.eval()
        model.set_mode(casmi_type)
        # get dataloaders
        spec_dl = casmi_ds.get_dataloader(run_d, "spec")
        # get ground truth spectra, merge across collision energies
        spec, spec_spec_id, spec_group_id, spec_mol_id, spec_casmi_fp = [], [], [], [], []
        for b_idx, b in get_pbar(
                enumerate(spec_dl), run_d, desc=f"> spec", total=len(spec_dl)):
            spec.append(b["spec"])
            spec_spec_id.append(b["spec_id"])
            spec_group_id.append(b["group_id"])
            spec_mol_id.append(b["mol_id"])
            spec_casmi_fp.append(b["casmi_fp"])
        spec = th.cat(spec, dim=0)
        spec_spec_id = th.cat(spec_spec_id, dim=0)
        spec_group_id = th.cat(spec_group_id, dim=0)
        spec_mol_id = th.cat(spec_mol_id, dim=0)
        spec_casmi_fp = th.cat(spec_casmi_fp, dim=0)
        # merge spectra across CE
        spec_merge, spec_group_id_merge, spec_mol_id_merge, spec_casmi_fp_merge = merge_spec(
            spec, spec_group_id, casmi_ds.transform, casmi_ds.spectrum_normalization, spec_mol_id, spec_casmi_fp)
        # if not verify_merge(spec_group_id,spec_casmi_fp,spec_casmi_fp_merge):
        #     import pdb; pdb.set_trace()
        if pred_all:
            # get predicted spectra
            group_dl = casmi_ds.get_dataloader(run_d, "group")
            cand_pred, cand_group_id, cand_mol_id, cand_spec_id, cand_casmi_fp = [], [], [], [], []
            with th.no_grad():
                for b_idx, b in get_pbar(
                        enumerate(group_dl), run_d, desc=f"> group all", total=len(group_dl)):
                    b = data_to_device(b, dev, nb)
                    b_group_id = b["group_id"]
                    b_mol_id = b["mol_id"]
                    b_spec_id = b["spec_id"]
                    b_casmi_fp = b["casmi_fp"]
                    b_pred = model(data=b, amp=run_d["amp"])["pred"]
                    cand_pred.append(b_pred.cpu())
                    cand_group_id.append(b_group_id.cpu())
                    cand_mol_id.append(b_mol_id.cpu())
                    cand_spec_id.append(b_spec_id.cpu())
                    cand_casmi_fp.append(b_casmi_fp.cpu())
            cand_pred_all = th.cat(cand_pred, dim=0)
            cand_group_id_all = th.cat(cand_group_id, dim=0)
            cand_mol_id_all = th.cat(cand_mol_id, dim=0)
            cand_spec_id_all = th.cat(cand_spec_id, dim=0)
            cand_casmi_fp_all = th.cat(cand_casmi_fp, dim=0)
            assert th.isin(spec_mol_id, cand_mol_id_all).all()
            assert th.isin(spec_mol_id_merge, cand_mol_id_all).all()
        # find the matches
        rm_ds, um_rm_ds = casmi_d["rm_ds"], casmi_d["um_rm_ds"]
        sims, sims2, group_ids = casmi_d["sims"], casmi_d["sims2"], casmi_d["group_ids"]
        sim_func = get_sim_func(run_d["sim"], data_d["mz_bin_res"])
        fp_sim_func = get_sim_func("jacc", None)
        for i in range(spec_group_id_merge.shape[0]):
            query_group_id = int(spec_group_id_merge[i])
            query_mol_id = int(spec_mol_id_merge[i])
            if query_group_id in casmi_d["query_group_ids"]:
                continue
            if pred_all:
                # merge predicted spectra
                cand_group_mask = th.as_tensor(
                    np.isin(
                        cand_group_id_all.numpy(),
                        query_group_id),
                    dtype=th.bool)  # num_preds
                cand_pred = cand_pred_all[cand_group_mask]
                cand_group_id = cand_group_id_all[cand_group_mask]
                cand_mol_id = cand_mol_id_all[cand_group_mask]
                cand_spec_id = cand_spec_id_all[cand_group_mask]
                cand_casmi_fp = cand_casmi_fp_all[cand_group_mask]
                cand_pred_merge, cand_mol_id_merge, cand_casmi_fp_merge = merge_spec(
                    cand_pred, cand_mol_id, casmi_ds.transform, casmi_ds.spectrum_normalization,
                    cand_casmi_fp)
            else:
                # get predicted spectra, then merge
                group_dl = casmi_ds.get_dataloader(
                    run_d, "group", group_id=query_group_id)
                cand_pred, cand_mol_id, cand_spec_id, cand_casmi_fp = [], [], [], []
                with th.no_grad():
                    for b_idx, b in get_pbar(enumerate(
                            group_dl), run_d, desc=f"> group {i+1} / {spec_group_id_merge.shape[0]}", total=len(group_dl)):
                        b = data_to_device(b, dev, nb)
                        b_mol_id = b["mol_id"]
                        b_spec_id = b["spec_id"]
                        b_casmi_fp = b["casmi_fp"]
                        b_pred = model(data=b, amp=run_d["amp"])["pred"]
                        cand_pred.append(b_pred.cpu())
                        cand_mol_id.append(b_mol_id.cpu())
                        cand_spec_id.append(b_spec_id.cpu())
                        cand_casmi_fp.append(b_casmi_fp.cpu())
                cand_pred = th.cat(cand_pred, dim=0)
                cand_mol_id = th.cat(cand_mol_id, dim=0)
                # cand_spec_id only used for unmerged stats
                cand_spec_id = th.cat(cand_spec_id, dim=0)
                cand_casmi_fp = th.cat(cand_casmi_fp, dim=0)
                cand_pred_merge, cand_mol_id_merge, cand_casmi_fp_merge = merge_spec(
                    cand_pred, cand_mol_id, casmi_ds.transform, casmi_ds.spectrum_normalization,
                    cand_casmi_fp)
            # first, do merged
            cand_match_mask = th.as_tensor(
                cand_mol_id_merge == query_mol_id,
                dtype=th.bool)  # num_cands
            assert cand_match_mask.sum() == 1, cand_match_mask.sum()
            cand_spec = cand_pred_merge
            targ_spec = spec_merge[i].unsqueeze(
                0).expand(cand_spec.shape[0], -1)
            sim_obj = sim_func(cand_spec, targ_spec)
            cand_fp = cand_casmi_fp_merge
            targ_fp = spec_casmi_fp_merge[i].unsqueeze(
                0).expand(cand_fp.shape[0], -1)
            sim_fp = fp_sim_func(cand_fp, targ_fp)
            if run_d["casmi_save_sim"]:
                sims.append(sim_obj)
                sims2.append(sim_fp)
                group_ids.append(query_group_id)
            rm_d = sims_to_rank_metrics(sim_obj, sim_fp, casmi_type, cand_match_mask)
            rm_ds.append(rm_d)
            # then, do unmerged
            if casmi_type == "pcasmi" and run_d["log_pcasmi_um"]:
                group_um_rm_ds = []
                un_cand_spec_id = th.unique(cand_spec_id)
                spec_per_mol = un_cand_spec_id.shape[0]
                for spec_id in un_cand_spec_id:
                    spec_mask = th.as_tensor(
                        cand_spec_id == spec_id, dtype=th.bool)
                    assert spec_mask.sum() == (
                        cand_spec_id.shape[0] // spec_per_mol)
                    cand_match_mask = th.as_tensor(
                        cand_mol_id[spec_mask] == query_mol_id,
                        dtype=th.bool)  # num_cands
                    assert cand_match_mask.sum() == 1
                    cand_spec = cand_pred[spec_mask]
                    targ_spec = spec_merge[i].unsqueeze(
                        0).expand(cand_spec.shape[0], -1)
                    group_um_sim_obj = sim_func(cand_spec, targ_spec)
                    cand_fp = cand_casmi_fp[spec_mask]
                    targ_fp = spec_casmi_fp_merge[i].unsqueeze(
                        0).expand(cand_fp.shape[0], -1)
                    group_um_sim_fp = fp_sim_func(cand_fp, targ_fp)
                    group_um_rm_d = sims_to_rank_metrics(
                        group_um_sim_obj, group_um_sim_fp, casmi_type + "_um", cand_match_mask)
                    group_um_rm_ds.append(group_um_rm_d)
                # average over all members in the group
                um_rm_d = {k: np.mean(
                    np.array([d[k] for d in group_um_rm_ds])) for k in group_um_rm_ds[0]}
                um_rm_ds.append(um_rm_d)
            # PREEMPT: add query_group_id to completed set
            casmi_d["query_group_ids"].add(query_group_id)
            # PREEMPT: update mr_d
            update_mr_d(mr_d,**{f"{casmi_type}_d": casmi_d})
        rm_d = {k: np.array([d[k] for d in rm_ds]) for k in rm_ds[0]}
        if len(um_rm_ds) > 0:
            um_rm_d = {k: np.array([d[k] for d in um_rm_ds])
                       for k in um_rm_ds[0]}
        else:
            um_rm_d = {}
        log_dict = {}
        for k, v in rm_d.items():
            log_dict[k] = np.mean(v)
        for k, v in um_rm_d.items():
            log_dict[k] = np.mean(v)
        # setup histograms
        num_cands_all = th.tensor([_sims.shape[0] for _sims in sims])
        weights = th.repeat_interleave(
            F.normalize(num_cands_all.float(),p=1,dim=0)*(1./num_cands_all.float()),
            num_cands_all,
            dim=0
        ).cpu().numpy()
        sims_all = th.cat(sims,dim=0).cpu().numpy()
        sims_hist = plot_sim_hist(sims_all, sim_type="cosine", weights=weights)
        sim2s_all = th.cat(sims2,dim=0).cpu().numpy()
        sim2s_hist = plot_sim_hist(sim2s_all, sim_type="tanimoto", weights=weights)
        if use_wandb:
            log_dict[f"{casmi_type}_spec_sim_hist"] = wandb.Image(sims_hist)
            log_dict[f"{casmi_type}_fp_sim_hist"] = wandb.Image(sim2s_hist)
        if use_wandb:
            wandb.log(log_dict, commit=False)
            if run_d["casmi_save_sim"]:
                save_d = {
                    "group_ids": group_ids,
                    "sims": sims,
                    "sims2": sims2,
                    "rm_d": rm_d,
                    "um_rm_d": um_rm_d}
                save_fp = os.path.join(run_dir, f"{casmi_type}_sims.pkl")
                th.save(save_d, save_fp)
                wandb.save(f"{casmi_type}_sims.pkl")
            wandb.log({}, commit=True)
        if run_d["print_stats"]:
            pprint(log_dict)
    return step, epoch


def get_ds_model(data_d, model_d, run_d):

    with th_temp_seed(model_d["model_seed"]):
        embed_types = model_d["embed_types"]
        dset_types = get_dset_types(embed_types)
        assert len(dset_types) > 0, dset_types
        ds = BaseDataset(*dset_types, **data_d)
        ds.compute_lda(run_d)
        dim_d = ds.get_data_dims()
        if model_d["cfm_model"]:
            assert run_d["device"] == "cpu"
            assert not model_d["esp_model"]
            model = CFMPredictor(
                data_d["cfm_dp"],
                ds,
                run_d["do_casmi"],
                run_d["do_pcasmi"],
                run_d["do_casmi22"],
                use_rb=model_d["cfm_rb"]
            )
        elif model_d["esp_model"]:
            assert not model_d["cfm_model"]
            assert "esp" in dset_types, dset_types
            model = ESPPredictor(
                dim_d["o_dim"],
                len(ELEMENT_LIST)+1,
                7,
                len(ds.prec_type_c2i)+1,
                model_d["esp_new_reverse"],
                model_d["prec_mass_offset"]
            )
        else:
            model = Predictor(dim_d, **model_d)
        if run_d["do_casmi"]:
            casmi_ds = CASMIDataset(ds, "casmi", *dset_types, **data_d)
            ds.update_casmi_info(casmi_ds)
        else:
            casmi_ds = None
        if run_d["do_pcasmi"]:
            pcasmi_ds = CASMIDataset(ds, "pcasmi", *dset_types, **data_d)
            ds.update_casmi_info(pcasmi_ds)
        else:
            pcasmi_ds = None
        if run_d["do_casmi22"]:
            casmi22_ds = CASMIDataset(ds, "casmi22", *dset_types, **data_d)
            ds.update_casmi_info(casmi22_ds)
        else:
            casmi22_ds = None
    dev = th.device(run_d["device"])
    model.to(dev)
    return ds, model, casmi_ds, pcasmi_ds, casmi22_ds


def init_casmi_d():
    d = {}
    d["query_group_ids"] = set()
    for k in ["rm_ds","um_rm_ds","sims","sims2","group_ids"]:
        d[k] = []
    return d


def train_and_eval(data_d, model_d, run_d, use_wandb):

    if use_wandb:
        run_dir = wandb.run.dir
    else:
        tmp_dir = tempfile.TemporaryDirectory()
        run_dir = tmp_dir.name

    # set seeds
    th.manual_seed(run_d["train_seed"])
    np.random.seed(run_d["train_seed"] // 2)

    # set parallel strategy
    if run_d["parallel_strategy"] == "fd":
        parallel_strategy = "file_descriptor"
    else:
        parallel_strategy = "file_system"
    th.multiprocessing.set_sharing_strategy(parallel_strategy)

    # set determinism (this seems to only affect CNN)
    if run_d["cuda_deterministic"]:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    th.use_deterministic_algorithms(run_d["cuda_deterministic"])

    # load dataset, set up model
    ds, model, casmi_ds, pcasmi_ds, casmi22_ds = get_ds_model(data_d, model_d, run_d)
    num_params = count_parameters(model, requires_grad=False)
    mol_embed_params, mlp_params, total_params = model.count_parameters()
    assert num_params == total_params, (num_params, total_params)
    print(f">>> mol_embed_params = {mol_embed_params}, mlp_params = {mlp_params}, total_params = {total_params}")

    if run_d["dp"]:
        assert run_d["device"] == "cuda:0"
        assert run_d["dp_num_gpus"] > 1
        model = th.nn.DataParallel(
            model, device_ids=[
                i for i in range(
                    run_d["dp_num_gpus"])])

    # set up dataloader
    dl_d, split_id_d = ds.get_dataloaders(run_d)

    # set up optimizer
    if run_d["optimizer"] == "adam":
        optimizer_fn = th.optim.Adam
    elif run_d["optimizer"] == "adamw":
        optimizer_fn = th.optim.AdamW
    else:
        raise NotImplementedError
    if run_d["pt_weight_decay"] == -1.:
        # use the same amount of weight decay for everything
        optimizer = optimizer_fn(
            model.parameters(),
            lr=run_d["learning_rate"],
            weight_decay=run_d["weight_decay"])
    else:
        # use different weight decay for pretrained part
        # this only works for pretrained models
        if run_d["dp"]:
            # dataparallel
            nopt_params, pt_params = model.module.get_split_params()
        else:
            nopt_params, pt_params = model.get_split_params()
        optimizer = optimizer_fn(
            [
                {"params": nopt_params, "weight_decay": run_d["weight_decay"]},
                {"params": pt_params, "weight_decay": run_d["pt_weight_decay"]}
            ],
            lr=run_d["learning_rate"]
        )

    # set up scheduler
    if run_d["scheduler"] == "step":
        scheduler = th.optim.lr_scheduler.StepLR(
            optimizer,
            run_d["scheduler_period"],
            gamma=run_d["scheduler_ratio"]
        )
    elif run_d["scheduler"] == "plateau":
        scheduler = th.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            patience=run_d["scheduler_period"],
            factor=run_d["scheduler_ratio"]
        )
    elif run_d["scheduler"] == "polynomial":
        # special kind of training for Graphormer
        # note: ignores the learning rate/weight decay stuff passed to
        # optimizer
        if run_d["num_decay_epochs"] == -1:
            num_decay_epochs = run_d["num_epochs"]
        else:
            num_decay_epochs = run_d["num_decay_epochs"]
            if run_d["num_decay_epochs"] > run_d["num_epochs"]:
                print(
                    f">>> WARNING: num_decay_epochs ({run_d['num_decay_epochs']}) > num_epochs ({run_d['num_epochs']})")
        if dl_d["primary"]["train"] is None:
            num_batches = 0
        else:
            num_batches = len(dl_d["primary"]["train"])
        tot_updates = num_decay_epochs * \
            (num_batches // run_d["grad_acc_interval"])
        warmup_updates = int(run_d["scheduler_warmup_frac"] * tot_updates)
        peak_lr = run_d["scheduler_peak_lr"]
        end_lr = run_d["scheduler_end_lr"]
        scheduler = PolynomialDecayLR(
            optimizer,
            warmup_updates=warmup_updates,  # warmup
            tot_updates=tot_updates,  # total
            lr=peak_lr,  # high
            end_lr=end_lr,  # low
            power=run_d["scheduler_power"]  # power
        )
    elif run_d["scheduler"] == "none":
        scheduler = th.optim.lr_scheduler.StepLR(
            optimizer,
            1,
            gamma=1.0
        )
    else:
        raise NotImplementedError

    # load saved model from checkpoint
    if model_d["checkpoint_name"] is not None:
        chkpt_fp = os.path.join(
            data_d["checkpoint_dp"],
            model_d["checkpoint_name"] + ".pkl")
        chkpt_d = th.load(chkpt_fp,map_location="cpu")
        model.load_state_dict(chkpt_d["best_model_sd"])

    best_val_loss_mean = np.inf
    best_val_metrics = {}
    best_epoch = -1
    best_state_dict = copy.deepcopy(model.state_dict())
    early_stop_count = 0
    early_stop_thresh = run_d["early_stop_thresh"]
    step = 0
    epoch = -1
    casmi_d = init_casmi_d()
    pcasmi_d = init_casmi_d()
    casmi22_d = init_casmi_d()
    dev = th.device(run_d["device"])

    mr_fp = os.path.join(run_dir, "chkpt.pkl")
    temp_mr_fp = os.path.join(run_dir, "temp_chkpt.pkl")
    split_id_fp = os.path.join(run_dir, "split_id.pkl")
    if os.path.isfile(mr_fp):
        print(">>> reloading model from most recent checkpoint")
        mr_d = th.load(mr_fp,map_location="cpu")
        model.load_state_dict(mr_d["mr_model_sd"])
        best_state_dict = copy.deepcopy(model.state_dict())
        optimizer.load_state_dict(mr_d["optimizer_sd"])
        scheduler.load_state_dict(mr_d["scheduler_sd"])
        best_val_loss_mean = mr_d["best_val_loss_mean"]
        best_val_metrics = mr_d["best_val_metrics"]
        best_epoch = mr_d["best_epoch"]
        early_stop_count = mr_d["early_stop_count"]
        step = mr_d["step"]
        epoch = mr_d["epoch"]
        casmi_d = mr_d["casmi_d"]
        pcasmi_d = mr_d["pcasmi_d"]
        casmi22_d = mr_d["casmi22_d"]
    else:
        print(">>> no checkpoint detected")
        mr_d = {
            "mr_model_sd": model.state_dict(),
            "best_model_sd": best_state_dict,
            "optimizer_sd": optimizer.state_dict(),
            "scheduler_sd": scheduler.state_dict(),
            "best_val_loss_mean": best_val_loss_mean,
            "best_val_metrics": best_val_metrics,
            "best_epoch": best_epoch,
            "early_stop_count": early_stop_count,
            "step": step,
            "epoch": epoch,
            "test": False,
            "casmi": False,
            "pcasmi": False,
            "casmi22": False,
            "casmi_d": casmi_d,
            "pcasmi_d": pcasmi_d,
            "casmi22_d": casmi22_d
        }
        if run_d["save_split"]:
            # save data split
            th.save(split_id_d, split_id_fp)
            if use_wandb:
                wandb.save("split_id.pkl")
        if run_d["save_state"]:
            # save model state
            th.save(mr_d,temp_mr_fp)
            os.replace(temp_mr_fp,mr_fp)
            if use_wandb:
                wandb.save("chkpt.pkl")
    model.to(dev)

    epoch += 1

    while epoch < run_d["num_epochs"]:

        print(f">>> start epoch {epoch}")

        # training, single epoch
        step, epoch, train_d = run_train_epoch(
            step, epoch, model, dl_d, data_d, model_d, run_d, use_wandb, optimizer, scheduler)

        # validation
        step, epoch, val_d = run_val(
            step, epoch, model, dl_d, data_d, model_d, run_d, use_wandb)

        # update scheduler
        if run_d["scheduler"] == "step":
            scheduler.step()
        elif run_d["scheduler"] == "plateau":
            scheduler.step(val_d[run_d["stop_key"]])

        # tracking
        step, epoch, track_d = run_track(
            step, epoch, model, dl_d, data_d, model_d, run_d, use_wandb, ds, val_d)

        # early stopping
        val_loss_mean = val_d[run_d["stop_key"]]
        if best_val_loss_mean == np.inf:
            print(f"> val loss delta: N/A")
        else:
            print(f"> val loss delta: {val_loss_mean-best_val_loss_mean}")
        if run_d["use_val_info"]:
            if best_val_loss_mean < val_loss_mean:
                early_stop_count += 1
                print(
                    f"> val loss DID NOT decrease, early stop count at {early_stop_count}/{early_stop_thresh}")
            else:
                best_val_loss_mean = val_loss_mean
                best_val_metrics = {
                    k: v for k, v in val_d.items() if (
                        "_mean" in k)}
                best_epoch = epoch
                early_stop_count = 0
                # update state dicts
                model.to("cpu")
                best_state_dict = copy.deepcopy(model.state_dict())
                model.to(dev)
                print("> val loss DID decrease, early stop count reset")
            if early_stop_count == early_stop_thresh:
                print("> early stopping NOW")
                break
        else:
            # always assume the most recent epoch is the best
            best_val_loss_mean = val_loss_mean
            best_val_metrics = {
                k: v for k,
                v in val_d.items() if (
                    "_mean" in k)}
            best_epoch = epoch
            early_stop_count = 0
            # update state dicts
            model.to("cpu")
            best_state_dict = copy.deepcopy(model.state_dict())
            model.to(dev)

        # save model
        mr_d = {
            "mr_model_sd": model.state_dict(),
            "best_model_sd": best_state_dict,
            "optimizer_sd": optimizer.state_dict(),
            "scheduler_sd": scheduler.state_dict(),
            "best_val_loss_mean": best_val_loss_mean,
            "best_val_metrics": best_val_metrics,
            "best_epoch": best_epoch,
            "early_stop_count": early_stop_count,
            "step": step,
            "epoch": epoch,
            "test": False,
            "casmi": False,
            "pcasmi": False,
            "casmi22": False,
            "casmi_d": casmi_d,
            "pcasmi_d": pcasmi_d,
            "casmi22_d": casmi22_d
        }
        if run_d["save_state"]:
            th.save(mr_d, temp_mr_fp)
            os.replace(temp_mr_fp,mr_fp)
            if use_wandb:
                wandb.save("chkpt.pkl")
        if use_wandb:
            # sync wandb (after epoch is complete!)
            wandb.log({"commit": epoch}, commit=True)

        epoch += 1
    
    def update_mr_d(mr_d,**kwargs):

        for k, v in kwargs.items():
            mr_d[k] = v
        if run_d["save_state"]:
            th.save(mr_d, temp_mr_fp)
            os.replace(temp_mr_fp, mr_fp)
            if use_wandb:
                wandb.save("chkpt.pkl")

    # test
    if not mr_d["test"]:
        compute_cross_sims(data_d, run_d, dl_d, use_wandb)
        model.load_state_dict(best_state_dict)
        step, epoch, test_d = run_test(step, epoch, model, dl_d, data_d,
                                    model_d, run_d, use_wandb, run_dir, test_sets=run_d["test_sets"])
        update_mr_d(mr_d,test=True)

    # casmi
    if not mr_d["casmi"]:
        model.load_state_dict(best_state_dict)
        step, epoch = run_casmi(
            step, epoch, model, casmi_ds, "casmi", data_d, model_d, run_d, use_wandb, run_dir, mr_d, update_mr_d)
        update_mr_d(mr_d,casmi=True,casmi_d={})

    # pcasmi
    if not mr_d["pcasmi"]:
        model.load_state_dict(best_state_dict)
        step, epoch = run_casmi(
            step, epoch, model, pcasmi_ds, "pcasmi", data_d, model_d, run_d, use_wandb, run_dir, mr_d, update_mr_d)
        update_mr_d(mr_d,pcasmi=True,pcasmi_d={})

    # casmi22
    if not mr_d["casmi22"]:
        model.load_state_dict(best_state_dict)
        step, epoch = run_casmi(
            step, epoch, model, casmi22_ds, "casmi22", data_d, model_d, run_d, use_wandb, run_dir, mr_d, update_mr_d)
        update_mr_d(mr_d,casmi22=True,casmi22_d={})

    # final save, only include the best state (to reduce size of uploads)
    mr_d = {
        "best_model_sd": best_state_dict,
        "best_val_loss_mean": best_val_loss_mean,
        "best_val_metrics": best_val_metrics,
        "best_epoch": best_epoch,
        "epoch": epoch,
        "step": step,
        "test": True,
        "casmi": True,
        "pcasmi": True,
        "casmi22": True,
    }
    if run_d["save_state"]:
        th.save(mr_d, temp_mr_fp)
        os.replace(temp_mr_fp,mr_fp)
        if use_wandb:
            wandb.save("chkpt.pkl")
    if use_wandb:
        # metrics
        log_dict = {
            "best_val_loss_mean": best_val_loss_mean,
            "best_epoch": best_epoch
        }
        for k, v in best_val_metrics.items():
            log_k = f"best_{k}"
            log_dict[log_k] = v
        wandb.log(log_dict, commit=False)
        # sync wandb
        wandb.log({"commit": epoch}, commit=True)

    # get memory usage
    if run_d["device"] != "cpu" and th.cuda.is_available():
        cuda_max_memory = th.cuda.max_memory_allocated(device=dev)/1e9
        print(f"> GPU memory: {cuda_max_memory:.2f} GB")

    return


def load_config(template_fp, custom_fp, device_id, checkpoint_name):

    assert os.path.isfile(template_fp), template_fp
    if custom_fp:
        assert os.path.isfile(custom_fp), custom_fp

    with open(template_fp, "r") as template_file:
        config_d = yaml.load(template_file, Loader=yaml.FullLoader)

    # overwrite parts of the config
    if custom_fp:

        with open(custom_fp, "r") as custom_file:
            custom_d = yaml.load(custom_file, Loader=yaml.FullLoader)

        for k, v in custom_d.items():
            if k in ["entity_name","project_name"]:
                config_d[k] = v
            elif k == "run_name":
                if v is None:
                    config_d["run_name"] = os.path.splitext(os.path.basename(custom_fp))[0]
                else:
                    config_d["run_name"] = custom_d["run_name"]
            else:
                for k2, v2 in v.items():
                    config_d[k][k2] = v2

    entity_name = config_d["entity_name"]
    project_name = config_d["project_name"]
    run_name = config_d["run_name"]
    data_d = config_d["data"]
    model_d = config_d["model"]
    run_d = config_d["run"]

    # overwrite device if necessary
    if device_id:
        if device_id < 0:
            run_d["device"] = "cpu"
        else:
            run_d["device"] = f"cuda:{device_id}"

    # overwrite checkpoint if necessary
    if checkpoint_name:
        model_d["checkpoint_name"] = checkpoint_name

    return entity_name, project_name, run_name, data_d, model_d, run_d


def init_wandb_run(
        entity_name=None,
        project_name=None,
        run_name=None,
        data_d=None,
        model_d=None,
        run_d=None,
        wandb_meta_dp=None,
        group_name=None,
        wandb_mode=None,
        job_id=None,
        job_id_dp=None,
        wandb_symlink_dp=None,
        is_sweep=False):

    use_wandb = wandb_mode != "off"
    do_preempt = not (job_id is None)
    do_symlink = use_wandb and not (wandb_symlink_dp is None)
    # check for sweep
    if is_sweep:
        assert do_preempt
    # check for existing run
    if do_preempt:
        assert not job_id_dp is None
        job_id_fp = os.path.join(job_id_dp,f"{job_id}.yml")
        if os.path.isfile(job_id_fp):
            # resume existing run
            print(f">>> resuming {job_id}")
            with open(job_id_fp,"r") as file:
                job_dict = yaml.safe_load(file)
            run_id = job_dict["run_id"]	
            wandb_config = None
        else:
            # start new run
            print(f">>> starting {job_id}")
            run_id = None
            wandb_config = {**data_d,**model_d,**run_d}
        resume = "allow"
    else:
        # start new run
        run_id = None
        wandb_config = {**data_d, **model_d, **run_d}
        resume = "never"
    # init run
    run = wandb.init(
        entity=entity_name,
        project=project_name,
        name=run_name,
        config=wandb_config,
        dir=wandb_meta_dp,
        group=group_name,
        mode=wandb_mode,
        id=run_id,
        resume=resume
    )
    # update config
    run_config_d = dict(run.config)
    for d in [data_d, model_d, run_d]:
        for k in d.keys():
            d[k] = run_config_d[k]
    # symlink
    if do_symlink:
        symlink_dst = os.path.join(wandb_symlink_dp,str(job_id))
        symlink_src = os.path.split(os.path.abspath(run.dir))[0]
        if os.path.islink(symlink_dst):
            os.unlink(symlink_dst)
        assert not os.path.exists(symlink_dst), symlink_dst
        os.symlink(symlink_src,symlink_dst)
    # copy files and update job file
    if do_preempt:
        if not run_id is None:
            wandb.restore("chkpt.pkl",root=run.dir,replace=True)
            assert os.path.isfile(os.path.join(run.dir,"chkpt.pkl"))
        with open(job_id_fp,"w") as file:
            yaml.dump(dict(run_id=run.id), file)
    # train and/or eval model
    train_and_eval(data_d, model_d, run_d, True)
    # cleanup
    if do_preempt:
        # remove the job_id file
        os.remove(job_id_fp)
    if is_sweep:
        # overwrite old chkpt.pkl (to reduce memory usage)
        assert os.path.isfile(os.path.join(run.dir,"chkpt.pkl"))
        th.save(dict(),os.path.join(run.dir,"chkpt.pkl"))
    # end wandb
    run.finish()

