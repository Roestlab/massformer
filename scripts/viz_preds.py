import torch as th
import torch.utils.data as th_data
import torch_scatter as th_s
import os
import wandb
from pprint import pprint
import tempfile
import pandas as pd
import numpy as np
from tqdm import tqdm
import glob
import argparse
from distutils.util import strtobool

from massformer.losses import get_sim_func
from massformer.spec_utils import merge_spec, unprocess_spec, process_spec
from massformer.misc_utils import flatten_lol
from massformer.dataset import data_to_device
from massformer.runner import load_config, get_ds_model
from massformer.plot_utils import plot_spec, plot_combined_kde, plot_progression, get_mol_im
from massformer.het_sep_utils import *


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
    drop_prec_peak,
    merge,
    sim_func_name,
    mz_bin_res):

    # translate tranform
    if transform == "std":
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
    if drop_prec_peak:
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
        m_pred, m_group_id, m_mol_id, m_prec_mz_idx = merge_spec(
            pred, group_id, t, n, mol_id, prec_mz_idx)
        m_targ, _ = merge_spec(targ, group_id, t, n)
        pred, targ = m_pred, m_targ
        group_id, mol_id, prec_mz_idx = m_group_id, m_mol_id, m_prec_mz_idx
    # compute sim
    sim_func = get_sim_func(sim_func_name, mz_bin_res)
    sim = sim_func(pred, targ)
    return sim, group_id, mol_id, pred, targ, prec_mz_idx


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
            for drop_prec_peak in [False, True]:
                for merge in [False, True]:
                    key = f"{sim_func_name}_{transform}"
                    if drop_prec_peak:
                        key += "_np"
                    else:
                        key += "_p"
                    if merge:
                        key += "_m"
                    else:
                        key += "_um"
                    a_sim, a_group_id, a_mol_id, a_pred, a_targ, a_prec_mz_idx = calculate_sim(
                        pred, targ, prec_mz_idx, group_id, mol_id,
                        transform, drop_prec_peak, merge, sim_func_name, mz_bin_res)
                    if th.any(th.isnan(a_sim)):
                        print(f"> Warning: NaN sim value ({key})")
                    sim_d[key] = (a_sim, a_group_id, a_mol_id, a_pred, a_targ, a_prec_mz_idx)
    agg_sim_d = {}
    for k,v in sim_d.items():
        a_sim, a_group_id, a_mol_id, _, _, _ = v
        for groupby_mol in [False, True]:
            key = k
            if groupby_mol:
                key += "_mol"
                un_mol_id, un_mol_idx = th.unique(
                    a_mol_id, dim=0, return_inverse=True)
                a_agg_sim = th_s.scatter_mean(
                    a_sim,
                    un_mol_idx,
                    dim=0,
                    dim_size=un_mol_id.shape[0])
                a_agg_sim = th.mean(a_agg_sim,dim=0)
            else:
                key += "_spec"
                a_agg_sim = th.mean(a_sim,dim=0)
            agg_sim_d[key] = a_agg_sim
    return sim_d, agg_sim_d


def get_preds(ds, model, run_d):

    # setup device things
    dev = th.device(run_d["device"])
    nb = run_d["non_blocking"]

    # setup dataloader
    dl_d, _ = ds.get_dataloaders(run_d)
    dl = dl_d["primary"]["test"]

    with th.no_grad():
        pred, targ, prec_mz_idx, spec_id, mol_id, group_id, smiles, ce = [], [], [], [], [], [], [], []
        for b_idx, b in tqdm(enumerate(dl), total=len(dl), desc="> test"):
            b = data_to_device(b, dev, nb)
            b_pred = model(data=b, amp=run_d["amp"])["pred"]
            b_targ = b["spec"]
            b_prec_mz_idx = b["prec_mz_idx"]
            b_spec_id = b["spec_id"]
            b_mol_id = b["mol_id"]
            b_group_id = b["group_id"]
            b_smiles = b["smiles"]
            b_ce = b["col_energy"]
            pred.append(b_pred.detach().to("cpu", non_blocking=nb))
            targ.append(b_targ.detach().to("cpu", non_blocking=nb))
            prec_mz_idx.append(
            b_prec_mz_idx.detach().to(
                "cpu", non_blocking=nb))
            spec_id.append(
                b_spec_id.detach().to(
                    "cpu", non_blocking=nb)
            )
            mol_id.append(
                b_mol_id.detach().to(
                    "cpu", non_blocking=nb))
            group_id.append(
                b_group_id.detach().to(
                    "cpu", non_blocking=nb))
            smiles.append(b_smiles)
            ce.append(b_ce)
        pred = th.cat(pred, dim=0)
        targ = th.cat(targ, dim=0)
        prec_mz_idx = th.cat(prec_mz_idx, dim=0)
        spec_id = th.cat(spec_id, dim=0)
        mol_id = th.cat(mol_id, dim=0)
        group_id = th.cat(group_id, dim=0)
        smiles = flatten_lol(smiles)
        ce = flatten_lol(ce)
    ret_d = {}
    ret_d["pred"] = pred
    ret_d["targ"] = targ
    ret_d["prec_mz_idx"] = prec_mz_idx
    ret_d["spec_id"] = spec_id
    ret_d["mol_id"] = mol_id
    ret_d["group_id"] = group_id
    ret_d["smiles"] = smiles
    ret_d["ce"] = ce
    return ret_d


def get_ds_model_2(data_d, model_d, run_d, chkpt):

    # load model, datasets, dataloaders
    ds, model, _, _, _ = get_ds_model(data_d, model_d, run_d)

    # checkpoints
    chkpt_type = "wandb"

    # setup device things
    dev = th.device(run_d["device"])
    nb = run_d["non_blocking"]

    model = init_model_from_chkpt(chkpt, chkpt_type, model)
    model.to(dev)
    model.eval()

    return ds, model


def plot_examples(ds, model, pred_d, data_d, model_d, run_d, **kwargs):
    """ differs from other implementation """

    spec_df, mol_df = ds.spec_df, ds.mol_df
    m_spec_df = spec_df.drop(columns=["peaks"]).drop_duplicates(subset=["group_id"], keep="first")
    m_spec_df = m_spec_df.reset_index(drop=True).set_index("group_id", drop=False)
    smiles = pred_d["smiles"]

    sim_d, agg_sim_d = compute_metrics(
        pred=pred_d["pred"],
        targ=pred_d["targ"],
        prec_mz_idx=pred_d["prec_mz_idx"],
        mol_id=pred_d["mol_id"],
        group_id=pred_d["group_id"],
        sim_func_names=["cos"],
        transforms=["std"],
        mz_bin_res=data_d["mz_bin_res"]
    )

    sim_ranges = [
        (0.4,0.5),
        (0.5,0.6),
        (0.6,0.7),
        (0.7,0.8),
        (0.8,0.9),
        (0.9,1.0),
    ]
    num_samples = 40
    max_idx = ds.bin_func([1000.],[1.],return_index=True)[0]

    # clear the directory
    for fp in glob.glob("figs/example_preds/*"):
        os.remove(fp)

    meta_df_rows = []

    for sr_idx, sim_range in enumerate(sim_ranges):
        lower_limit, upper_limit = sim_range
        sim, group_id, mol_id, pred, targ, prec_mz_idx = sim_d["cos_std_p_m"]
        assert len(group_id) == len(th.unique(group_id))
        s_mask = (sim >= lower_limit) & (sim <= upper_limit) & (prec_mz_idx < max_idx)
        s_sim = sim[s_mask].numpy()
        s_group_id = group_id[s_mask].numpy()
        s_mol_id = mol_id[s_mask].numpy()
        s_smiles = [smiles[i] for i in range(len(s_mask)) if s_mask[i]]
        s_pred = unprocess_spec(pred[s_mask],"none").numpy()
        s_targ = unprocess_spec(targ[s_mask],"none").numpy()
        s_prec_mz_idx = prec_mz_idx[s_mask].numpy()
        s_num_samples = min(num_samples, len(s_group_id))
        sr_sample_idxs = np.random.choice(np.arange(len(s_group_id)), size=s_num_samples, replace=False)
        for idx, sr_sample_idx in tqdm(enumerate(sr_sample_idxs), total=s_num_samples, desc=f"> sr{sr_idx:02d} [{lower_limit},{upper_limit})"):
            sr_group_id = s_group_id[sr_sample_idx]
            sr_mol_id = s_mol_id[sr_sample_idx]
            sr_pred = s_pred[sr_sample_idx]
            sr_targ = s_targ[sr_sample_idx]
            sr_sim = s_sim[sr_sample_idx]
            sr_smiles = s_smiles[sr_sample_idx]
            sr_prec_mz_idx = s_prec_mz_idx[sr_sample_idx]
            sr_prec_type = m_spec_df.loc[sr_group_id]["prec_type"]
            plot_d = plot_spec(
                sr_targ, 
                sr_pred,
                1000.0,
                1.0,
                smiles=sr_smiles,
                plot_title=False,
                sim_type="cos_std",
                height_ratios=[1,1,1],
                size=9,
                mol_image_overlay=True,
                rescale_mz_axis=True,
                figsize=(5,4),
                include_legend=False,
                return_as_data=False,
                include_grid=False
            )
            plot_d["fig"].savefig(f"figs/example_preds/plot_data_sr{sr_idx:02d}_sample{idx:02d}_sim{sr_sim:.3f}.png",format="png",dpi=300)
            plot_d["fig"].savefig(f"figs/example_preds/plot_data_sr{sr_idx:02d}_sample{idx:02d}_sim{sr_sim:.3f}.pdf",format="pdf")
            plot_d["mol_im_png"].save(f"figs/example_preds/mol_im_sr{sr_idx:02d}_sample{idx:02d}_sim{sr_sim:.3f}.png")
            with open(f"figs/example_preds/mol_im_sr{sr_idx:02d}_sample{idx:02d}_sim{sr_sim:.3f}.svg","wb") as f:
                f.write(plot_d["mol_im_svg"])
            plt.close("all")
            meta_df_rows.append({
                "sr_idx": sr_idx,
                "sample_idx": idx,
                "group_id": sr_group_id,    
                "mol_id": sr_mol_id,
                "prec_type": sr_prec_type,
                "sim": sr_sim,
                "smiles": sr_smiles,
            })
    
    meta_df = pd.DataFrame(meta_df_rows)
    meta_df = meta_df.merge(mol_df[["mol_id","inchikey_s"]],on="mol_id",how="inner")
    meta_df = meta_df[["sr_idx","sample_idx","group_id","mol_id","prec_type","sim","inchikey_s","smiles"]]
    meta_df = meta_df.sort_values(by=["sr_idx","sample_idx"],ascending=True)
    meta_df.to_csv(f"figs/example_preds/meta_df.csv",index=False)


def plot_ce(ds, model, pred_d, data_d, model_d, run_d, plot_ce_mol_id, plot_ce_ces, **kwargs):

    # with th.no_grad():
    #     mzs = th.arange(
    #         data_d["mz_bin_res"],
    #         data_d["mz_max"] + data_d["mz_bin_res"],
    #         step=data_d["mz_bin_res"],
    #         requires_grad=False)
    #     mzs = mzs.unsqueeze(1).float()
    #     mzs = mzs - 0.5*data_d["mz_bin_res"]
    #     pred = process_spec(unprocess_spec(pred_d["pred"], data_d["transform"]), "none", "l1")
    #     targ = process_spec(unprocess_spec(pred_d["targ"], data_d["transform"]), "none", "l1")
    #     pred_mean_mzs = pred @ mzs
    #     targ_mean_mzs = targ @ mzs
    # pred_mean_mzs = pred_mean_mzs.flatten().numpy()
    # targ_mean_mzs = targ_mean_mzs.flatten().numpy()
    # ces = np.array(pred_d["ce"])
    # ymax = 300 # max(np.max(real_num_peaks),np.max(pred_num_peaks))
    # plot_d = plot_combined_kde(
    #     ces,
    #     targ_mean_mzs,
    #     ces,
    #     pred_mean_mzs,
    #     ymin=0,
    #     ymax=ymax,
    #     xlabel="Collision Energy (Normalized)",
    #     ylabel="Mean Mass/Charge (m/z)",
    #     size=36,
    #     title1="Real Distribution",
    #     title2="Simulated Distribution",
    #     return_as_data=False
    # )
    # plot_d["fig"].savefig("figs/ce/ce_density.png", format="png", dpi=300)
    # plot_d["fig"].savefig("figs/ce/ce_density.pdf", format="pdf")
    # plt.close("all")

    ce_ds = ds.get_subset([plot_ce_mol_id], key="mol_id")
    ce_dl = th_data.DataLoader(
        ce_ds,
        batch_size=100,
        collate_fn=ds.get_collate_fn(),
        num_workers=0,
        pin_memory=False,
        shuffle=False,
        drop_last=False
    )
    dev = th.device(run_d["device"])
    nb = run_d["non_blocking"]
    with th.no_grad():
        pred, targ, ce = [], [], []
        for b_idx, b in tqdm(enumerate(ce_dl), total=len(ce_dl), desc="> test"):
            b = data_to_device(b, dev, nb)
            b_pred = model(data=b, amp=run_d["amp"])["pred"]
            b_targ = b["spec"]
            b_ce = b["col_energy"]
            pred.append(b_pred.detach().to("cpu", non_blocking=nb))
            targ.append(b_targ.detach().to("cpu", non_blocking=nb))
            ce.append(b_ce)
        pred = th.cat(pred, dim=0)
        targ = th.cat(targ, dim=0)
        ce = flatten_lol(ce)
    pred = process_spec(unprocess_spec(pred, data_d["transform"]), "none", "none")
    targ = process_spec(unprocess_spec(targ, data_d["transform"]), "none", "none")
    ces = np.array(ce)
    plot_ce_ces = np.array(plot_ce_ces)
    ce_mask = np.isin(ces, plot_ce_ces)
    ce_as = np.argsort(plot_ce_ces)
    plot_ce_ces = plot_ce_ces[ce_as]
    preds = pred.cpu().numpy()[ce_mask][ce_as]
    targs = targ.cpu().numpy()[ce_mask][ce_as]
    preds = [x[0] for x in np.vsplit(preds,4)]
    targs = [x[0] for x in np.vsplit(targs,4)]
    plot_d = plot_progression(
        plot_ce_ces,
        preds,
        targs,
        mz_res=1.0,
        mz_min=80.,
        mz_max=190.,
        size=36,
        ints_max=1000.,
        include_legend=False,
        dims=[1,4]
    )
    plot_d["fig"].savefig(f"figs/ce/ce_progression_hl.pdf",bbox_inches="tight",format="pdf")
    # for subfigs_idx, subfig in enumerate(plot_d["subfigs"]):
    #     subfig.savefig(f"figs/ce/ce_progression_hl_{subfig_idx}.pdf",bbox_inches="tight",format="pdf")
    plt.close("all")
    # plot_d = plot_progression(
    #     plot_ce_ces,
    #     preds,
    #     targs,
    #     mz_res=1.0,
    #     mz_min=80.,
    #     mz_max=190.,
    #     size=36,
    #     ints_max=1000.,
    #     include_legend=False,
    #     dims=[2,2]
    # )
    # # plot_d["fig"].savefig(f"figs/ce/ce_progression_sq.png",format="png",dpi=300)
    # plot_d["fig"].savefig(f"figs/ce/ce_progression_sq.pdf",bbox_inches="tight",format="pdf")
    # for subfig_idx, subfig in enumerate(plot_d["subfigs"]):
    #     subfig.savefig(f"figs/ce/ce_progression_sq_{subfig_idx}.pdf",bbox_inches="tight",format="pdf")
    # plt.close("all")


def plot_hs_hist(
        ds, 
        model, 
        pred_d, 
        data_d, 
        model_d, 
        run_d,
        hs_min_num_peaks=10,
        hs_sim_thresh=0.8,
        hs_tree_thresh=0.8,
        hs_prec_type="[M+H]+",
        hs_heteroatoms=("N",),
        hs_num_target_spec=-1,
        hs_sirius_project_dp="_data/sirius/project",
        hs_drop_ambiguous=True,
        hs_overlap_thresh=5,
        hs_neg_thresh=2,
        hs_pos_thresh=2,
        hs_input_feats="atom",
        hs_num_random_bl=10,
        **kwargs):

    spec_id = pred_d["spec_id"]
    sim, group_id, mol_id, pred, targ, prec_mz_idx = calculate_sim(
        pred_d["pred"], 
        pred_d["targ"], 
        pred_d["prec_mz_idx"], 
        pred_d["group_id"], 
        pred_d["mol_id"],
        "std", 
        False,
        False,
        "cos",
        data_d["mz_bin_res"]
    )
    assert sim.shape[0] == spec_id.shape[0], (sim.shape, spec_id.shape)
    spec_ids = spec_id
    sims = sim
    for heteroatom in hs_heteroatoms:
        print(f">> heteroatom: {heteroatom}")
        targ_spec_id = select_mol(
            ds,run_d,spec_ids,sims,None,
            min_num_peaks=hs_min_num_peaks,
            sim_thresh=hs_sim_thresh,
            ce=None,
            prec_type=hs_prec_type,
            heteroatom=heteroatom,
            num_samples=-1
        )
        if hs_num_target_spec > 0:
            targ_spec_id = targ_spec_id[:hs_num_target_spec]
        print(f"> selected {len(targ_spec_id)} specs")
        # estimate the overlap
        required_spec_id = set(targ_spec_id.tolist())
        available_spec_id = set()
        for dp in glob.glob(os.path.join(hs_sirius_project_dp,"spec_*")):
            fn = os.path.splitext(os.path.basename(dp))[0]
            spec_id = int(fn.split("_")[1])
            available_spec_id.add(spec_id)
        print(f"> overlap: {len(required_spec_id&available_spec_id)}")
        targ_ds = ds.get_subset(targ_spec_id, key="spec_id")
        targ_dl = th_data.DataLoader(
            targ_ds,
            batch_size=1,
            collate_fn=ds.get_collate_fn(),
            num_workers=0,
            pin_memory=False,
            shuffle=False,
            drop_last=False
        )
        model.to("cpu")
        model.eval()
        sim_error, missing_error, tree_score_error, ambiguous_error, overlap_error, label_error = 0, 0, 0 ,0, 0, 0
        spec_data, tree_data = [], []
        for d_idx, d in tqdm(enumerate(targ_dl),total=len(targ_dl),desc="> metadata+sirius"):
            # prepare df_entry
            spec_id = int(d["spec_id"][0])
            spec_idx = np.argmax(spec_ids == spec_id)
            sim = float(sims[spec_idx])
            # remove if low similarity
            if sim < hs_sim_thresh:
                print("> Low Similarity")
                sim_error += 1
                continue
            prec_type = d["prec_type"][0]
            # get actual spec
            with th.no_grad():
                pred = model(d,return_input_feats=False)["pred"][0]
            targ = d["spec"][0]
            spec_bins = th.nonzero((pred>0.).float()*(targ>0.).float(),as_tuple=True)[0].numpy()
            cur_spec_data = {}
            cur_spec_data["data"] = d
            cur_spec_data["spec_id"] = spec_id
            cur_spec_data["pred"] = pred
            cur_spec_data["targ"] = targ
            cur_spec_data["bins"] = spec_bins
            cur_spec_data["smiles"] = d["smiles"][0]
            # get sirius data
            tree_fp = os.path.join(hs_sirius_project_dp,f"spec_{spec_id}.json")
            # remove if no tree
            if not os.path.isfile(tree_fp):
                missing_error += 1
                continue
            cur_tree_data = get_tree_data(tree_fp,heteroatom)
            # remove inaccurate trees
            if cur_tree_data["score"] < hs_tree_thresh:
                tree_score_error += 1
                continue
            # remove ambiguous trees
            if hs_drop_ambiguous and len(cur_tree_data["bins"]) != len(set(cur_tree_data["bins"])):
                ambiguous_error += 1
                continue
            # compute overlap with true bins
            both_mask = np.isin(cur_tree_data["bins"],spec_bins)
            both_tree_labels = cur_tree_data["labels"][both_mask]
            # remove if not enough overlap
            if np.sum(both_mask) < hs_overlap_thresh:
                overlap_error += 1
                continue
            # remove if there isn't enough of each label
            if np.sum(both_tree_labels==0) < hs_neg_thresh or np.sum(both_tree_labels==1) < hs_pos_thresh:
                label_error += 1
                continue
            spec_data.append(cur_spec_data)
            tree_data.append(cur_tree_data)
        print(f"> Errors: sim_error = {sim_error}, missing_error = {missing_error}, tree_score_error = {tree_score_error}, ambiguous_error = {ambiguous_error}, overlap_error = {overlap_error}, label_error = {label_error}")
        print(f"> Final Number of Spectra: {len(spec_data)}")
        assert len(spec_data) == len(tree_data), (len(spec_data),len(tree_data))
        num_peaks, num_zeros, num_ones, accs = [], [], [], []
        avg_dists, pos_avg_dists, neg_avg_dists = [], [], []
        accs_bl, seps_bl = [], []
        pc1_var_ratios, pc2_var_ratios = [], []
        for i in tqdm(range(len(spec_data)),total=len(spec_data),desc="> gi_embed"):
            spec_entry = spec_data[i]
            tree_entry = tree_data[i]
            data = spec_entry["data"]
            pred = spec_entry["pred"]
            spec_id = spec_entry["spec_id"]
            spec_bins = spec_entry["bins"]
            tree_bins = tree_entry["bins"]
            tree_labels = tree_entry["labels"]
            gis, gi_labels, _, _, _ = compute_gis(
                data,
                model,
                hs_input_feats,
                pred=pred,
                spec_bins=spec_bins,
                tree_bins=tree_bins,
                tree_labels=tree_labels
            )
            # fit PCA on all of the data
            gis_pca = F.normalize(th.as_tensor(gis.reshape(gis.shape[0],-1)),p=2,dim=1).numpy()
            pca = PCA(n_components=2)
            gis_pca = pca.fit_transform(gis_pca)
            pc1_var_ratios.append(pca.explained_variance_ratio_[0])
            pc2_var_ratios.append(pca.explained_variance_ratio_[1])
            # fit linear model on overlapping data
            assert not np.all(gi_labels==-1)
            both_mask = gi_labels!=-1
            both_gis_pca = gis_pca[both_mask]
            both_gi_labels = gi_labels[both_mask]
            logreg = LogisticRegression(penalty=NONE_PENALTY,random_state=0)
            logreg.fit(both_gis_pca,both_gi_labels)
            accuracy = logreg.score(both_gis_pca,both_gi_labels)
            def compute_avg_dist(emb,mask):
                if mask is None:
                    N = len(emb)
                else:
                    N = np.sum(mask.astype(int))
                    emb = emb[mask,:]
                dist_mat = np.sum((emb.reshape(N,1,-1)-emb.reshape(1,N,-1))**2,axis=2)
                return 0.5*np.sum(dist_mat)/(N*(N-1.))
            avg_dist = compute_avg_dist(both_gis_pca,None)
            pos_avg_dist = compute_avg_dist(both_gis_pca,both_gi_labels==1)
            neg_avg_dist = compute_avg_dist(both_gis_pca,both_gi_labels==0)
            # random baselines
            accuracy_bl = []
            for j in range(hs_num_random_bl):
                bl_gi_labels = np.copy(both_gi_labels)
                np.random.shuffle(bl_gi_labels)
                logreg_bl = LogisticRegression(penalty=NONE_PENALTY,random_state=0)
                logreg_bl.fit(both_gis_pca,bl_gi_labels)
                accuracy_bl.append(logreg_bl.score(both_gis_pca,bl_gi_labels))
            accuracy_bl = np.array(accuracy_bl)
            num_peaks.append(len(both_gi_labels))
            num_zeros.append(int(np.sum(both_gi_labels==0.)))
            num_ones.append(int(np.sum(both_gi_labels==1.)))
            accs.append(accuracy)
            avg_dists.append(avg_dist)
            pos_avg_dists.append(pos_avg_dist)
            neg_avg_dists.append(neg_avg_dist)
            accs_bl.append(accuracy_bl)
            seps_bl.append(accuracy_bl==1.)
        num_peaks = np.array(num_peaks)
        num_zeros = np.array(num_zeros)
        num_ones = np.array(num_ones)
        accs = np.array(accs)
        seps = accs==1.
        avg_dists = np.array(avg_dists)
        pos_avg_dists = np.array(pos_avg_dists)
        neg_avg_dists = np.array(neg_avg_dists)
        accs_bl = np.concatenate(accs_bl,axis=0)
        seps_bl = np.concatenate(seps_bl,axis=0)
        pc1_var_ratios = np.array(pc1_var_ratios)
        pc2_var_ratios = np.array(pc2_var_ratios)
        print(f"> Number of Peaks:  {np.mean(num_peaks)} +/- {np.std(num_peaks)}")
        print(f"> Fractions: Negative = {np.mean(num_zeros/num_peaks)}, Positive = {np.mean(num_ones/num_peaks)}")
        print(f"> Linear Separability: {np.mean(seps)}")
        print(f"> Linear Accuracy: {np.mean(accs)} +/- {np.std(accs)}")
        print(f"> All Embedding Distance: {np.mean(avg_dists)} +/- {np.std(avg_dists)}")
        print(f"> Pos Embedding Distance: {np.mean(pos_avg_dists)} +/- {np.std(pos_avg_dists)}")
        print(f"> Neg Embedding Distance: {np.mean(neg_avg_dists)} +/- {np.std(neg_avg_dists)}")
        print(f"> Baseline Linear Separability: {np.mean(seps_bl)}")
        print(f"> Baseline Linear Accuracy: {np.mean(accs_bl)} +/- {np.std(accs_bl)}")
        print(f"> PC1 Var Ratio: {np.mean(pc1_var_ratios)} +/- {np.std(pc1_var_ratios)}")
        print(f"> PC2 Var Ratio: {np.mean(pc2_var_ratios)} +/- {np.std(pc2_var_ratios)}")
        # make histogram
        log_d = {
            "num_peaks_mean": np.mean(num_peaks),
            "num_peaks_std": np.std(num_peaks),
            "neg_frac_mean": np.mean(num_zeros/num_peaks),
            "pos_frac_mean": np.mean(num_ones/num_peaks),
            "acc_mean": np.mean(accs),
            "acc_std": np.std(accs),
            "sep": np.mean(seps),
            "avg_dists_mean": np.mean(avg_dists),
            "pos_avg_dists_mean": np.mean(pos_avg_dists),
            "neg_avg_dists_mean": np.mean(neg_avg_dists),
            "acc_bl_mean": np.mean(accs_bl),
            "acc_bl_std": np.std(accs_bl),
            "sep_bl": np.mean(seps_bl),
            "sep_ratio": np.mean(seps)/np.mean(seps_bl),
            "acc_ratio": np.mean(accs)/np.mean(accs_bl),
            "pc1_var_ratio_mean": np.mean(pc1_var_ratios),
            "pc1_var_ratio_std": np.std(pc1_var_ratios),
            "pc2_var_ratio_mean": np.mean(pc2_var_ratios),
            "pc2_var_ratio_std": np.std(pc2_var_ratios)
        }
        log_df = pd.DataFrame([log_d])
        log_df.to_csv(f"figs/het_sep/{heteroatom}_stats.csv",index=False)

        seps_mean = np.mean(seps)
        seps_bl_mean = np.mean(seps_bl)
        # make the plot
        plot_d = plot_het_histogram(
            heteroatom,
            seps_mean,
            seps_bl_mean,
            accs,
            accs_bl,
            include_legend=False)
        plot_d["fig"].savefig(f"figs/het_sep/{heteroatom}_hist.png",format="png",dpi=300)
        plot_d["fig"].savefig(f"figs/het_sep/{heteroatom}_hist.pdf",format="pdf")
        plot_d["pval_df"].to_csv(f"figs/het_sep/{heteroatom}_pvals.csv",index=False)
        plt.close("all")


def plot_hs_single(
        ds, 
        model, 
        pred_d, 
        data_d, 
        model_d, 
        run_d, 
        hs_single_heteroatom="N",
        hs_spec_id=57053,
        hs_sirius_project_dp="_data/sirius/project",
        hs_cmap="coolwarm",
        hs_input_feats="atom",
        **kwargs):

    model.to("cpu")
    model.eval()
    spec_data = {}
    targ_ds = ds.get_subset([hs_spec_id], key="spec_id")
    targ_dl = th_data.DataLoader(
        targ_ds,
        batch_size=1,
        collate_fn=ds.get_collate_fn(),
        num_workers=0,
        pin_memory=False,
        shuffle=False,
        drop_last=False
    )
    data = next(iter(targ_dl))
    spec_id = data["spec_id"][0]
    smiles = data["smiles"][0]
    tree_fp = os.path.join(hs_sirius_project_dp,f"spec_{spec_id}.json")
    tree_data = get_tree_data(tree_fp,hs_single_heteroatom)
    tree_bins = tree_data["bins"]
    tree_labels = tree_data["labels"]
    tree_formulae = tree_data["formulae"]	
    # compute gis
    gis, gi_labels, gi_formulae, gi_bins, targ_vals = compute_gis(
        data,
        model,
        hs_input_feats,
        tree_bins=tree_bins,
        tree_labels=tree_labels,
        tree_formulae=tree_formulae
    )
    # plot pca
    pca_fig, gi_svgs, pca_gi_bins = plot_annotated_pca(
        gis, 
        gi_labels, 
        gi_formulae, 
        gi_bins, 
        targ_vals, 
        smiles, 
        hs_single_heteroatom, 
        hs_cmap)
    pca_fig.savefig(f"figs/het_sep/pca_{spec_id}.png",format="png",dpi=300)
    pca_fig.savefig(f"figs/het_sep/pca_{spec_id}.pdf",format="pdf")
    for gi_idx, gi_svg in enumerate(gi_svgs):
        with open(f"figs/het_sep/gi_svg_{gi_idx}.svg","wb") as f:
            f.write(gi_svg)
    plt.close("all")
    # plot spectrum
    targ = data["spec"]
    with th.no_grad():
        pred = model(data,return_input_feats=False)["pred"]
    # unprocess (but scale to 1000)
    un_targ = unprocess_spec(targ,data_d["transform"]).flatten().cpu().numpy()
    un_pred = unprocess_spec(pred,data_d["transform"]).flatten().cpu().numpy()
    mpl_d = plot_spec(
        un_targ,
        un_pred,
        data_d["mz_max"],
        data_d["mz_bin_res"],
        smiles=None,
        plot_title=False,
        height_ratios=[1,2,2],
        size=40,
        return_as_data=False,
        figsize=(20,20),
        rescale_mz_min=0.,
        include_grid=False
    )
    # annotate
    fig = mpl_d["fig"]
    ax_top = mpl_d["ax_top"]
    bar_top = mpl_d["bar_top"]
    ax_bottom = mpl_d["ax_bottom"]
    leg = mpl_d["leg"]
    pos_x = [0.15,0.30,0.6,0.8,0.15,0.35,0.6,0.8]
    for gi_idx, gi_bin in enumerate(pca_gi_bins):
        bar_xy = bar_top[gi_bin].get_xy()
        bar_height = bar_top[gi_bin].get_height()
        bar_pos = (bar_xy[0],bar_xy[1]) #+bar_height)
        if gi_idx < 4:
            # ann_pos_x = (gi_idx+1.)/5.
            ann_ax = ax_top
            ann_color = "green"
            if gi_idx % 2 == 0:
                ann_pos_y = 0.5
            else:
                ann_pos_y = 0.6
            ann_valign = "top"
        else:
            # ann_pos_x = (gi_idx-4+1.)/5.
            ann_ax = ax_bottom
            ann_color = "purple"
            if gi_idx % 2 == 0:
                ann_pos_y = 0.5
            else:
                ann_pos_y = 0.4
            ann_valign = "bottom"
        ann_pos_x = pos_x[gi_idx]
        ann_label = f"{gi_bin-1} Da"
        ann_pos = (ann_pos_x,ann_pos_y)
        ann_arrow = dict(arrowstyle="-",color=ann_color)
        ann_ax.annotate(
            ann_label,
            xy=bar_pos,
            xycoords="data",
            xytext=ann_pos,
            textcoords="axes fraction",
            fontsize=36,
            color=ann_color,
            arrowprops=ann_arrow,
            zorder=10,
            verticalalignment=ann_valign,
            horizontalalignment="center"
        )
    fig.savefig(f"figs/het_sep/spec_{spec_id}.png",format="png",dpi=300)
    fig.savefig(f"figs/het_sep/spec_{spec_id}.pdf",format="pdf")
    plt.close("all")
    # save structure
    mol_im_png = get_mol_im(smiles,svg=False)
    mol_im_png.save(f"figs/het_sep/spec_{spec_id}_mol.png")
    mol_im_svg = get_mol_im(smiles,svg=True)
    with open(f"figs/het_sep/spec_{spec_id}_mol.svg","wb") as f:
        f.write(mol_im_svg)

def main(
        custom_fp, 
        chkpt, 
        do_plot_examples, 
        do_plot_ce, 
        do_hs_hist, 
        do_hs_single,
        num_entries,
        skip_preds,
        **kwargs):

    template_fp = "config/template.yml"
    device_id = 0

    # load config
    entity_name, project_name, run_name, data_d, model_d, run_d = load_config(
        template_fp,
        custom_fp,
        device_id,
        None
    )
    if num_entries > 0:
        data_d["num_entries"] = num_entries

    ds, model = get_ds_model_2(data_d, model_d, run_d, chkpt)

    if skip_preds:
        # faster, for debugging
        pred_d = {}
    else:
        # get predictions
        pred_d = get_preds(ds, model, run_d)

    if do_plot_examples:
        plot_examples(ds, model, pred_d, data_d, model_d, run_d, **kwargs)

    if do_plot_ce:
        plot_ce(ds, model, pred_d, data_d, model_d, run_d, **kwargs)

    if do_hs_hist:
        plot_hs_hist(ds, model, pred_d, data_d, model_d, run_d, **kwargs)

    if do_hs_single:
        plot_hs_single(ds, model, pred_d, data_d, model_d, run_d, **kwargs)


if __name__ == "__main__":

    def booltype(x):
        return bool(strtobool(x))

    parser = argparse.ArgumentParser()
    parser.add_argument("--custom_fp", type=str, default="config/all_prec_type/nist_inchikey_all_MF.yml")
    # config/all_prec_type/nist_scaffold_all_MF.yml
    parser.add_argument("--chkpt", type=str, default="2pxvtxis")
    # exp4.2_gf20_split1_0: 2f90fe93
    parser.add_argument("--do_plot_examples", type=booltype, default=False)
    parser.add_argument("--do_plot_ce", type=booltype, default=False)
    parser.add_argument("--do_hs_hist", type=booltype, default=False)
    parser.add_argument("--do_hs_single", type=booltype, default=False)
    parser.add_argument("--num_entries", type=int, default=-1)
    parser.add_argument("--plot_ce_mol_id", type=int, default=32249)
    parser.add_argument("--plot_ce_ces", type=float, nargs="+", default=[60.,90.,130.,160.])
    parser.add_argument("--hs_min_num_peaks", type=int, default=10)
    parser.add_argument("--hs_sim_thresh", type=float, default=0.8)
    parser.add_argument("--hs_tree_thresh", type=float, default=0.8)
    parser.add_argument("--hs_prec_type", type=str, default="[M+H]+")
    parser.add_argument("--hs_heteroatoms", type=str, nargs="+", default=["N","Cl","F","P","S"])
    parser.add_argument("--hs_num_target_spec", type=int, default=-1) #1000)
    parser.add_argument("--hs_sirius_project_dp", type=str, default="_data/sirius/project")
    parser.add_argument("--hs_drop_ambiguous", type=booltype, default=True)
    parser.add_argument("--hs_overlap_thresh", type=int, default=5)
    parser.add_argument("--hs_neg_thresh", type=int, default=2)
    parser.add_argument("--hs_pos_thresh", type=int, default=2)
    parser.add_argument("--hs_input_feats", type=str, default="atom")
    parser.add_argument("--hs_num_random_bl", type=int, default=10)
    parser.add_argument("--hs_spec_id", type=int, default=57053)
    parser.add_argument("--hs_single_heteroatom", type=str, default="N")
    parser.add_argument("--hs_cmap", type=str, default="coolwarm")
    parser.add_argument("--skip_preds", type=booltype, default=False)
    args = parser.parse_args()

    np.random.seed(420)
    th.manual_seed(69)
    
    kwargs = vars(args)

    main(**kwargs)
