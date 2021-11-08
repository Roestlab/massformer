import wandb
import argparse
import numpy as np
import torch as th
import torch.nn.functional as F
import yaml
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import rdkit
import rdkit.Chem as Chem
import copy
import seaborn as sns

from dataset import data_to_device
from plot_utils import plot_atom_attn, plot_progression, plot_spec, viz_attention, plot_combined_kde, plot_atom_attn
from runner import get_ds_model

def load_dicts(new_run_dp,template_d,flags):

	with open(os.path.join(new_run_dp,"config.yaml"),"r") as config_file:
		config_d = yaml.load(config_file, Loader=yaml.FullLoader)
	data_d_keys = template_d["data"].keys()
	model_d_keys = template_d["model"].keys()
	run_d_keys = template_d["run"].keys()
	data_d, model_d, run_d = {}, {}, {}
	for k,v in config_d.items():
		if k in data_d_keys:
			data_d[k] = v["value"]
		elif k in model_d_keys:
			model_d[k] = v["value"]
		elif k in run_d_keys:
			run_d[k] = v["value"]
	run_d["do_test"] = False
	run_d["do_matching_2"] = False
	run_d["batch_size"] = 100
	return data_d, model_d, run_d

def get_attention_map(attn_mats,layer_idx=-1):

	assert attn_mats.shape[0] == 1, attn_mats.shape
	# stack attention matrices, remove batch dimension
	attn_mats = attn_mats.squeeze(0) # [num_layers, num_heads, num_nodes+1, num_nodes+1]
	# average over heads
	attn_mats = th.mean(attn_mats,dim=1) # [num_layers, num_nodes+1, num_nodes+1]
	# residual
	res_attn = th.eye(attn_mats.shape[1],device=attn_mats.device) # [num_nodes+1, num_nodes+1]
	aug_attn_mats = attn_mats + res_attn.unsqueeze(0)
	aug_attn_mats = aug_attn_mats / th.sum(aug_attn_mats,dim=2).unsqueeze(-1)
	# multiply weight matrices
	joint_attn = th.zeros_like(aug_attn_mats) # [num_layers, num_nodes+1, num_nodes+1]
	joint_attn[0] = aug_attn_mats[0]
	for n in range(1,aug_attn_mats.shape[0]):
		joint_attn[n] = th.matmul(aug_attn_mats[n],joint_attn[n-1])
	# get attention map, excluding [CLS]
	attn_map = joint_attn[layer_idx,0,1:].detach()
	# renormalize
	attn_map = attn_map / th.sum(attn_map)
	return attn_map.numpy()

def do_ce_viz(flags,model,ds,data_d,model_d,run_d):

	# get data
	dl_d = ds.get_dataloaders(run_d)
	val_idx = dl_d["primary"]["val"].dataset.indices
	val_df = ds.spec_df.iloc[val_idx]
	val_num_peaks = val_df["peaks"].apply(len)
	n_val_df = val_df[val_num_peaks>flags.min_num_peaks]
	# targ_row = n_val_df.sample(n=1,replace=False,random_state=3405)
	assert 48991 in set(n_val_df["spectrum_id"])
	targ_row = n_val_df[n_val_df["spectrum_id"]==48991]
	# targ_idx = val_df[val_df["spectrum_id"]==targ_id].index
	# create a bunch of ce examples
	targ_ce = targ_row[ds.ce_key].item()
	ces = (np.linspace(-1.0,1.0,num=flags.num_ces)*ds.std_ce+ds.mean_ce).tolist()
	new_rows = [targ_row]
	for ce in ces:
		new_row = targ_row.copy()
		new_row[ds.ce_key] = ce
		new_rows.append(new_row)
	# add rows to ds
	for new_row in new_rows:
		ds.spec_df = ds.spec_df.append(new_row,ignore_index=True)
	targ_idx = np.arange(len(ds)-len(new_rows),len(ds))
	other_idx = np.arange(0,targ_idx.shape[0])
	track_dl_d = ds.get_track_dl(
		targ_idx,
		other_idx=other_idx
	)
	model.eval()
	dev = run_d["device"]
	model.to(dev)
	log_d = {}
	dl = track_dl_d["other"]
	with th.no_grad():
		specs = []
		ces = []
		for d_idx, d in tqdm(enumerate(dl),desc="> ce_viz",total=len(dl)):
			d = data_to_device(d,dev,False)
			target_spec = d["spec"]
			prec_mz_bin = d["prec_mz_bin"][0]
			if d_idx == 0:
				specs.append(target_spec.cpu().detach().numpy())
				ces.append(d["col_energy"][0])
			pred_spec = model(d)
			specs.append(pred_spec.cpu().detach().numpy())
			ces.append(d["col_energy"][0])
	idxs = [2,8,1,0]
	plot_specs = [specs[idx] for idx in idxs]
	plot_ces = [int(np.around(ces[idx])) for idx in idxs]
	plot_data = plot_progression(plot_ces,plot_specs,len(idxs)-1,flags.min_ints_frac,data_d["mz_bin_res"],size=36)
	# log to wandb
	log_d = {
		"ce_viz": wandb.Image(plot_data)
	}
	wandb.log(log_d,commit=True)
	return plot_data

def do_ce_density(flags,model,ds,data_d,model_d,run_d):

	model.eval()
	dev = run_d["device"]
	nb = False
	model.to(dev)
	dl_d = ds.get_dataloaders(run_d)
	dl = dl_d["primary"]["val"]
	mzs = (th.arange(data_d["mz_max"]).float()*data_d["mz_bin_res"]).to(dev)
	ces, real_num_peaks, pred_num_peaks, real_mean_mzs, pred_mean_mzs = [], [], [], [], []
	with th.no_grad():
		for b_idx, b in tqdm(enumerate(dl),desc="> kde_info",total=len(dl)):
			b = data_to_device(b,dev,nb)
			b_pred = model(b)
			b_targ = b["spec"]
			b_pred = F.normalize(b_pred,dim=1,p=1)
			b_targ = F.normalize(b_targ,dim=1,p=1)
			b_real_num_peaks = th.sum(b_targ>flags.min_ints_frac,dim=1)
			b_pred_num_peaks = th.sum(b_pred>flags.min_ints_frac,dim=1)
			b_real_mean_mzs = th.sum(b_targ*mzs.unsqueeze(0),dim=1)
			b_pred_mean_mzs = th.sum(b_pred*mzs.unsqueeze(0),dim=1)
			ces.extend(b["col_energy"])
			real_num_peaks.append(b_real_num_peaks.cpu())
			pred_num_peaks.append(b_pred_num_peaks.cpu())
			real_mean_mzs.append(b_real_mean_mzs.cpu())
			pred_mean_mzs.append(b_pred_mean_mzs.cpu())
	ces = np.array(ces)
	real_num_peaks = th.cat(real_num_peaks,dim=0).numpy()
	pred_num_peaks = th.cat(pred_num_peaks,dim=0).numpy()
	real_mean_mzs = th.cat(real_mean_mzs,dim=0).numpy()
	pred_mean_mzs = th.cat(pred_mean_mzs,dim=0).numpy()
	ymax = 300 #max(np.max(real_num_peaks),np.max(pred_num_peaks))
	plot_data = plot_combined_kde(ces,real_mean_mzs,ces,pred_mean_mzs,ymin=0,ymax=ymax,xlabel="Collision Energy (Normalized)",ylabel="Mean Mass/Charge (m/z)",size=36)
	log_d = {
		"mean_mz_vz_ce": wandb.Image(plot_data)
	}
	wandb.log(log_d,commit=True)
	return plot_data

def do_attention_viz(flags,model,ds,data_d,model_d,run_d):

	model.eval()
	dev = run_d["device"]
	model.to(dev)
	dl_d = ds.get_dataloaders(run_d)
	val_idx = dl_d["primary"]["val"].dataset.indices
	val_df = ds.spec_df.iloc[val_idx]
	ds.mol_df.index.name = None
	val_df = val_df.merge(ds.mol_df[["mol_id","scaffold"]],on=["mol_id"],how="inner")
	# print(val_df["scaffold"].nunique())
	same_g = val_df.groupby(by=["scaffold","prec_type","nce"]).size().reset_index(name="counts")
	# good ones: 1492, 232323
	target_row = same_g[same_g["counts"]>=5].sample(n=1,random_state=232323)
	target_df = val_df.merge(target_row[["scaffold","prec_type","nce"]],how="inner")
	# print(target_df)
	target_idx = ds.spec_df[ds.spec_df["spectrum_id"].isin(target_df["spectrum_id"])].index
	track_dl_d = ds.get_track_dl(
		target_idx,
		other_idx=np.arange(len(target_df))
	)
	log_d = {}
	dl = track_dl_d["other"]
	with th.no_grad():
		for d_idx, d in tqdm(enumerate(dl),desc="> attention_viz",total=len(dl)):
			d = data_to_device(d,dev,False)
			target_spec = d["spec"].cpu().numpy().flatten()
			target_smiles = d["smiles"][0]
			# print(target_smiles)
			pred_spec = model(d).cpu().numpy().flatten()
			attn_mats = model.get_attn_mats(d).cpu()
			attn_map = get_attention_map(attn_mats)
			attn_map = (attn_map-np.min(attn_map))/(np.max(attn_map)-np.min(attn_map))
			attn_im = viz_attention(target_smiles,attn_map,cmap="viridis")
			plot_data = plot_spec(
				target_spec,
				pred_spec,
				data_d["mz_max"],
				data_d["mz_bin_res"],
				mol_image=attn_im,
				plot_title=False,
				height_ratios=[1,1,1],
				size=36
			)
			log_d[f"attn_viz_{d_idx}"] = wandb.Image(plot_data)
	wandb.log(log_d,commit=True)

def do_atom_attention(flags,model,ds,data_d,model_d,run_d):

	model.eval()
	dev = run_d["device"]
	nb = False
	model.to(dev)
	dl_d = ds.get_dataloaders(run_d)
	dl = dl_d["primary"]["val"]
	all_attn_mats, all_smileses = [], []
	with th.no_grad():
		for b_idx, b in tqdm(enumerate(dl),desc="> attn_maps",total=len(dl)):
			b = data_to_device(b,dev,nb)
			attn_mats = model.get_attn_mats(b)
			all_attn_mats.append(attn_mats.cpu())
			all_smileses.append(b["smiles"])
	atom_d_1 = {
		"C": [],
		"N": [],
		"O": [],
		"P": [],
		"S": [],
		"Cl": [],
		"?": []
	}
	atom_d_2 = copy.deepcopy(atom_d_1)
	for i in tqdm(range(len(all_attn_mats)),desc="> atom_stats",total=len(all_attn_mats)):
		b_attn_mats = all_attn_mats[i]
		b_smileses = all_smileses[i]
		for j in range(b_attn_mats.shape[0]):
			attn_mats = b_attn_mats[j:j+1]
			smiles = b_smileses[j]
			mol = Chem.MolFromSmiles(smiles)
			attn_map = get_attention_map(attn_mats)
			cur_atom_d = {k:[] for k in atom_d_1.keys()}
			for atom_idx, atom in enumerate(mol.GetAtoms()):
				symbol = atom.GetSymbol()
				if symbol in cur_atom_d:
					cur_atom_d[symbol].append(attn_map[atom_idx].item())
				else:
					cur_atom_d["?"].append(attn_map[atom_idx].item())
			assert len(cur_atom_d["C"]) > 0, smiles
			C_mean = np.mean(cur_atom_d["C"])
			for k,v in cur_atom_d.items():
				if k != "C" and len(v) > 0:
					atom_d_1[k].append(np.mean(v)/C_mean)
			for k,v in cur_atom_d.items():
				if len(v) > 0:
					atom_d_2[k].append(np.mean(v))
	atom_avg_d_1 = {k:(np.mean(v),np.std(v)) for k,v in atom_d_1.items()}
	atom_avg_d_1["C"] = (1.,0.)
	data = plot_atom_attn(atom_avg_d_1,size=36)
	log_d = {
		"atom_attn": wandb.Image(data)
	}
	wandb.log(log_d,commit=True)
	return data

def do_setup(flags):

	if flags.device_id == -1:
		dev = th.device("cpu")
	else:
		dev = th.device(f"cuda:{flags.device_id}")

	with open(flags.template_fp,"r") as template_file:
		template_d = yaml.load(template_file, Loader=yaml.FullLoader)

	wandb_base = os.path.join(flags.account_name,flags.project_name)
	wandb_config = {
		"run_id": flags.run_id
	}
	wandb.init(project=flags.project_name,name=flags.explain_name,config=wandb_config,mode=flags.wandb_mode)

	old_run_dp = os.path.join(wandb_base,flags.run_id)
	new_run_dp = os.path.join(wandb.run.dir,flags.run_id)
	os.makedirs(new_run_dp,exist_ok=True)
	wandb.restore("config.yaml",run_path=old_run_dp,root=new_run_dp,replace=False)
	wandb.restore("chkpt.pkl",run_path=old_run_dp,root=new_run_dp,replace=False)
	data_d, model_d, run_d = load_dicts(new_run_dp,template_d,flags)
	run_d["device"] = str(dev)
	run_d["num_workers"] = 0
	ds, model = get_ds_model(data_d,model_d,run_d)
	
	# setup model
	chkpt_d = th.load(os.path.join(new_run_dp,"chkpt.pkl"),map_location=dev)
	model.load_state_dict(chkpt_d["best_model_sd"])
	os.remove(os.path.join(new_run_dp,"chkpt.pkl"))
	model.eval()
	model.to(dev)

	return model, ds, data_d, model_d, run_d

if __name__ == "__main__":

	parser = argparse.ArgumentParser()
	parser.add_argument("run_id", type=str)
	parser.add_argument("--template_fp", type=str, default="config/template.yml")
	parser.add_argument("--explain_name", type=str, default="explain")
	parser.add_argument("--project_name", type=str, default="massformer")
	parser.add_argument("--account_name", type=str, default="adamoyoung")
	parser.add_argument("--device_id", type=int, default=0, choices=[-1,0,1,2,3])
	parser.add_argument("--min_num_peaks", type=int, default=10)
	parser.add_argument("--num_ces", type=int, default=10)
	parser.add_argument("--wandb_mode", type=str, default="offline", choices=["online","offline"])
	parser.add_argument("--min_ints_frac", type=float, default=0.01)
	flags = parser.parse_args()

	model,ds,data_d,model_d,run_d = do_setup(flags)

	do_ce_viz(flags,model,ds,data_d,model_d,run_d)
	
	do_ce_density(flags,model,ds,data_d,model_d,run_d)
	
	do_attention_viz(flags,model,ds,data_d,model_d,run_d)

	do_atom_attention(flags,model,ds,data_d,model_d,run_d)