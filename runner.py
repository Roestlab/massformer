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

from dataset import BaseDataset, data_to_device
from model import Predictor
from misc_utils import np_temp_seed, th_temp_seed, booltype, DummyContext, DummyScaler
from plot_utils import *
from losses import get_loss_func, get_sim_func


def run_train_epoch(step,epoch,model,dl_d,run_d,use_wandb,optimizer,amp_context,scaler,stdout=True):

	# stuff related to device
	dev = th.device(run_d["device"])
	nb = run_d["non_blocking"]
	# set up loss func
	loss_func = get_loss_func(run_d["loss"])
	# train
	model.train()
	for b_idx, b in tqdm(enumerate(dl_d["primary"]["train"]),desc="> train",total=len(dl_d["primary"]["train"])):
		optimizer.zero_grad()
		b = data_to_device(b,dev,nb)
		with amp_context:
			b_pred = model(b)
			b_targ = b["spec"]
			b_loss = loss_func(b_pred,b_targ)
			b_mean_loss = th.mean(b_loss,dim=0)
			b_sum_loss = th.sum(b_loss,dim=0)
		if run_d["batch_loss_agg"] == "mean":
			scaler.scale(b_mean_loss).backward()
		else:
			assert run_d["batch_loss_agg"] == "sum"
			scaler.scale(b_sum_loss).backward()
		scaler.step(optimizer)
		scaler.update()
		step += 1
	optimizer.zero_grad()
	return step, epoch, {}

def run_val(step,epoch,model,dl_d,run_d,use_wandb,amp_context,stdout=True):

	# stuff related to device
	dev = th.device(run_d["device"])
	nb = run_d["non_blocking"]
	# set up loss func
	loss_func = get_loss_func(run_d["loss"])
	sim_func = get_sim_func(run_d["sim"])
	# validation
	model.eval()
	pred, targ, sim, loss, mol_id = [], [], [], [], []
	with th.no_grad():
		for b_idx, b in tqdm(enumerate(dl_d["primary"]["val"]),desc="> val",total=len(dl_d["primary"]["val"])):
			b = data_to_device(b,dev,nb)
			with amp_context:
				b_pred = model(b)
				b_targ = b["spec"]
				b_loss = loss_func(b_pred,b_targ)
				b_sim = sim_func(b_pred,b_targ)
			b_mol_id = b["mol_id"]
			pred.append(b_pred.detach().to("cpu",non_blocking=nb))
			targ.append(b_targ.detach().to("cpu",non_blocking=nb))
			loss.append(b_loss.detach().to("cpu",non_blocking=nb))
			sim.append(b_sim.detach().to("cpu",non_blocking=nb))
			mol_id.append(b_mol_id.detach().to("cpu",non_blocking=nb))
	pred = th.cat(pred,dim=0)
	targ = th.cat(targ,dim=0)
	spec_loss = th.cat(loss,dim=0)
	spec_sim = th.cat(sim,dim=0)
	mol_id = th.cat(mol_id,dim=0)
	un_mol_id, un_mol_idx = th.unique(mol_id,dim=0,return_inverse=True)
	mol_loss = th_s.scatter_mean(spec_loss,un_mol_idx,dim=0,dim_size=un_mol_id.shape[0])
	mol_sim = th_s.scatter_mean(spec_sim,un_mol_idx,dim=0,dim_size=un_mol_id.shape[0])
	spec_mean_loss = th.mean(spec_loss,dim=0)
	spec_mean_sim = th.mean(spec_sim,dim=0)
	mol_mean_loss = th.mean(mol_loss,dim=0)
	mol_mean_sim = th.mean(mol_sim,dim=0)
	out_d = {
		"pred": pred.float(),
		"targ": targ.float(),
		"mol_id": mol_id,
		"spec_sim": spec_sim.float(), # for tracking
		"spec_mean_loss": spec_mean_loss.float(),
		"mol_mean_loss": mol_mean_loss.float(),
		"spec_mean_sim": spec_mean_sim.float(),
		"mol_mean_sim": mol_mean_sim.float()
	}
	spec_sim_hist = plot_sim_hist(run_d["sim"],spec_sim.numpy())
	mol_sim_hist = plot_sim_hist(run_d["sim"],mol_sim.numpy())
	if stdout:	
		print(f"> step {step}, epoch {epoch}: val, spec_mean_loss = {spec_mean_loss:.4}, mol_mean_loss = {mol_mean_loss:.4}")
	if use_wandb:
		log_dict = {
			"Epoch": epoch,
			"val_spec_loss": spec_mean_loss,
			"val_spec_sim": spec_mean_sim,
			"val_mol_loss": mol_mean_loss,
			"val_mol_sim": mol_mean_sim
		}
		if run_d["save_media"]:
			log_dict["val_spec_sim_hist"] = wandb.Image(spec_sim_hist)
			log_dict["val_mol_sim_hist"] = wandb.Image(mol_sim_hist)
		wandb.log(log_dict, commit=False)
	return step, epoch, out_d

def run_track(step,epoch,model,dl_d,run_d,use_wandb,ds,data_d,score_d,stdout=True):

	# stuff related to device
	dev = th.device(run_d["device"])
	nb = run_d["non_blocking"]
	# set up loss func
	loss_func = get_loss_func(run_d["loss"])
	sim_func = get_sim_func(run_d["sim"])
	# tracking
	if run_d["save_media"] and run_d["num_track"] > 0:
		model.to(dev)
		model.eval()
		# get top/bottom k similarity
		topk, argtopk = th.topk(score_d["spec_sim"],run_d["num_track"],largest=True)
		bottomk, argbottomk = th.topk(score_d["spec_sim"],run_d["num_track"],largest=False)
		topk_str = "[" + ",".join([f"{val:.2f}" for val in topk.tolist()]) + "]"
		bottomk_str = "[" + ",".join([f"{val:.2f}" for val in bottomk.tolist()]) + "]"
		if stdout:
			print(f"> tracking: topk = {topk_str} , bottomk = {bottomk_str}")
		val_idx = dl_d["primary"]["val"].dataset.indices
		track_dl_d = ds.get_track_dl(
			val_idx,
			num_rand_idx=run_d["num_track"],
			topk_idx=argtopk.numpy(),
			bottomk_idx=argbottomk.numpy()
		)
		for dl_type,dl in track_dl_d.items():
			for d_idx, d in enumerate(dl):
				# import pdb; pdb.set_trace()
				d = data_to_device(d,dev,nb)
				pred = model(d)
				targ = d["spec"]
				loss = loss_func(pred,targ)
				sim = sim_func(pred,targ)
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
				plot_data = plot_spec(
					targ,pred,
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
					plot_title=run_d["track_plot_title"]
				)
				if use_wandb:
					dl_type = dl_type.rstrip("")
					log_dict = {
						"Epoch": epoch, 
						f"{dl_type}_{d_idx}": wandb.Image(plot_data)
					}
					wandb.log(log_dict, commit=False)
	return step, epoch, {}

def run_test(step,epoch,model,dl_d,run_d,use_wandb,amp_context,exclude=["train","val"],stdout=True):

	# stuff related to device
	dev = th.device(run_d["device"])
	nb = run_d["non_blocking"]
	# set up loss func
	loss_func = get_loss_func(run_d["loss"])
	sim_func = get_sim_func(run_d["sim"])
	# test
	if run_d["do_test"]:
		model.to(dev)
		model.eval()
		out_d = {}
		for order in ["primary","secondary"]:
			out_d[order] = {}
			for dl_key,dl in dl_d[order].items():
				if dl_key in exclude:
					continue
				pred, targ, sim, loss, mol_id = [], [], [], [], []
				with th.no_grad():
					for b_idx, b in tqdm(enumerate(dl),desc=f"> {dl_key}",total=len(dl)):
						b = data_to_device(b,dev,nb)
						with amp_context:
							b_pred = model(b)
							b_targ = b["spec"]
							b_loss = loss_func(b_pred,b_targ)
							b_sim = sim_func(b_pred,b_targ)
						b_mol_id = b["mol_id"]
						pred.append(b_pred.detach().to("cpu",non_blocking=nb))
						targ.append(b_targ.detach().to("cpu",non_blocking=nb))
						loss.append(b_loss.detach().to("cpu",non_blocking=nb))
						sim.append(b_sim.detach().to("cpu",non_blocking=nb))
						mol_id.append(b_mol_id.detach().to("cpu",non_blocking=nb))
				pred = th.cat(pred,dim=0)
				targ = th.cat(targ,dim=0)
				spec_loss = th.cat(loss,dim=0)
				spec_sim = th.cat(sim,dim=0)
				mol_id = th.cat(mol_id,dim=0)
				un_mol_id, un_mol_idx = th.unique(mol_id,dim=0,return_inverse=True)
				mol_loss = th_s.scatter_mean(spec_loss,un_mol_idx,dim=0,dim_size=un_mol_id.shape[0])
				mol_sim = th_s.scatter_mean(spec_sim,un_mol_idx,dim=0,dim_size=un_mol_id.shape[0])
				spec_mean_loss = th.mean(spec_loss,dim=0)
				spec_mean_sim = th.mean(spec_sim,dim=0)
				mol_mean_loss = th.mean(mol_loss,dim=0)
				mol_mean_sim = th.mean(mol_sim,dim=0)
				out_d[order][dl_key] = {
					"pred": pred.float(),
					"targ": targ.float(),
					"mol_id": mol_id
				}
				spec_sim_hist = plot_sim_hist(run_d["sim"],spec_sim.numpy())
				mol_sim_hist = plot_sim_hist(run_d["sim"],mol_sim.numpy())
				if stdout:	
					print(f"> {dl_key}, spec_mean_loss = {spec_mean_loss:.4}, mol_mean_loss = {mol_mean_loss:.4}")
				if use_wandb:
					log_dict = {
						f"{dl_key}_spec_loss": spec_mean_loss,
						f"{dl_key}_spec_sim": spec_mean_sim,
						f"{dl_key}_mol_loss": mol_mean_loss,
						f"{dl_key}_mol_sim": mol_mean_sim,
					}
					if run_d["save_media"]:
						log_dict[f"{dl_key}_spec_sim_hist"] = wandb.Image(spec_sim_hist)
						log_dict[f"{dl_key}_mol_sim_hist"] = wandb.Image(mol_sim_hist)
					wandb.log(log_dict, commit=False)
	else:
		out_d = {}
	return step, epoch, out_d

def run_match_2(step,epoch,model,dl_d,run_d,use_wandb,amp_context,stdout=True):
	# TODO: make this function more general!

	# stuff related to device
	dev = th.device(run_d["device"])
	nb = run_d["non_blocking"]
	# set up loss func
	sim_func = get_sim_func(run_d["sim"])
	assert run_d["sim"] == "cos"
	# match
	if run_d["do_matching_2"]:
		model.to(dev)
		model.eval()
		out_d = {}

		# rename dl_d
		dl_d = dl_d.copy()
		dl_d["primary"]["train"] = dl_d["primary"]["train_2"]
		del dl_d["primary"]["train_2"]
		
		ds_d = {}
		for order in ["primary","secondary"]:
			ds_d[order] = {}
			for dl_key in dl_d[order].keys():
				ds_d[order][dl_key] = {
					"real_spec": [],
					"pred_spec": [],
					"spec_ids": [],
					"mol_ids": [],
					"prec_types": [],
					"col_energies": []
				}

		splits = [(("primary","test"),(("primary","val"),("primary","train")))]
		for dl_key in dl_d["secondary"].keys():
			query = ("secondary",dl_key)
			ref = (("primary","train"),("primary","val"),("primary","test"))
			splits.append((query,ref))
		# print(splits)

		base_ds = dl_d["primary"]["train"].dataset.dataset

		for order in ["primary","secondary"]:
			for dl_key in ds_d[order].keys():
				real_spec, pred_spec, spec_ids, mol_ids, prec_types, col_energies = [], [], [], [], [], []
				dl = dl_d[order][dl_key]
				with th.no_grad():
					for b_idx, b in tqdm(enumerate(dl),desc=f"> collecting {order} {dl_key}",total=len(dl)):
						b_targ = b["spec"]
						b_spec_id = b["spectrum_id"]
						b_mol_id = b["mol_id"]
						b_prec_type = b["prec_type"]
						b_col_energy = b["col_energy"]
						real_spec.append(b_targ.cpu())
						spec_ids.append(b_spec_id.cpu())
						mol_ids.append(b_mol_id.cpu())
						prec_types.extend(b_prec_type)
						col_energies.extend(b_col_energy)
						if not (order == "primary" and dl_key in ["train","val"]):
							b = data_to_device(b,dev,nb)
							with amp_context:
								b_pred = model(b)
							pred_spec.append(b_pred.cpu())
				cur_d = ds_d[order][dl_key]
				cur_d["real_spec"] = th.cat(real_spec,dim=0)
				cur_d["spec_ids"] = th.cat(spec_ids,dim=0)
				cur_d["mol_ids"] = th.cat(mol_ids,dim=0)
				cur_d["prec_types"] = th.tensor([base_ds.prec_type_c2i[prec_type] for prec_type in prec_types])
				cur_d["col_energies"] = th.tensor(col_energies)
				if len(pred_spec) > 0:
					cur_d["pred_spec"] = th.cat(pred_spec,dim=0)
		
		for split in splits:
			query_ds = split[0]
			ref_dses = split[1]
			r_spec, r_spec_ids, r_mol_ids, r_prec_types, r_col_energies = [], [], [], [], []
			q_spec, q_spec_ids, q_mol_ids, q_prec_types, q_col_energies = [], [], [], [], []
			# set up query
			q_order, q_ds_key = query_ds
			query_d = ds_d[q_order][q_ds_key]
			r_spec.append(query_d["pred_spec"])
			r_spec_ids.append(query_d["spec_ids"])
			r_mol_ids.append(query_d["mol_ids"])
			r_prec_types.append(query_d["prec_types"])
			r_col_energies.append(query_d["col_energies"])
			q_spec.append(query_d["real_spec"])
			q_spec_ids.append(query_d["spec_ids"])
			q_mol_ids.append(query_d["mol_ids"])
			q_prec_types.append(query_d["prec_types"])
			q_col_energies.append(query_d["col_energies"])
			# set up ref
			for ref_ds in ref_dses:
				r_order, r_ds_key = ref_ds
				ref_d = ds_d[r_order][r_ds_key]
				r_spec.append(ref_d["real_spec"])
				r_spec_ids.append(ref_d["spec_ids"])
				r_mol_ids.append(ref_d["mol_ids"])
				r_prec_types.append(ref_d["prec_types"])
				r_col_energies.append(ref_d["col_energies"])
			r_spec = th.cat(r_spec,dim=0)
			r_spec_ids = th.cat(r_spec_ids,dim=0)
			r_mol_ids = th.cat(r_mol_ids,dim=0)
			r_prec_types = th.cat(r_prec_types,dim=0)
			r_col_energies = th.cat(r_col_energies,dim=0)
			q_spec = th.cat(q_spec,dim=0)
			q_spec_ids = th.cat(q_spec_ids,dim=0)
			q_mol_ids = th.cat(q_mol_ids,dim=0)
			q_prec_types = th.cat(q_prec_types,dim=0)
			q_col_energies = th.cat(q_col_energies,dim=0)
			# set up dataloader
			q_ds = th.utils.data.TensorDataset(q_spec,q_spec_ids,q_prec_types,q_col_energies)
			q_dl = th.utils.data.DataLoader(
				q_ds,
				num_workers=0,
				batch_size=200,
				drop_last=False,
				shuffle=False,
				pin_memory=True
			)
			q_ranks, q_norm_ranks, q_mean_sims = [], [], []
			# send r stuff to device
			r_spec = F.normalize(r_spec,p=2,dim=1).to(dev,non_blocking=nb)
			r_spec_ids = r_spec_ids.to(dev,non_blocking=nb)
			r_prec_types = r_prec_types.to(dev,non_blocking=nb)
			r_col_energies = r_col_energies.to(dev,non_blocking=nb)
			# import pdb; pdb.set_trace()
			# r = b_r_spec.shape[0]
			# q = b_q_spec.shape[0]
			# b_r_spec = r_spec.repeat(q,1)
			# b_q_spec = b_q_spec.repeat(r,1)
			# b_q_sims = sim_func(b_q_spec,b_r_spec).reshape(q,r)
			for b_idx, b in tqdm(enumerate(q_dl),desc=f"> m2 {q_ds_key}",total=len(q_dl)):
				b_q_spec = b[0].to(dev,non_blocking=nb)
				b_q_spec_ids = b[1].to(dev,non_blocking=nb)
				b_q_prec_types = b[2].to(dev,non_blocking=nb)
				b_q_col_energies = b[3].to(dev,non_blocking=nb)
				b_q_prec_types_mask = b_q_prec_types.unsqueeze(1)==r_prec_types.unsqueeze(0)
				b_q_col_energies_mask = th.isclose(b_q_col_energies.unsqueeze(1),r_col_energies.unsqueeze(0))
				b_q_mask = b_q_prec_types_mask&b_q_col_energies_mask
				assert th.all(th.any(b_q_mask,dim=1))
				b_q_spec = F.normalize(b_q_spec,p=2,dim=1)
				b_q_sims = th.matmul(b_q_spec,r_spec.T)
				b_q_sims = b_q_mask.float()*b_q_sims+((~b_q_mask).float())*-1.
				b_q_sort = th.argsort(-b_q_sims+0.00001*th.rand_like(b_q_sims),dim=1)
				b_q_match = (r_spec_ids[b_q_sort]==b_q_spec_ids.unsqueeze(1)).float()
				b_q_num_sims = th.sum(b_q_mask.float(),dim=1)
				b_q_rank = (th.argmax(b_q_match,dim=1)+1).float()
				b_q_norm_rank = (b_q_rank-1.) / th.clamp(b_q_num_sims-1.,1.)
				b_q_sum_sims = th.sum(b_q_mask.float()*b_q_sims,dim=1)
				b_q_mean_sims = b_q_sum_sims / b_q_num_sims
				q_ranks.append(b_q_rank.cpu())
				q_norm_ranks.append(b_q_norm_rank.cpu())
				q_mean_sims.append(b_q_mean_sims.cpu())
			spec_rank = th.cat(q_ranks,dim=0)
			spec_norm_rank = th.cat(q_norm_ranks,dim=0)
			spec_mean_sim = th.cat(q_mean_sims,dim=0)
			assert th.all(spec_norm_rank<=1.) and th.all(spec_norm_rank>=0.), (th.min(spec_norm_rank),th.max(spec_norm_rank))
			assert th.all(spec_mean_sim<=1.) and th.all(spec_mean_sim>=0.), (th.min(spec_mean_sim),th.max(spec_mean_sim))
			un_mol_ids, mol_idx = th.unique(q_mol_ids,return_inverse=True)
			mol_rank = th_s.scatter_mean(spec_rank,mol_idx,dim=0,dim_size=un_mol_ids.shape[0])
			mol_norm_rank = th_s.scatter_mean(spec_norm_rank,mol_idx,dim=0,dim_size=un_mol_ids.shape[0])
			mol_mean_sim = th_s.scatter_mean(spec_mean_sim,mol_idx,dim=0,dim_size=un_mol_ids.shape[0])
			spec_mean_rank = th.mean(spec_rank)
			spec_mean_norm_rank = th.mean(spec_norm_rank)
			spec_recall_at_1 = th.mean((spec_norm_rank<=0.01).float())
			spec_recall_at_5 = th.mean((spec_norm_rank<=0.05).float())
			spec_recall_at_10 = th.mean((spec_norm_rank<=0.10).float())
			mol_mean_rank = th.mean(mol_rank)
			mol_mean_norm_rank = th.mean(mol_norm_rank)
			mol_recall_at_1 = th.mean((mol_norm_rank<=0.01).float())
			mol_recall_at_5 = th.mean((mol_norm_rank<=0.05).float())
			mol_recall_at_10 = th.mean((mol_norm_rank<=0.10).float())
			if stdout:
				print(f"> by spectrum_id: mean_rank = {spec_mean_rank:.2f}, recall @1 = {spec_recall_at_1:.2f}, @5 = {spec_recall_at_5:.2f}, @10 = {spec_recall_at_10:.2f}")
				print(f"> by mol_id: mean_rank = {mol_mean_rank:.2f}, recall @1 = {mol_recall_at_1:.2f}, @5 = {mol_recall_at_5:.2f}, @10 = {mol_recall_at_10:.2f}")
			spec_cand_sim_mean_hist = plot_cand_sim_mean_hist(spec_mean_sim.numpy())
			mol_cand_sim_mean_hist = plot_cand_sim_mean_hist(mol_mean_sim.numpy())
			if use_wandb:
				log_dict = {
					f"m2_{q_ds_key}_spec_rank": spec_mean_rank,
					f"m2_{q_ds_key}_spec_norm_rank": spec_mean_norm_rank,
					f"m2_{q_ds_key}_spec_top1%": spec_recall_at_1,
					f"m2_{q_ds_key}_spec_top5%": spec_recall_at_5,
					f"m2_{q_ds_key}_spec_top10%": spec_recall_at_10,
					f"m2_{q_ds_key}_mol_rank": mol_mean_rank,
					f"m2_{q_ds_key}_mol_norm_rank": mol_mean_norm_rank,
					f"m2_{q_ds_key}_mol_top1%": mol_recall_at_1,
					f"m2_{q_ds_key}_mol_top5%": mol_recall_at_5,
					f"m2_{q_ds_key}_mol_top10%": mol_recall_at_10
				}
				if run_d["save_media"]:
					log_dict[f"m2_{q_ds_key}_spec_cand_sim_mean_hist"] = wandb.Image(spec_cand_sim_mean_hist)
					log_dict[f"m2_{q_ds_key}_mol_cand_sim_mean_hist"] = wandb.Image(mol_cand_sim_mean_hist)
				wandb.log(log_dict, commit=False)
		
	else:
		out_d = {}
	return step, epoch, out_d

def get_ds_model(data_d,model_d,run_d):

	with th_temp_seed(model_d["model_seed"]):
		dset_types = set()
		embed_types = model_d["embed_types"]
		for embed_type in embed_types:
			if embed_type == "fp":
				dset_types.add("fp")
			elif embed_type in ["gat","wln","gin_pt"]:
				dset_types.add("graph")
			elif embed_type in ["cnn","tf"]:
				dset_types.add("seq")
			elif embed_type in ["gf"]:
				dset_types.add("gf")
			else:
				raise ValueError(f"invalid embed_type {embed_type}")
		dset_types = list(dset_types)
		assert len(dset_types)>0, dset_types
		ds = BaseDataset(*dset_types,**data_d)
		dim_d = ds.get_data_dims()
		model = Predictor(dim_d,**model_d)
	dev = th.device(run_d["device"])
	model.to(dev)
	return ds, model

def train(data_d,model_d,run_d,use_wandb):

	# set seeds
	th.manual_seed(run_d["train_seed"])
	np.random.seed(run_d["train_seed"]//2)

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
	ds, model = get_ds_model(data_d,model_d,run_d)

	# set up dataloader
	dl_d = ds.get_dataloaders(run_d)

	# set up optimizer
	if run_d["optimizer"] == "adam":
		optimizer = th.optim.Adam(model.parameters(),lr=run_d["learning_rate"],weight_decay=run_d["weight_decay"])
	elif run_d["optimizer"] == "adamw":
		optimizer = th.optim.AdamW(model.parameters(),lr=run_d["learning_rate"],weight_decay=run_d["weight_decay"])
	else:
		raise NotImplementedError

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
	else:
		raise NotImplementedError

	# set up amp stuff
	if run_d["amp"]:
		amp_context = th.cuda.amp.autocast()
		scaler = th.cuda.amp.GradScaler()
	else:
		amp_context = DummyContext()
		scaler = DummyScaler()

	# basic stuff
	best_val_mean_loss = 1000000. # start with really big num
	best_val_mean_sim = 0.
	best_epoch = -1
	best_state_dict = copy.deepcopy(model.state_dict())
	early_stop_count = 0
	early_stop_thresh = run_d["early_stop_thresh"]
	step = 0
	epoch = 0
	dev = th.device(run_d["device"])

	# restore from previous run (if applicable)
	if use_wandb:
		mr_fp = os.path.join(wandb.run.dir,"chkpt.pkl")
		best_fp = os.path.join(wandb.run.dir,"best_chkpt.pkl")
		temp_mr_fp = os.path.join(wandb.run.dir,"temp_chkpt.pkl")
		if os.path.isfile(mr_fp):
			print(">>> reloading model from most recent checkpoint")
			mr_d = th.load(mr_fp,map_location="cpu")
			model.load_state_dict(mr_d["mr_model_sd"])
			model.to(dev)
			best_state_dict = copy.deepcopy(model.state_dict())
			optimizer.load_state_dict(mr_d["optimizer_sd"])
			scheduler.load_state_dict(mr_d["scheduler_sd"])
			best_val_mean_loss = mr_d["best_val_mean_loss"]
			best_val_mean_sim = mr_d["best_val_mean_sim"]
			best_epoch = mr_d["best_epoch"]
			early_stop_count = mr_d["early_stop_count"]
			step = mr_d["step"]
			epoch = mr_d["epoch"]+1
		elif os.path.isfile(best_fp):
			print(">>> reloading model from best checkpoint")
			best_d = th.load(best_fp,map_location="cpu")
			model.load_state_dict(best_d["best_model_sd"])
			model.to(dev)
			best_state_dict = copy.deepcopy(model.state_dict())
			os.remove(best_fp)
		else:
			print(">>> no checkpoint detected")
			mr_d = {
				"mr_model_sd": model.state_dict(),
				"best_model_sd": best_state_dict,
				"optimizer_sd": optimizer.state_dict(),
				"scheduler_sd": scheduler.state_dict(),
				"best_val_mean_loss": best_val_mean_loss,
				"best_val_mean_sim": best_val_mean_sim,
				"best_epoch": best_epoch,
				"early_stop_count": early_stop_count,
				"step": step,
				"epoch": epoch-1
			}
			if run_d["save_state"]:
				th.save(mr_d,temp_mr_fp)
				os.replace(temp_mr_fp,mr_fp)
				wandb.save("chkpt.pkl")

	while epoch < run_d["num_epochs"]:
		
		print(f">>> start epoch {epoch}")

		# training, single epoch
		step, epoch, train_d = run_train_epoch(step,epoch,model,dl_d,run_d,use_wandb,optimizer,amp_context,scaler)

		# validation
		step, epoch, val_d = run_val(step,epoch,model,dl_d,run_d,use_wandb,amp_context)
		if run_d["scheduler"] == "step":
			scheduler.step()
		elif run_d["scheduler"] == "plateau":
			scheduler.step(val_d[f'{run_d["stop_key"]}_mean_loss'])

		# tracking
		step, epoch, track_d = run_track(step,epoch,model,dl_d,run_d,use_wandb,ds,data_d,val_d)

		# early stopping
		val_mean_loss = val_d[f'{run_d["stop_key"]}_mean_loss']
		val_mean_sim = val_d[f'{run_d["stop_key"]}_mean_sim']
		print(f"> val loss delta: {val_mean_loss-best_val_mean_loss}")
		if best_val_mean_loss < val_mean_loss:
			early_stop_count += 1
			print(f"> val loss DID NOT decrease, early stop count at {early_stop_count}/{early_stop_thresh}")
		else:
			best_val_mean_loss = val_mean_loss
			best_val_mean_sim = val_mean_sim
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

		if use_wandb:
			# save model			
			mr_d = {
				"mr_model_sd": model.state_dict(),
				"best_model_sd": best_state_dict,
				"optimizer_sd": optimizer.state_dict(),
				"scheduler_sd": scheduler.state_dict(),
				"best_val_mean_loss": best_val_mean_loss,
				"best_val_mean_sim": best_val_mean_sim,
				"best_epoch": best_epoch,
				"early_stop_count": early_stop_count,
				"step": step,
				"epoch": epoch
			}
			if run_d["save_state"]:
				th.save(mr_d,temp_mr_fp)
				os.replace(temp_mr_fp,mr_fp)
				wandb.save("chkpt.pkl")
			# sync wandb
			wandb.log({"commit": epoch}, commit=True)

		epoch += 1

	model.load_state_dict(best_state_dict)
	step, epoch, test_d = run_test(step,epoch,model,dl_d,run_d,use_wandb,amp_context)

	model.load_state_dict(best_state_dict)
	step, epoch, match_2_d = run_match_2(step,epoch,model,dl_d,run_d,use_wandb,amp_context)

	if use_wandb:
		# final save, only include the best state (to reduce size of uploads)
		mr_d = {
			"best_model_sd": best_state_dict,
			"best_val_mean_loss": best_val_mean_loss,
			"best_val_mean_sim": best_val_mean_sim,
			"best_epoch": best_epoch
		}
		if run_d["save_state"]:
			# saving to temp first for atomic
			th.save(mr_d,temp_mr_fp)
			os.replace(temp_mr_fp,mr_fp)
			wandb.save("chkpt.pkl")
		# metrics
		log_dict = {
			"best_val_mean_loss": best_val_mean_loss,
			"best_val_mean_sim": best_val_mean_sim,
			"best_epoch": best_epoch
		}
		wandb.log(log_dict, commit=False)
		# sync wandb
		wandb.log({"commit": epoch}, commit=True)

	return


def load_config(template_fp,custom_fp,device_id):

	assert os.path.isfile(template_fp), template_fp
	if custom_fp:
		assert os.path.isfile(custom_fp), custom_fp

	with open(template_fp,"r") as template_file:
		config_d = yaml.load(template_file, Loader=yaml.FullLoader)

	# overwrite parts of the config
	if custom_fp:

		with open(custom_fp,"r") as custom_file:
			custom_d = yaml.load(custom_file, Loader=yaml.FullLoader)

		config_d["account_name"] = custom_d["account_name"]
		config_d["project_name"] = custom_d["project_name"]
		if custom_d["run_name"] is None:
			config_d["run_name"] = os.path.splitext(os.path.basename(custom_fp))[0]
		else:
			config_d["run_name"] = custom_d["run_name"]
		for k,v in custom_d.items():
			if k not in ["account_name","project_name","run_name"]:
				for k2,v2 in v.items():
					config_d[k][k2] = v2

	account_name = config_d["account_name"]
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

	return account_name, project_name, run_name, data_d, model_d, run_d


def init_or_resume_wandb_run(
		account_name=None,
		project_name=None,
		run_name=None,
		data_d=None,
		model_d=None,
		run_d=None,
		wandb_meta_dp=None,
		job_id=None,
		job_id_dp=None,
		is_sweep=False,
		group_name=None):
	"""
	for compatibility with VV preemption
	"""

	# imp_keys = ["learning_rate","ff_num_layers","ff_layer_type","dropout"]
	do_preempt = not (job_id is None)
	do_load = not (run_d["load_id"] is None)
	if is_sweep:
		assert do_preempt
	# set up run_id
	if do_preempt:
		assert not (job_id_dp is None)
		assert not do_load
		job_id_fp = os.path.join(job_id_dp,f"{job_id}.yml")
		if os.path.isfile(job_id_fp):
			print(f">>> resuming {job_id}")
			with open(job_id_fp,"r") as file:
				job_dict = yaml.safe_load(file)
			run_id = job_dict["run_id"]
			# run_dp = job_dict["run_dp"]
			wandb_config = None
			# print("wandb_config",wandb_config)
		else:
			print(f">>> starting {job_id}")
			run_id = None
			# run_dp = None
			wandb_config = {**data_d,**model_d,**run_d}
			# print("wandb_config",[(k,wandb_config[k]) for k in imp_keys])
		resume = "allow"
	else:
		run_id = None
		resume = "never"
		wandb_config = {**data_d,**model_d,**run_d}
	# init run
	# if this is a sweep, some things passed into config will be overwritten
	run = wandb.init(project=project_name,name=run_name,id=run_id,config=wandb_config,resume=resume,dir=wandb_meta_dp,group=group_name)
	# update config
	run_config_d = dict(run.config)
	# print("run_config_d",[(k,run_config_d[k]) for k in imp_keys])
	for d in [data_d,model_d,run_d]:
		for k in d.keys():
			d[k] = run_config_d[k]
	# copy files and update job file
	if do_preempt:
		if not run_id is None:
			# shutil.copy(os.path.join(run_dp,"chkpt.pkl"),os.path.join(wandb.run.dir,"chkpt.pkl"))
			wandb.restore("chkpt.pkl",root=run.dir,replace=True)
			assert os.path.isfile(os.path.join(run.dir,"chkpt.pkl"))
		with open(job_id_fp,"w") as file:
			yaml.dump(dict(run_id=run.id), file)
	if do_load:
		assert not do_preempt
		load_run_dp = os.path.join(account_name,project_name,run_d["load_id"])
		wandb.restore("chkpt.pkl",run_path=load_run_dp,root=run.dir,replace=False)
		os.replace(os.path.join(run.dir,"chkpt.pkl"),os.path.join(run.dir,"best_chkpt.pkl"))
	# train model
	train(data_d,model_d,run_d,True)
	# cleanup
	if do_preempt:
		# remove the job_id file
		os.remove(job_id_fp)
	if is_sweep:
		# overwrite old chkpt.pkl (to reduce memory usage)
		th.save(dict(),os.path.join(run.dir,"chkpt.pkl"))
	run.finish()

if __name__ == "__main__":

	parser = argparse.ArgumentParser()
	parser.add_argument("-t", "--template_fp", type=str, default="config/template.yml", help="path to template config file")
	parser.add_argument("-w", "--use_wandb", type=booltype, default=False, help="whether to turn wandb on")
	parser.add_argument("-d", "--device_id", type=int, required=False, help="device id (-1 for cpu)")
	parser.add_argument("-c", "--custom_fp", type=str, required=False, help="path to custom config file")
	parser.add_argument("-i", "--job_id", type=int, required=False, help="job_id for preemption")
	parser.add_argument("-j", "--job_id_dp", type=str, default="job_id", help="directory where job_id files are stored")
	parser.add_argument("-m", "--wandb_meta_dp", type=str, default=os.getcwd(), help="path to directory in which the wandb directory will exist")
	parser.add_argument("-n", "--num_seeds", type=int, default=0)
	flags = parser.parse_args()

	use_wandb = flags.use_wandb

	account_name, project_name, run_name, data_d, model_d, run_d = load_config(flags.template_fp,flags.custom_fp,flags.device_id)

	if use_wandb:

		if flags.num_seeds > 0:
			assert not flags.job_id is None
			assert flags.num_seeds > 1, flags.num_seeds
			# check how many have been completed
			job_id_fp = os.path.join(flags.job_id_dp,f"{flags.job_id}.yml")
			if os.path.isfile(job_id_fp):
				with open(job_id_fp,"r") as file:
					num_complete = yaml.safe_load(file)["num_complete"]
			else:
				num_complete = 0
				with open(job_id_fp,"w") as file:
					yaml.dump(dict(num_complete=num_complete),file)
			# run multiple
			if run_d["train_seed"] is None:
				meta_seed = 420420420
			else:
				meta_seed = run_d["train_seed"]
			with np_temp_seed(meta_seed):
				seed_range = np.arange(0,int(1e6))
				model_seeds = np.random.choice(seed_range,replace=False,size=(flags.num_seeds,))
				train_seeds = np.random.choice(seed_range,replace=False,size=(flags.num_seeds,))
			group_name = f"{run_name}_rand"
			for i in range(num_complete,flags.num_seeds):
				model_d["model_seed"] = model_seeds[i]
				run_d["train_seed"] = train_seeds[i]
				run_d["cuda_deterministic"] = False
				job_id_i = f"{flags.job_id}_{i}"
				run_name_i = f"{run_name}_{i}"
				init_or_resume_wandb_run(
					account_name=account_name,
					project_name=project_name,
					run_name=run_name_i,
					data_d=data_d,
					model_d=model_d,
					run_d=run_d,
					wandb_meta_dp=flags.wandb_meta_dp,
					job_id=job_id_i,
					job_id_dp=flags.job_id_dp,
					group_name=group_name
				)
				num_complete += 1
				with open(job_id_fp,"w") as file:
					yaml.dump(dict(num_complete=num_complete),file)
			# cleanup
			os.remove(job_id_fp)
		else:
			# just run one
			init_or_resume_wandb_run(
				account_name=account_name,
				project_name=project_name,
				run_name=run_name,
				data_d=data_d,
				model_d=model_d,
				run_d=run_d,
				wandb_meta_dp=flags.wandb_meta_dp,
				job_id=flags.job_id,
				job_id_dp=flags.job_id_dp
			)

	else:

		train(data_d,model_d,run_d,use_wandb)
