import torch as th
import torch.utils.data as th_data
import torch.nn.functional as F
import json
import pandas as pd
import numpy as np
import os
import dgl
import dgllife.utils as chemutils
import torch_geometric.data

from misc_utils import np_temp_seed, np_one_hot, flatten_lol, none_or_nan
import data_utils
import gf_data_utils

EPS = np.finfo(np.float32).eps

def data_to_device(data_d,device,non_blocking):

	new_data_d = {}
	for k,v in data_d.items():
		if isinstance(v,th.Tensor) or isinstance(v,dgl.DGLGraph) or isinstance(v,gf_data_utils.Batch):
			new_data_d[k] = v.to(device,non_blocking=non_blocking)
		else:
			new_data_d[k] = v
	return new_data_d

class TrainSubset(th_data.Subset):

	def __getitem__(self, idx):
		return self.dataset.__getitem__(self.indices[idx],train=True)

class BaseDataset(th_data.Dataset):

	def __init__(self,*dset_types,**kwargs):

		self.is_fp_dset = "fp" in dset_types
		self.is_graph_dset = "graph" in dset_types
		self.is_seq_dset = "seq" in dset_types
		self.is_gf_dset = "gf" in dset_types
		assert (self.is_fp_dset or self.is_graph_dset or self.is_seq_dset or self.is_gf_dset)
		for k,v in kwargs.items():
			setattr(self,k,v)
		assert os.path.isdir(self.proc_dp), self.proc_dp
		self.spec_df = pd.read_pickle(os.path.join(self.proc_dp,"spec_df.pkl"))
		self.mol_df = pd.read_pickle(os.path.join(self.proc_dp,"mol_df.pkl"))
		self._select_spec()
		self._setup_spec_metadata_dicts()
		self._seq_setup()
		# use mol_id as index for speedy access
		self.mol_df = self.mol_df.set_index("mol_id",drop=False).sort_index()

	def _select_spec(self):

		masks = []
		# dataset mask
		dset_mask = self.spec_df["dset"].isin(self.primary_dset+self.secondary_dset)
		masks.append(dset_mask)
		# instrument type
		inst_type_mask = self.spec_df["inst_type"].isin(self.inst_type)
		masks.append(inst_type_mask)
		# frag mode
		frag_mode_mask = self.spec_df["frag_mode"].isin(self.frag_mode)
		masks.append(frag_mode_mask)
		# ion mode
		ion_mode_mask = self.spec_df["ion_mode"] == self.ion_mode
		masks.append(ion_mode_mask)
		# precursor type
		if self.ion_mode == "P":
			prec_type_mask = self.spec_df["prec_type"].isin(self.pos_prec_type)
		else:
			assert self.ion_mode == "N", self.ion_mode
			prec_type_mask  = self.spec_df["prec_type"].isin(self.neg_prec_type)
		masks.append(prec_type_mask)
		# resolution
		if self.res != []:
			res_mask = self.spec_df["res"].isin(self.res)
			masks.append(res_mask)
		# collision energy
		ce_mask = ~self.spec_df[self.ce_key].isna()
		masks.append(ce_mask)
		# spectrum type 
		spec_type_mask = self.spec_df["spec_type"] == "MS2"
		masks.append(spec_type_mask)
		# maximum mz allowed
		mz_mask = self.spec_df["peaks"].apply(lambda peaks: max(peak[0] for peak in peaks) < self.mz_max)
		masks.append(mz_mask)
		# precursor mz
		prec_mz_mask = ~self.spec_df["prec_mz"].isna()
		masks.append(prec_mz_mask)
		# single molecule
		multi_mol_ids = self.mol_df[self.mol_df["smiles"].str.contains("\.")]["mol_id"]
		single_mol_mask = ~self.spec_df["mol_id"].isin(multi_mol_ids)
		masks.append(single_mol_mask)
		# put them together
		all_mask = masks[0]
		for mask in masks[1:]:
			all_mask = all_mask & mask
		if np.sum(all_mask) == 0:
			raise ValueError("select removed all items")
		self.spec_df = self.spec_df[all_mask].reset_index(drop=True)
		# subsample
		if self.subsample_size > 0:
			self.spec_df = self.spec_df.groupby("mol_id").sample(n=self.subsample_size,random_state=self.subsample_seed,replace=True)
			self.spec_df = self.spec_df.reset_index(drop=True)
		else:
			self.spec_df = self.spec_df
		# num_entries
		if self.num_entries > 0:
			self.spec_df = self.spec_df.sample(n=self.num_entries,random_state=self.subsample_seed,replace=False)
			self.spec_df = self.spec_df.reset_index(drop=True)
		# # only keep mols with spectra
		self.mol_df = self.mol_df[self.mol_df["mol_id"].isin(self.spec_df["mol_id"])]
		self.mol_df = self.mol_df.reset_index(drop=True)

	def _setup_spec_metadata_dicts(self):

		# featurize spectral metadata
		# we can assume that the dataset is filtered (using the method above) to only include these values
		inst_type_list = self.inst_type
		if self.ion_mode == "P":
			prec_type_list = self.pos_prec_type
		else:
			assert self.ion_mode == "N", self.ion_mode
			prec_type_list = self.neg_prec_type
		frag_mode_list = self.frag_mode
		self.inst_type_c2i = {string:i for i,string in enumerate(inst_type_list)}
		self.inst_type_i2c = {i:string for i,string in enumerate(inst_type_list)}
		self.prec_type_c2i = {string:i for i,string in enumerate(prec_type_list)}
		self.prec_type_i2c = {i:string for i,string in enumerate(prec_type_list)}
		self.frag_mode_c2i = {string:i for i,string in enumerate(frag_mode_list)}
		self.frag_mode_i2c = {i:string for i,string in enumerate(frag_mode_list)}
		self.num_inst_type = len(inst_type_list)
		self.num_prec_type = len(prec_type_list)
		self.num_frag_mode = len(frag_mode_list)
		self.max_ce = self.spec_df[self.ce_key].max()
		self.mean_ce = self.spec_df[self.ce_key].mean()
		self.std_ce = self.spec_df[self.ce_key].std()

	def _seq_setup(self):

		if self.is_seq_dset:
			if self.selfies_encoding:
				with open(os.path.join(self.proc_dp,"selfies_c2i.json"),"r") as file:
					self.c2i = json.load(file)
				with open(os.path.join(self.proc_dp,"selfies_i2c.json"),"r") as file:
					self.i2c = json.load(file)
			else:
				with open(os.path.join(self.proc_dp,"smiles_c2i.json"),"r") as file:
					self.c2i = json.load(file)
				with open(os.path.join(self.proc_dp,"smiles_i2c.json"),"r") as file:
					self.i2c = json.load(file)
			self.alphabet_size = len(self.c2i)+2 # +2 for unknown and pad
			with np_temp_seed(1337):
				self.can_seeds = np.random.choice(np.arange(0,1000000),size=self.num_canonicals,replace=False)
				self.can_seeds[0] = -1

	def __getitem__(self,idx,train=False):

		spec_entry = self.spec_df.iloc[idx]
		mol_id = spec_entry["mol_id"]
		# mol_entry = self.mol_df[self.mol_df["mol_id"] == mol_id].iloc[0]
		mol_entry = self.mol_df.loc[mol_id]
		data = self.process_entry(spec_entry,mol_entry["mol"],train=train)
		return data

	def __len__(self):

		return self.spec_df.shape[0]

	def bin_func(self,mzs,ints,return_index=False):

		# import pdb; pdb.set_trace()
		mzs = np.array(mzs,dtype=np.float32)
		bins = np.arange(self.mz_bin_res,self.mz_max+self.mz_bin_res,step=self.mz_bin_res).astype(np.float32)
		bin_idx = np.searchsorted(bins,mzs,side="right")
		if return_index:
			return bin_idx.tolist()
		else:
			ints = np.array(ints,dtype=np.float32)
			bin_spec = np.zeros([len(bins)],dtype=np.float32)
			for i in range(len(mzs)):
				if bin_idx[i] < len(bin_spec) and ints[i] >= self.ints_thresh:
					bin_spec[bin_idx[i]] = max(bin_spec[bin_idx[i]],ints[i])
			if np.all(bin_spec == 0.):
				import pdb; pdb.set_trace()
			return bin_spec

	def transform_func(self,spec):

		# scale spectrum so that max value is 1000
		spec = spec * (1000. / np.max(spec))
		# remove noise
		spec = spec * (spec > self.ints_thresh*np.max(spec)).astype(float)
		# transform signal
		if self.transform == "log10":
			spec = np.log10(spec + 1)
		elif self.transform == "log10over3":
			spec = np.log10(spec + 1) / 3
		elif self.transform == "loge":
			spec = np.log(spec + 1)
		elif self.transform == "sqrt":
			spec = np.sqrt(spec)
		elif self.transform == "linear":
			raise NotImplementedError
		elif self.transform == "none":
			pass
		else:
			raise ValueError("invalid transform")
		# normalize
		if self.spectrum_normalization == "l1":
			spec = spec / np.sum(np.abs(spec))
		elif self.spectrum_normalization == "l2":
			spec = spec / np.sqrt(np.sum(spec**2))
		elif self.spectrum_normalization == "none":
			pass
		else:
			raise ValueError("invalid spectrum_normalization")
		return spec

	def get_split_masks(self,val_frac,test_frac,split_key,split_seed):

		assert split_key in ["inchikey_s","scaffold"], split_key
		# primary
		prim_mask = self.spec_df["dset"].isin(self.primary_dset)
		prim_mol_id = self.spec_df[prim_mask]["mol_id"].unique()
		prim_key = set(self.mol_df[self.mol_df["mol_id"].isin(prim_mol_id)][split_key])
		# secondary
		sec_mask = self.spec_df["dset"].isin(self.secondary_dset)
		sec_mol_id = self.spec_df[sec_mask]["mol_id"].unique()
		sec_key = set(self.mol_df[self.mol_df["mol_id"].isin(sec_mol_id)][split_key])
		# get keys (secondary might same compounds as primary does!)
		prim_only_key = prim_key - sec_key
		sec_only_key = sec_key
		prim_key_list = sorted(list(prim_key))
		prim_only_key_list = sorted(list(prim_only_key))
		both_key = prim_key & sec_key
		# split keys
		test_num = round(len(prim_only_key_list)*test_frac)
		val_num = round(len(prim_only_key_list)*val_frac)
		with np_temp_seed(split_seed):
			test_key = set(np.random.choice(prim_only_key_list,size=test_num,replace=False))
			train_val_key = prim_only_key - test_key
			val_key = set(np.random.choice(sorted(list(train_val_key)),size=val_num,replace=False))
			train_key = train_val_key - val_key
			assert len(train_key & sec_only_key) == 0
			assert len(val_key & sec_only_key) == 0
			assert len(test_key & sec_only_key) == 0
		# get ids and create masks
		train_mol_id = self.mol_df["mol_id"][self.mol_df[split_key].isin(list(train_key))].unique()
		val_mol_id = self.mol_df["mol_id"][self.mol_df[split_key].isin(list(val_key))].unique()
		test_mol_id = self.mol_df["mol_id"][self.mol_df[split_key].isin(list(test_key))].unique()
		train_mask = self.spec_df["mol_id"].isin(train_mol_id)
		val_mask = self.spec_df["mol_id"].isin(val_mol_id)
		test_mask = self.spec_df["mol_id"].isin(test_mol_id)
		sec_masks = [self.spec_df["dset"] == dset for dset in self.secondary_dset]
		assert (train_mask & val_mask & test_mask).sum() == 0
		print("> primary")
		print("split: train, val, test, sec, total")
		print(f"spec: {train_mask.sum()}, {val_mask.sum()}, {test_mask.sum()}, {sec_mask.sum()}, {len(self.spec_df)}")
		print(f"mol: {len(train_mol_id)}, {len(val_mol_id)}, {len(test_mol_id)}, {len(sec_mol_id)}, {self.spec_df['mol_id'].nunique()}")
		print("> secondary")
		for sec_idx,sec_dset in enumerate(self.secondary_dset):
			cur_sec = self.spec_df[sec_masks[sec_idx]]
			cur_sec_mol_id = cur_sec["mol_id"]
			cur_both_mol_mask = self.spec_df["mol_id"].isin(set(prim_mol_id)&set(cur_sec_mol_id))
			cur_prim_both = self.spec_df[prim_mask&cur_both_mol_mask]
			cur_sec_both = self.spec_df[sec_mask&cur_both_mol_mask]
			print(f"{sec_dset} spec = {cur_sec.shape[0]}, mol = {cur_sec_mol_id.nunique()}")
			print(f"{sec_dset} overlap: prim spec = {cur_prim_both.shape[0]}, sec spec = {cur_sec_both.shape[0]}, mol = {cur_prim_both['mol_id'].nunique()}")
		all_both_mol_mask = self.spec_df["mol_id"].isin(set(prim_mol_id)&set(sec_mol_id))
		all_prim_both = self.spec_df[prim_mask&all_both_mol_mask]
		all_sec_both = self.spec_df[sec_mask&all_both_mol_mask]
		print(f"all overlap: prim spec = {all_prim_both.shape[0]}, sec spec = {all_sec_both.shape[0]}, mol = {all_prim_both['mol_id'].nunique()}")
		return train_mask, val_mask, test_mask, sec_masks

	def get_spec_feats(self,spec_entry):

		# convert to a dense vector
		mol_id = th.tensor(spec_entry["mol_id"]).unsqueeze(0)
		spectrum_id = th.tensor(spec_entry["spectrum_id"]).unsqueeze(0)
		mzs = [peak[0] for peak in spec_entry["peaks"]]
		ints = [peak[1] for peak in spec_entry["peaks"]]
		prec_mz = spec_entry["prec_mz"]
		prec_mz_bin = self.bin_func([prec_mz],None,return_index=True)[0]
		prec_diff = max(mz-prec_mz for mz in mzs)
		num_peaks = len(mzs)
		bin_spec = self.transform_func(self.bin_func(mzs,ints))
		spec = th.as_tensor(bin_spec,dtype=th.float32).unsqueeze(0)
		col_energy = spec_entry[self.ce_key]
		inst_type = spec_entry["inst_type"]
		prec_type = spec_entry["prec_type"]
		frag_mode = spec_entry["frag_mode"]
		inst_type_idx = self.inst_type_c2i[inst_type]
		prec_type_idx = self.prec_type_c2i[prec_type]
		frag_mode_idx = self.frag_mode_c2i[frag_mode]
		if self.preproc_ce == "normalize":
			col_energy_meta = th.tensor([(col_energy-self.mean_ce)/self.std_ce],dtype=th.float32)
		elif self.preproc_ce == "quantize":
			ce_bins = np.arange(0,161,step=20) # 8 bins
			ce_idx = np.digitize(col_energy,bins=ce_bins,right=False)
			col_energy_meta = th.ones([len(ce_bins)+1],dtype=th.float32)
			col_energy_meta[ce_idx] = 1.
		else:
			assert self.preproc_ce == "none", self.preproc_ce
			col_energy_meta = th.tensor([col_energy],dtype=th.float32)
		inst_type_meta = th.as_tensor(np_one_hot(inst_type_idx,num_classes=self.num_inst_type),dtype=th.float32)
		prec_type_meta = th.as_tensor(np_one_hot(prec_type_idx,num_classes=self.num_prec_type),dtype=th.float32)
		frag_mode_meta = th.as_tensor(np_one_hot(frag_mode_idx,num_classes=self.num_frag_mode),dtype=th.float32)
		spec_meta_list = [col_energy_meta,inst_type_meta,prec_type_meta,frag_mode_meta,col_energy_meta]
		# spec_meta_list = [prec_type_meta,col_energy_meta]
		spec_meta = th.cat(spec_meta_list,dim=0).unsqueeze(0)
		spec_feats = {
			"spec": spec,
			"prec_mz": [prec_mz],
			"prec_mz_bin": [prec_mz_bin],
			"prec_diff": [prec_diff],
			"num_peaks": [num_peaks],
			"inst_type": [inst_type],
			"prec_type": [prec_type],
			"frag_mode": [frag_mode],
			"col_energy": [col_energy],
			"spec_meta": spec_meta,
			"mol_id": mol_id,
			"spectrum_id": spectrum_id
		}
		return spec_feats

	def _get_mb_prf(self,bin_mbs,spec):

		spec = spec[0].numpy()
		binary_spec = (spec > 0.).astype(np.float32)
		binary_mb = np.zeros_like(binary_spec)
		for mb in bin_mbs:
			for idx in range(mb[0],mb[1]):
				binary_mb[idx] = 1.
		# compute precision
		if np.sum(binary_mb) == 0.:
			bin_prec = np.nan
		else:
			bin_prec = np.sum(binary_spec*binary_mb) / np.sum(binary_mb)
		# compute recall
		if np.sum(binary_spec) == 0.:
			bin_rec = np.nan
		else:
			bin_rec = np.sum(binary_spec*binary_mb) / np.sum(binary_spec)
		# compute intensity fraction
		if np.sum(spec) == 0.:
			bin_irec = np.nan
		else:
			bin_irec = np.sum(spec*binary_mb) / np.sum(spec)
		return bin_prec, bin_rec, bin_irec

	def get_dataloaders(self,run_d):

		val_frac = run_d["val_frac"]
		test_frac = run_d["test_frac"]
		split_key = run_d["split_key"]
		split_seed = run_d["split_seed"]
		batch_size = run_d["batch_size"]
		num_workers = run_d["num_workers"]
		pin_memory = run_d["pin_memory"]
		num_track = run_d["num_track"]

		train_mask, val_mask, test_mask, sec_masks = self.get_split_masks(val_frac,test_frac,split_key,split_seed)
		all_idx = np.arange(len(self))
		train_ss = TrainSubset(self,all_idx[train_mask]) #th_data.RandomSampler()
		val_ss = th_data.Subset(self,all_idx[val_mask]) #th_data.RandomSampler(th_data.Subset(self,all_idx[val_mask]))
		test_ss = th_data.Subset(self,all_idx[test_mask]) #th_data.RandomSampler(th_data.Subset(self,all_idx[test_mask]))
		sec_ss = [th_data.Subset(self,all_idx[sec_mask]) for sec_mask in sec_masks]

		collate_fn = self.get_collate_fn()
		train_dl = th_data.DataLoader(
			train_ss,
			batch_size=batch_size,
			collate_fn=collate_fn,
			num_workers=num_workers,
			pin_memory=pin_memory,
			shuffle=True,
			drop_last=True # this is to prevent single data batches that mess with batchnorm
		)
		train_dl_2 = th_data.DataLoader(
			train_ss,
			batch_size=batch_size,
			collate_fn=collate_fn,
			num_workers=num_workers,
			pin_memory=pin_memory,
			shuffle=False,
			drop_last=False
		)
		val_dl = th_data.DataLoader(
			val_ss,
			batch_size=batch_size,
			collate_fn=collate_fn,
			num_workers=num_workers,
			pin_memory=pin_memory,
			shuffle=False,
			drop_last=False
		)
		test_dl = th_data.DataLoader(
			test_ss,
			batch_size=batch_size,
			collate_fn=collate_fn,
			num_workers=num_workers,
			pin_memory=pin_memory,
			shuffle=False,
			drop_last=False
		)
		sec_dls = []
		for ss in sec_ss:
			dl = th_data.DataLoader(
				ss,
				batch_size=batch_size,
				collate_fn=collate_fn,
				num_workers=num_workers,
				pin_memory=pin_memory,
				shuffle=False,
				drop_last=False
			)
			sec_dls.append(dl)

		dl_dict = {}
		dl_dict["primary"] = {
			"train": train_dl,
			"val": val_dl,
			"test": test_dl,
			"train_2": train_dl_2
		}
		dl_dict["secondary"] = {}
		for sec_idx,sec_dset in enumerate(self.secondary_dset):
			dl_dict["secondary"][f"{sec_dset}"] = sec_dls[sec_idx]

		return dl_dict

	def get_track_dl(self,idx,num_rand_idx=0,topk_idx=None,bottomk_idx=None,other_idx=None):

		# import pdb; pdb.set_trace()
		track_seed = 5585
		track_dl_dict = {}
		collate_fn = self.get_collate_fn()
		if num_rand_idx > 0:
			with np_temp_seed(track_seed):
				rand_idx = np.random.choice(idx,size=num_rand_idx,replace=False)
			rand_dl = th_data.DataLoader(
				th_data.Subset(self,rand_idx),
				batch_size=1,
				collate_fn=collate_fn,
				num_workers=0,
				pin_memory=False,
				shuffle=False,
				drop_last=False
			)
			track_dl_dict["rand"] = rand_dl
		if not (topk_idx is None):
			topk_idx = idx[topk_idx]
			topk_dl = th_data.DataLoader(
			th_data.Subset(self,topk_idx),
				batch_size=1,
				collate_fn=collate_fn,
				num_workers=0,
				pin_memory=False,
				shuffle=False,
				drop_last=False
			)
			track_dl_dict["topk"] = topk_dl
		if not (bottomk_idx is None):
			bottomk_idx = idx[bottomk_idx]
			bottomk_dl = th_data.DataLoader(
			th_data.Subset(self,bottomk_idx),
				batch_size=1,
				collate_fn=collate_fn,
				num_workers=0,
				pin_memory=False,
				shuffle=False,
				drop_last=False
			)
			track_dl_dict["bottomk"] = bottomk_dl
		if not (other_idx is None):
			other_idx = idx[other_idx]
			other_dl = th_data.DataLoader(
				th_data.Subset(self,other_idx),
				batch_size=1,
				collate_fn=collate_fn,
				num_workers=0,
				pin_memory=False,
				shuffle=False,
				drop_last=False
			)
			track_dl_dict["other"] = other_dl
		return track_dl_dict

	def get_data_dims(self):

		data = self.__getitem__(0)
		dim_d = {}
		if self.is_fp_dset:
			fp_dim = data["fp"].shape[1]
		else:
			fp_dim = -1
		if self.is_graph_dset:
			# node
			if self.atom_feature_mode == "pretrain":
				n_dim = -1
			else:
				n_dim = data["graph"].ndata['h'].shape[1]
			# edge
			if self.bond_feature_mode == "none":
				e_dim = 0
			elif self.bond_feature_mode == "pretrain":
				e_dim = -1
			else:
				e_dim = data["graph"].edata['h'].shape[1]
		elif self.is_gf_dset:
			n_dim, e_dim = gf_data_utils.compute_embed_sizes()
		else:
			n_dim = e_dim = -1
		if self.is_seq_dset:
			c_dim = self.alphabet_size
			l_dim = self.max_seq_len
		else:
			c_dim = l_dim = -1
		if self.spec_meta_global:
			g_dim = data["spec_meta"].shape[1]
		else:
			g_dim = 0 # -1
		o_dim = data["spec"].shape[1]
		dim_d = {
			"fp_dim": fp_dim,
			"n_dim": n_dim,
			"e_dim": e_dim,
			"c_dim": c_dim,
			"l_dim": l_dim,
			"g_dim": g_dim,
			"o_dim": o_dim
		}
		return dim_d

	def get_collate_fn(self):

		def _collate(data_ds):
			# check for rebatching
			if isinstance(data_ds[0],list):
				data_ds = flatten_lol(data_ds)
			assert isinstance(data_ds[0],dict)
			batch_data_d = {k:[] for k in data_ds[0].keys()}
			for data_d in data_ds:
				for k,v in data_d.items():
					batch_data_d[k].append(v)
			for k,v in batch_data_d.items():
				if isinstance(data_ds[0][k],th.Tensor):
					batch_data_d[k] = th.cat(v,dim=0)
				elif isinstance(data_ds[0][k],list):
					batch_data_d[k] = flatten_lol(v)
				elif isinstance(data_ds[0][k],dgl.DGLGraph):
					batch_data_d[k] = dgl.batch(v)
				elif isinstance(data_ds[0][k],torch_geometric.data.Data):
					batch_data_d[k] = gf_data_utils.collator(
						v,
						max_node=self.max_node,
						multi_hop_max_dist=self.multi_hop_max_dist,
						spatial_pos_max=self.spatial_pos_max
					)
				else:
					raise ValueError
			return batch_data_d

		return _collate

	def process_entry(self,spec_entry,mol,train=False):

		# initialize data with shared attributes
		spec_feats = self.get_spec_feats(spec_entry)
		data = {**spec_feats}
		mol_entry = self.mol_df.loc[spec_entry["mol_id"]]
		data["smiles"] = [data_utils.mol_to_smiles(mol)]
		# add dset_type specific attributes
		if self.is_fp_dset:
			assert len(self.fp_types) > 0, self.fp_types
			fps = []
			if "morgan" in self.fp_types:
				fp = data_utils.make_morgan_fingerprint(mol)
				fps.append(fp)
			if "rdkit" in self.fp_types:
				fp = data_utils.make_rdkit_fingerprint(mol)
				fps.append(fp)
			if "maccs" in self.fp_types:
				fp = data_utils.make_maccs_fingerprint(mol)
				fps.append(fp)
			fps = th.cat([th.as_tensor(fp,dtype=th.float32) for fp in fps],dim=0).unsqueeze(0)
			data["fp"] = fps
		if self.is_graph_dset:
			atom_featurizer = self.get_atom_featurizer()
			bond_featurizer = self.get_bond_featurizer()
			graph = chemutils.mol_to_bigraph(
				mol, 
				node_featurizer = atom_featurizer, 
				edge_featurizer = bond_featurizer,
				add_self_loop = self.self_loop,
				num_virtual_nodes = 0 #self.num_virtual_nodes
			)
			if self.spec_meta_node:
				assert self.atom_feature_mode not in ["pretrain"], self.atom_feature_mode
				spec_meta_node = spec_feats["spec_meta"].repeat(graph.num_nodes(),1)
				graph.ndata['h'] = th.cat((graph.ndata['h'], spec_meta_node), dim=1)
			data["graph"] = graph
		if self.is_seq_dset:
			smiles_str = data_utils.mol_to_smiles(mol,canonical=True,isomericSmiles=False,kekuleSmiles=False)
			if none_or_nan(smiles_str):
				import pdb; pdb.set_trace()
			if self.num_canonicals > 1 and train:
				# sample a different canonicalization (only in training)
				# using torch for random instead of numpy
				can_seed_idx = int(th.randint(self.num_canonicals,(1,)))
				can_seed = self.can_seeds[can_seed_idx]
				try:
					smiles_str = data_utils.randomize_smiles(smiles_str,can_seed)
				except:
					import pdb; pdb.set_trace()
			if self.selfies_encoding:
				selfies_str = data_utils.smiles_to_selfies(smiles_str)
				mol_seq = data_utils.split_selfies(selfies_str)
			else:
				mol_seq = data_utils.split_smiles(smiles_str)
			# unknown characters are mapped to self.alphabet_size-2
			mol_seq = th.tensor([self.c2i[c] if c in self.c2i else self.alphabet_size-2 for c in mol_seq],dtype=th.int64)
			mask = th.ones_like(mol_seq)
			if mol_seq.shape[0] > self.max_seq_len:
				if train and self.subsample_long_seq:
					# sample
					start_idx = th.randint(0,mol_seq.shape[0]-self.max_seq_len+1)
					end_idx = start_idx+self.max_seq_len
					mol_seq = mol_seq[start_idx:end_idx]
				else:
					# just take the first max_seq_len chars
					mol_seq = mol_seq[:self.max_seq_len]
				mask = mask[:self.max_seq_len]
			else:
				# padding is self.alphabet_size-1
				mol_seq = F.pad(mol_seq,(0,self.max_seq_len-mol_seq.shape[0]),mode="constant",value=self.alphabet_size-1)
				mask = F.pad(mask,(0,self.max_seq_len-mask.shape[0]),mode="constant",value=0)
			mol_seq = mol_seq.unsqueeze(0)
			mask = mask.unsqueeze(0).bool()
			data["seq"] = mol_seq
			data["mask"] = mask
		if self.is_gf_dset:
			gf_data = gf_data_utils.gf_preprocess(mol)
			data["gf_data"] = gf_data
		return data

	def get_atom_featurizer(self):

		assert self.is_graph_dset
		if self.atom_feature_mode == "canonical":
			return chemutils.CanonicalAtomFeaturizer()
		elif self.atom_feature_mode == "pretrain":
			return chemutils.PretrainAtomFeaturizer()
		elif self.atom_feature_mode == 'light':
			atom_featurizer_funs = chemutils.ConcatFeaturizer([
				chemutils.atom_mass,
				data_utils.atom_type_one_hot
			])
		elif self.atom_feature_mode == 'full':
			atom_featurizer_funs = chemutils.ConcatFeaturizer([
				chemutils.atom_mass,
				data_utils.atom_type_one_hot, 
				data_utils.atom_bond_type_one_hot,
				chemutils.atom_degree_one_hot, 
				chemutils.atom_total_degree_one_hot,
				chemutils.atom_explicit_valence_one_hot,
				chemutils.atom_implicit_valence_one_hot,
				chemutils.atom_hybridization_one_hot,
				chemutils.atom_total_num_H_one_hot,
				chemutils.atom_formal_charge_one_hot,
				chemutils.atom_num_radical_electrons_one_hot,
				chemutils.atom_is_aromatic_one_hot,
				chemutils.atom_is_in_ring_one_hot,
				chemutils.atom_chiral_tag_one_hot
			])
		elif self.atom_feature_mode == 'medium':
			atom_featurizer_funs = chemutils.ConcatFeaturizer([
				chemutils.atom_mass,
				data_utils.atom_type_one_hot, 
				data_utils.atom_bond_type_one_hot,
				chemutils.atom_total_degree_one_hot,
				chemutils.atom_total_num_H_one_hot,
				chemutils.atom_is_aromatic_one_hot,
				chemutils.atom_is_in_ring_one_hot,
			])
		else:
			raise ValueError(f"Invalid atom_feature_mode: {self.atom_feature_mode}")

		# atom_mass_fun = chemutils.ConcatFeaturizer(
		# 	[chemutils.atom_mass]
		# )

		return chemutils.BaseAtomFeaturizer(
			{"h": atom_featurizer_funs} #, "m": atom_mass_fun}
		)

	def get_bond_featurizer(self):
		
		assert self.is_graph_dset
		if self.bond_feature_mode == "canonical":
			return chemutils.CanonicalBondFeaturizer()
		elif self.bond_feature_mode == "pretrain":
			return chemutils.PretrainBondFeaturizer()
		elif self.bond_feature_mode == 'light':
			return chemutils.BaseBondFeaturizer(
				featurizer_funcs = {'h': chemutils.ConcatFeaturizer([
					chemutils.bond_type_one_hot
				])}, self_loop = self.self_loop
			)
		elif self.bond_feature_mode == 'full':
			return chemutils.CanonicalBondFeaturizer(
				bond_data_field='h', self_loop = self.self_loop
			)
		else:
			assert self.bond_feature_mode == 'none'
			return None

	def batch_from_smiles(self,smiles_list,ref_spec_entry):

		data_list = []
		for smiles in smiles_list:
			mol = data_utils.mol_from_smiles(smiles)
			assert not none_or_nan(mol)
			data = self.process_entry(ref_spec_entry,mol,train=False)
			data_list.append(data)
		collate_fn = self.get_collate_fn()
		batch_data = collate_fn(data_list)
		return batch_data

