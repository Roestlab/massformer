# Adapted from https://github.com/microsoft/Graphormer
# Copyright (c) Microsoft Corporation 2021
# See licenses/GRAPHORMER_LICENSE

import numpy as np
import torch as th
from torch_geometric.data import Data
import pyximport

pyximport.install(setup_args={'include_dirs': np.get_include()})
import algos

# atom type
a_list = ['C','O','N','F','S','Cl','Br','I','P']
a_map = {x:idx for idx,x in enumerate(a_list)}
# formal charge
f_list = [-3,-2,-1,0,1,2,3]
f_map = {f:idx for idx,f in enumerate(f_list)}
#number of bonded hydrogens
max_num_hs = 6
# bond type
b_list = [
	'SINGLE',
	'DOUBLE',
	'TRIPLE',
	'AROMATIC'
]
b_map = {e:idx for idx,e in enumerate(b_list)}
# offset
embed_offset = 64
num_node_embed = 3
num_edge_embed = 1

def pyg_zinc_feat(mol):

	xs = []
	for atom in mol.GetAtoms():
		x = []
		atom_type = atom.GetSymbol()
		if atom_type in a_map:
			x.append(a_map[atom_type])
		else:
			x.append(len(a_map))
		formal_charge = atom.GetFormalCharge()
		if formal_charge in f_map:
			x.append(f_map[formal_charge])
		else:
			x.append(len(f_map))
		num_hs = atom.GetTotalNumHs()
		if num_hs <= max_num_hs:
			x.append(num_hs)
		else:
			x.append(max_num_hs+1)
		xs.append(x)
	x = th.tensor(xs,dtype=th.long).view(-1, num_node_embed)

	edge_indices, edge_attrs = [], []
	for bond in mol.GetBonds():
		i = bond.GetBeginAtomIdx()
		j = bond.GetEndAtomIdx()
		e = []
		bond_type = bond.GetBondType()
		if bond_type in b_map:
			e.append(b_map[bond_type])
		else:
			e.append(len(b_map))
		edge_indices += [[i, j], [j, i]]
		edge_attrs += [e, e]
	edge_index = th.tensor(edge_indices)
	edge_index = edge_index.t().to(th.long).view(2, -1)
	edge_attr = th.tensor(edge_attrs, dtype=th.long).view(-1, num_edge_embed)

	# Sort indices.
	assert edge_index.numel() > 0
	perm = (edge_index[0] * x.size(0) + edge_index[1]).argsort()
	edge_index, edge_attr = edge_index[:, perm], edge_attr[perm]

	# Construct PyG Data object
	data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

	return data

def convert_to_single_emb(x, offset):
	feature_num = x.size(1) if len(x.size()) > 1 else 1
	feature_offset = 1 + \
		th.arange(0, feature_num * offset, offset, dtype=th.long)
	x = x + feature_offset
	return x

def compute_embed_sizes():

	return 1+num_node_embed*embed_offset, 1+num_edge_embed*embed_offset

def gf_preprocess(mol):

	data = pyg_zinc_feat(mol)

	edge_attr, edge_index, x = data.edge_attr, data.edge_index, data.x
	N = x.size(0)
	x = convert_to_single_emb(x,embed_offset)

	# node adj matrix [N, N] bool
	adj = th.zeros([N, N], dtype=th.bool)
	adj[edge_index[0, :], edge_index[1, :]] = True

	# edge feature here
	if len(edge_attr.size()) == 1:
		edge_attr = edge_attr[:, None]
	attn_edge_type = th.zeros([N, N, edge_attr.size(-1)], dtype=th.long)
	attn_edge_type[edge_index[0, :], edge_index[1, :]
				   ] = convert_to_single_emb(edge_attr,embed_offset) + 1

	shortest_path_result, path = algos.floyd_warshall(adj.numpy())
	max_dist = np.amax(shortest_path_result)
	assert max_dist < 80, max_dist
	edge_input = algos.gen_edge_input(max_dist, path, attn_edge_type.numpy())
	spatial_pos = th.from_numpy((shortest_path_result)).long()
	attn_bias = th.zeros(
		[N + 1, N + 1], dtype=th.float)  # with graph token

	# combine
	data.x = x
	data.adj = adj
	data.attn_bias = attn_bias
	data.attn_edge_type = attn_edge_type
	data.spatial_pos = spatial_pos
	data.in_degree = adj.long().sum(dim=1).view(-1)
	data.out_degree = adj.long().sum(dim=0).view(-1)
	data.edge_input = th.from_numpy(edge_input).long()
	# dummy variables
	data.y = th.tensor([0.])
	data.idx = 0

	return data

def pad_1d_unsqueeze(x, padlen):
	x = x + 1  # pad id = 0
	xlen = x.size(0)
	if xlen < padlen:
		new_x = x.new_zeros([padlen], dtype=x.dtype)
		new_x[:xlen] = x
		x = new_x
	return x.unsqueeze(0)


def pad_2d_unsqueeze(x, padlen):
	x = x + 1  # pad id = 0
	xlen, xdim = x.size()
	if xlen < padlen:
		new_x = x.new_zeros([padlen, xdim], dtype=x.dtype)
		new_x[:xlen, :] = x
		x = new_x
	return x.unsqueeze(0)


def pad_attn_bias_unsqueeze(x, padlen):
	xlen = x.size(0)
	if xlen < padlen:
		new_x = x.new_zeros(
			[padlen, padlen], dtype=x.dtype).fill_(float('-inf'))
		new_x[:xlen, :xlen] = x
		new_x[xlen:, :xlen] = 0
		x = new_x
	return x.unsqueeze(0)


def pad_edge_type_unsqueeze(x, padlen):
	xlen = x.size(0)
	if xlen < padlen:
		new_x = x.new_zeros([padlen, padlen, x.size(-1)], dtype=x.dtype)
		new_x[:xlen, :xlen, :] = x
		x = new_x
	return x.unsqueeze(0)


def pad_spatial_pos_unsqueeze(x, padlen):
	x = x + 1
	xlen = x.size(0)
	if xlen < padlen:
		new_x = x.new_zeros([padlen, padlen], dtype=x.dtype)
		new_x[:xlen, :xlen] = x
		x = new_x
	return x.unsqueeze(0)


def pad_3d_unsqueeze(x, padlen1, padlen2, padlen3):
	x = x + 1
	xlen1, xlen2, xlen3, xlen4 = x.size()
	if xlen1 < padlen1 or xlen2 < padlen2 or xlen3 < padlen3:
		new_x = x.new_zeros([padlen1, padlen2, padlen3, xlen4], dtype=x.dtype)
		new_x[:xlen1, :xlen2, :xlen3, :] = x
		x = new_x
	return x.unsqueeze(0)


class Batch():
	def __init__(self, idx, attn_bias, attn_edge_type, spatial_pos, in_degree, out_degree, x, edge_input, y):
		super(Batch, self).__init__()
		self.idx = idx
		self.in_degree, self.out_degree = in_degree, out_degree
		self.x, self.y = x, y
		self.attn_bias, self.attn_edge_type, self.spatial_pos = attn_bias, attn_edge_type, spatial_pos
		self.edge_input = edge_input

	def to(self, device, non_blocking=False):
		# NOTE: ignores non_blocking
		self.idx = self.idx.to(device)
		self.in_degree, self.out_degree = self.in_degree.to(
			device), self.out_degree.to(device)
		self.x, self.y = self.x.to(device), self.y.to(device)
		self.attn_bias, self.attn_edge_type, self.spatial_pos = self.attn_bias.to(
			device), self.attn_edge_type.to(device), self.spatial_pos.to(device)
		self.edge_input = self.edge_input.to(device)
		return self

	def __len__(self):
		return self.in_degree.size(0)


def collator(items, max_node=512, multi_hop_max_dist=20, spatial_pos_max=20):
	items = [
		item for item in items if item is not None and item.x.size(0) <= max_node]
	items = [(item.idx, item.attn_bias, item.attn_edge_type, item.spatial_pos, item.in_degree,
			  item.out_degree, item.x, item.edge_input[:, :, :multi_hop_max_dist, :], item.y) for item in items]
	idxs, attn_biases, attn_edge_types, spatial_poses, in_degrees, out_degrees, xs, edge_inputs, ys = zip(
		*items)

	for idx, _ in enumerate(attn_biases):
		attn_biases[idx][1:, 1:][spatial_poses[idx] >= spatial_pos_max] = float('-inf')
	max_node_num = max(i.size(0) for i in xs)
	max_dist = max(i.size(-2) for i in edge_inputs)
	y = th.cat(ys)
	x = th.cat([pad_2d_unsqueeze(i, max_node_num) for i in xs])
	edge_input = th.cat([pad_3d_unsqueeze(
		i, max_node_num, max_node_num, max_dist) for i in edge_inputs])
	attn_bias = th.cat([pad_attn_bias_unsqueeze(
		i, max_node_num + 1) for i in attn_biases])
	attn_edge_type = th.cat(
		[pad_edge_type_unsqueeze(i, max_node_num) for i in attn_edge_types])
	spatial_pos = th.cat([pad_spatial_pos_unsqueeze(i, max_node_num)
						for i in spatial_poses])
	in_degree = th.cat([pad_1d_unsqueeze(i, max_node_num)
						  for i in in_degrees])
	out_degree = th.cat([pad_1d_unsqueeze(i, max_node_num)
						   for i in out_degrees])
	idx = th.LongTensor(idxs)
	return Batch(
		idx=idx,
		attn_bias=attn_bias,
		attn_edge_type=attn_edge_type,
		spatial_pos=spatial_pos,
		in_degree=in_degree,
		out_degree=out_degree,
		x=x,
		edge_input=edge_input,
		y=y,
	)