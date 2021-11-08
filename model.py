from misc_utils import th_temp_seed
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch_scatter as th_s
import numpy as np
import dgl
from dgl.nn import MaxPooling, AvgPooling, GlobalAttentionPooling
from dgllife.model.gnn.wln import WLN
from dgllife.model.gnn import GAT
from dgllife.model import load_pretrained
import copy
import math

"""
model takes in a batched Data object that has node_feats, edge_index, global_feats, spec

outputs the predicted spectrum

> best setup:
- 10 layers
- 64-dim hidden dim (throughout)
- linear embedding
- GLU to predict output
- the paper does not specify information about the FF NN (like skip connections, dimension, etc)
- LeakyReLU is used throughout since the GAT uses LeakyReLU
- num_heads was not specified (assuming 10)
- dropout was not specific (assuming 0.)
"""

class LinearBlock(nn.Module):

	def __init__(self, in_feats, out_feats, dropout=0.1):
		super(LinearBlock, self).__init__()
		self.linear = nn.Linear(in_feats, out_feats)
		self.bn = nn.BatchNorm1d(out_feats)
		self.dropout = nn.Dropout(dropout)
			
	def forward(self, x):
		return self.bn(self.dropout(F.relu(self.linear(x))))


class NeimsBlock(nn.Module):
	"""
	Adapted from: https://github.com/brain-research/deep-molecular-massspec
	Copyright (c) 2018 Google LLC
	See licenses/NEIMS_LICENSE
	"""

	def __init__(self,in_dim,out_dim,dropout):
		
		super(NeimsBlock, self).__init__()
		bottleneck_factor = 0.5
		bottleneck_size = int(round(bottleneck_factor*out_dim))
		self.in_batch_norm = nn.BatchNorm1d(in_dim)
		self.in_activation = nn.LeakyReLU()
		self.in_linear = nn.Linear(in_dim, bottleneck_size)
		self.out_batch_norm = nn.BatchNorm1d(bottleneck_size)
		self.out_linear = nn.Linear(bottleneck_size, out_dim)
		self.out_activation = nn.LeakyReLU()
		self.dropout = nn.Dropout(p=dropout)

	def forward(self,x):
		
		h = x
		h = self.in_batch_norm(h)
		h = self.in_activation(h)
		h = self.dropout(h)
		h = self.in_linear(h)
		h = self.out_batch_norm(h)
		h = self.out_activation(h)
		h = self.out_linear(h)
		return h


class Conv1dBlock(nn.Module):

	def __init__(self,in_channels,out_channels,kernel_size=5,stride=1,pool_type="avg",pool_size=4,batch_norm=True,dropout=0.1,activation="relu"):
		"""
		padding is always the same
		order of ops:
		1. conv
		2. batch norm (if applicable)
		3. activation
		4. pool (if applicable)
		5. dropout (if applicable)
		"""

		super(Conv1dBlock,self).__init__()
		self.in_channels = in_channels
		self.out_channels = out_channels
		ops_list = []
		assert kernel_size % 2 == 1, kernel_size
		padding = kernel_size // 2
		ops_list.append(nn.Conv1d(in_channels,out_channels,kernel_size,stride=stride,padding=padding))
		if batch_norm:
			ops_list.append(nn.BatchNorm1d(out_channels))
		if activation == "relu":
			ops_list.append(nn.ReLU())
		else:
			raise ValueError
		if pool_type == "max":
			ops_list.append(nn.MaxPool1d(pool_size,ceil_mode=True))
		elif pool_type == "avg":
			ops_list.append(nn.AvgPool1d(pool_size,ceil_mode=True))
		else:
			assert pool_type == "none", pool_type
		if dropout > 0.:
			ops_list.append(nn.Dropout(dropout))
		self.ops = nn.Sequential(*ops_list)

	def forward(self, b_x):

		assert b_x.ndim == 3
		assert b_x.shape[1] == self.in_channels
		b_y = self.ops(b_x)
		return b_y


class MyAvgPooling(nn.Module):
	
	def forward(self,graph,node_feat):

		# get mean readout
		batch_num_nodes = graph.batch_num_nodes()
		n_graphs = graph.batch_size
		seg_id = th.repeat_interleave(th.arange(n_graphs,device=node_feat.device).long(),batch_num_nodes,dim=0)
		mean_node_feat = th_s.scatter_mean(node_feat,seg_id,dim=0,dim_size=n_graphs)
		return mean_node_feat



class FPEmbedder(nn.Module):

	def __init__(self,dim_d,**kwargs):

		super(FPEmbedder,self).__init__()
		self.fp_dim = dim_d["fp_dim"]
		self.embed_dim = self.fp_dim

	def get_embed_dim(self):

		return self.embed_dim

	def forward(self,data):

		return data["fp"]

class GNNEmbedder(nn.Module):

	def __init__(self,dim_d,**kwargs):

		super(GNNEmbedder,self).__init__()
		self.n_dim = dim_d["n_dim"]
		self.e_dim = dim_d["e_dim"]
		for k,v in kwargs.items():
			setattr(self,k,v)

		# gnn layers go here
		self._gnn_layers = None
		self.gnn_layers = None

		if self.gnn_pool_type == "max":
			self._pool = MaxPooling()
		elif self.gnn_pool_type == "avg":
			self._pool = MyAvgPooling() #AvgPooling()
		elif self.gnn_pool_type == "attn":
			pooling_gate_nn = nn.Linear(self.gnn_h_dim, 1)
			self._pool = GlobalAttentionPooling(pooling_gate_nn)
		self._pool_activation = nn.ReLU()
		self.pool = lambda g, nh: self._pool_activation(self._pool(g,nh))
		self.embed_dim = self.gnn_h_dim

	def get_embed_dim(self):

		return self.embed_dim

	def forward(self,data):

		g = data["graph"]
		nh = g.ndata["h"]
		eh = g.edata["h"]
		nh = self.gnn_layers(g,nh,eh)
		gh = self.pool(g,nh)
		return gh


class WLNEmbedder(GNNEmbedder):

	def __init__(self, dim_d, **kwargs):

		super(WLNEmbedder,self).__init__(dim_d,**kwargs)

		self._gnn_layers = WLN(
			self.n_dim,
			self.e_dim,
			self.gnn_h_dim,
			self.gnn_num_layers
		)
		self.gnn_layers = lambda g,nh,eh: self._gnn_layers(g,nh,eh)


class Identity(nn.Module):
	""" 
	for deleting layers in pretrained models
	from https://discuss.pytorch.org/t/how-to-delete-layer-in-pretrained-model/17648 
	"""

	def __init__(self):
		super(Identity,self).__init__()

	def forward(self,x):
		return x


class GNIdentity(Identity):
	"""
	identity for operation that takes graph, node attributes (for example, graph pool layer)
	"""

	def forward(self,g,nh):
		return nh


class GNEIdentity(Identity):
	"""
	identity for operation that takes graph, node, edge attributes (for example. message passing layer)
	"""

	def forward(self,g,nh,eh):
		return nh
		

class CNNEmbedder(nn.Module):

	def __init__(self,dim_d,**kwargs):

		super(CNNEmbedder,self).__init__()
		self.c_dim = dim_d["c_dim"]-1 # get rid of padding dimension
		self.l_dim = dim_d["l_dim"] # maximum length
		for k,v in kwargs.items():
			setattr(self,k,v)

		conv_hp = dict(
			kernel_size=self.conv_kernel_size,
			stride=self.conv_stride,
			pool_type=self.conv_pool_type,
			pool_size=self.conv_pool_size,
			dropout=self.dropout,
			activation="relu"
		)

		def compute_conv_out_length(in_length,kernel_size,stride):
			padding = kernel_size // 2
			dilation = 1
			out_length = np.floor((in_length + 2*padding - dilation * (kernel_size-1) - 1) / stride + 1)
			return int(out_length)
		def compute_pool_out_length(in_length,kernel_size):
			if self.conv_pool_type == "none":
				return in_length
			else:
				padding = 0
				dilation = 1
				stride = kernel_size
				out_length = np.ceil((in_length + 2*padding - dilation * (kernel_size-1) - 1) / stride + 1)
				return int(out_length)
		
		conv_list = []
		conv_list.append(Conv1dBlock(self.c_dim,self.conv_num_kernels,**conv_hp))
		conv_out_length = compute_conv_out_length(self.l_dim,self.conv_kernel_size,self.conv_stride)
		pool_out_length = compute_pool_out_length(conv_out_length,self.conv_pool_size)
		# print(0,conv_out_length,pool_out_length)
		for i in range(self.conv_num_layers-1):
			conv_list.append(Conv1dBlock(self.conv_num_kernels,self.conv_num_kernels,**conv_hp))
			conv_out_length = compute_conv_out_length(pool_out_length,self.conv_kernel_size,self.conv_stride)
			pool_out_length = compute_pool_out_length(conv_out_length,self.conv_pool_size)
			# print(i+1,conv_out_length,pool_out_length)
		self.conv_layers = nn.Sequential(*conv_list)
		self.embed_dim = pool_out_length*self.conv_num_kernels

	def get_embed_dim(self):

		return self.embed_dim

	def forward(self, data):

		# assumes seq is padded with special self.c_dim+1 characters
		seq_oh = F.one_hot(data["seq"],num_classes=self.c_dim+1).float()
		seq_oh = seq_oh[:,:,:-1].transpose(1,2) # this becomes 0 padding
		batch = seq_oh.shape[0]
		ch = self.conv_layers(seq_oh)
		ch = ch.view(batch,-1)
		return ch


class Predictor(nn.Module):

	def __init__(self,dim_d,**kwargs):

		super(Predictor,self).__init__()
		self.g_dim = dim_d["g_dim"]
		self.o_dim = dim_d["o_dim"]
		for k,v in kwargs.items():
			setattr(self,k,v)
		self.embedders = nn.ModuleList([])
		if "fp" in self.embed_types:
			self.embedders.append(FPEmbedder(dim_d,**kwargs))
		if "wln" in self.embed_types:
			self.embedders.append(WLNEmbedder(dim_d,**kwargs))
		if "cnn" in self.embed_types:
			self.embedders.append(CNNEmbedder(dim_d,**kwargs))
		if "gf" in self.embed_types:
			self.embedders.append(GFEmbedder(dim_d))
		assert len(self.embedders) > 0, len(self.embedders)
		self.embed_dim = sum(embedder.get_embed_dim() for embedder in self.embedders)+self.g_dim
		self.ff_layers = nn.ModuleList([])
		if self.ff_layer_type == "standard":
			ff_layer = LinearBlock
		else:
			assert self.ff_layer_type == "neims", self.ff_layer_type
			ff_layer = NeimsBlock
		self.ff_layers.append(nn.Linear(self.embed_dim,self.ff_h_dim))
		for i in range(self.ff_num_layers):
			self.ff_layers.append(ff_layer(self.ff_h_dim,self.ff_h_dim,self.dropout))
		self.out_layer = nn.Linear(self.ff_h_dim,self.o_dim)
		if self.gate_prediction:
			self.out_gate = nn.Sequential(*[nn.Linear(self.ff_h_dim,self.o_dim),nn.Sigmoid()])

	def forward(self,data):

		# concatenate all embeddings
		fh = th.cat([embedder(data) for embedder in self.embedders],dim=1)
		# add on metadata
		if self.g_dim > 0:
			fh = th.cat([fh,data["spec_meta"]],dim=1)
		# apply feedforward layers
		fh = self.ff_layers[0](fh)
		for ff_layer in self.ff_layers[1:]:
			if self.ff_skip:
				fh = fh + ff_layer(fh)
			else:
				fh = ff_layer(fh)
		# apply output layer
		fo = self.out_layer(fh)
		# apply gating
		if self.gate_prediction:
			fg = self.out_gate(fh)
			fo = fg * fo
		# apply normalization		
		if self.output_normalization == "l1":
			fo = F.relu(fo)
			fo = F.normalize(fo,p=1,dim=1)
		elif self.output_normalization == "l2":
			fo = F.relu(fo)
			fo = F.normalize(fo,p=2,dim=1)
		elif self.output_normalization == "sm":
			fo = F.softmax(fo,dim=1)
		else:
			raise ValueError(f"invalid output_normalization: {self.output_normalization}")
		return fo

	def get_attn_mats(self,data):

		assert "gf" in self.embed_types, self.embed_types
		for embedder in self.embedders:
			if isinstance(embedder,GFEmbedder):
				return embedder.forward(data,return_attn_mats=True)[1]

### Graphormer stuff ###

def init_params(module, n_layers):
	if isinstance(module, nn.Linear):
		module.weight.data.normal_(mean=0.0, std=0.02 / math.sqrt(n_layers))
		if module.bias is not None:
			module.bias.data.zero_()
	if isinstance(module, nn.Embedding):
		module.weight.data.normal_(mean=0.0, std=0.02)


class GFEmbedder(nn.Module):

	"""
	Adapted from https://github.com/microsoft/Graphormer
	Copyright (c) Microsoft Corporation 2021
	See licenses/GRAPHORMER_LICENSE
	"""

	def __init__(
		self,
		dim_d,
		n_layers=12, #
		num_heads=8, #
		hidden_dim=80, #
		dropout_rate=0.1, #
		intput_dropout_rate=0.1,
		ffn_dim=80,
		edge_type="multi_hop",
		multi_hop_max_dist=20,
		attention_dropout_rate=0.1
	):
		super(GFEmbedder,self).__init__()
		self.n_dim = dim_d["n_dim"]
		self.e_dim = dim_d["e_dim"]
		self.num_heads = num_heads
		self.atom_encoder = nn.Embedding(self.n_dim, hidden_dim, padding_idx=0)
		self.edge_encoder = nn.Embedding(self.e_dim, num_heads, padding_idx=0)
		self.edge_type = edge_type
		if self.edge_type == 'multi_hop':
			self.edge_dis_encoder = nn.Embedding(
				40 * num_heads * num_heads, 1)
		self.spatial_pos_encoder = nn.Embedding(80, num_heads, padding_idx=0)
		self.in_degree_encoder = nn.Embedding(
			64, hidden_dim, padding_idx=0)
		self.out_degree_encoder = nn.Embedding(
			64, hidden_dim, padding_idx=0)
		self.input_dropout = nn.Dropout(intput_dropout_rate)
		encoders = [EncoderLayer(hidden_dim, ffn_dim, dropout_rate, attention_dropout_rate, num_heads)
					for _ in range(n_layers)]
		self.layers = nn.ModuleList(encoders)
		self.final_ln = nn.LayerNorm(hidden_dim)
		self.graph_token = nn.Embedding(1, hidden_dim)
		self.graph_token_virtual_distance = nn.Embedding(1, num_heads)
		self.multi_hop_max_dist = multi_hop_max_dist
		self.hidden_dim = hidden_dim
		self.automatic_optimization = True
		self.apply(lambda module: init_params(module, n_layers=n_layers))

	def get_embed_dim(self):
		return self.hidden_dim

	def forward(self, data, return_attn_mats=False):
		batched_data = data["gf_data"]
		attn_bias, spatial_pos, x = batched_data.attn_bias, batched_data.spatial_pos, batched_data.x
		in_degree, out_degree = batched_data.in_degree, batched_data.in_degree
		edge_input, attn_edge_type = batched_data.edge_input, batched_data.attn_edge_type
		# graph_attn_bias
		n_graph, n_node = x.size()[:2]
		graph_attn_bias = attn_bias.clone()
		graph_attn_bias = graph_attn_bias.unsqueeze(1).repeat(
			1, self.num_heads, 1, 1)  # [n_graph, n_head, n_node+1, n_node+1]
		# spatial pos
		# [n_graph, n_node, n_node, n_head] -> [n_graph, n_head, n_node, n_node]
		spatial_pos_bias = self.spatial_pos_encoder(spatial_pos).permute(0, 3, 1, 2)
		graph_attn_bias[:, :, 1:, 1:] = graph_attn_bias[:,
														:, 1:, 1:] + spatial_pos_bias
		# reset spatial pos here
		t = self.graph_token_virtual_distance.weight.view(1, self.num_heads, 1)
		graph_attn_bias[:, :, 1:, 0] = graph_attn_bias[:, :, 1:, 0] + t
		graph_attn_bias[:, :, 0, :] = graph_attn_bias[:, :, 0, :] + t
		# edge feature
		if self.edge_type == 'multi_hop':
			spatial_pos_ = spatial_pos.clone()
			spatial_pos_[spatial_pos_ == 0] = 1  # set pad to 1
			# set 1 to 1, x > 1 to x - 1
			spatial_pos_ = th.where(spatial_pos_ > 1, spatial_pos_ - 1, spatial_pos_)
			if self.multi_hop_max_dist > 0:
				spatial_pos_ = spatial_pos_.clamp(0, self.multi_hop_max_dist)
				edge_input = edge_input[:, :, :, :self.multi_hop_max_dist, :]
			# [n_graph, n_node, n_node, max_dist, n_head]
			edge_input = self.edge_encoder(edge_input).mean(-2)
			max_dist = edge_input.size(-2)
			edge_input_flat = edge_input.permute(
				3, 0, 1, 2, 4).reshape(max_dist, -1, self.num_heads)
			edge_input_flat = th.bmm(edge_input_flat, self.edge_dis_encoder.weight.reshape(
				-1, self.num_heads, self.num_heads)[:max_dist, :, :])
			edge_input = edge_input_flat.reshape(
				max_dist, n_graph, n_node, n_node, self.num_heads).permute(1, 2, 3, 0, 4)
			edge_input = (edge_input.sum(-2) /
						  (spatial_pos_.float().unsqueeze(-1))).permute(0, 3, 1, 2)
		else:
			# # [n_graph, n_node, n_node, n_head] -> [n_graph, n_head, n_node, n_node]
			# edge_input = self.edge_encoder(
			# 	attn_edge_type).mean(-2).permute(0, 3, 1, 2)
			raise ValueError
		graph_attn_bias[:, :, 1:, 1:] = graph_attn_bias[:,
														:, 1:, 1:] + edge_input
		graph_attn_bias = graph_attn_bias + attn_bias.unsqueeze(1)  # reset
		# node feauture + graph token
		node_feature = self.atom_encoder(x).sum(
			dim=-2)           # [n_graph, n_node, n_hidden]
		node_feature = node_feature + \
			self.in_degree_encoder(in_degree) + \
			self.out_degree_encoder(out_degree)
		graph_token_feature = self.graph_token.weight.unsqueeze(
			0).repeat(n_graph, 1, 1)
		graph_node_feature = th.cat(
			[graph_token_feature, node_feature], dim=1)
		# transformer encoder
		output = self.input_dropout(graph_node_feature)
		all_attn_mats = []
		for enc_layer in self.layers:
			output, attn_mat = enc_layer(output, graph_attn_bias)
			all_attn_mats.append(attn_mat)
		all_attn_mats = th.stack(all_attn_mats,dim=0).transpose(0,1)
		output = self.final_ln(output)[:, 0, :]
		if return_attn_mats:
			return output, all_attn_mats
		else:
			return output


class FeedForwardNetwork(nn.Module):
	
	def __init__(self, hidden_size, ffn_size, dropout_rate):
		super(FeedForwardNetwork, self).__init__()
		self.layer1 = nn.Linear(hidden_size, ffn_size)
		self.gelu = nn.GELU()
		self.layer2 = nn.Linear(ffn_size, hidden_size)

	def forward(self, x):
		x = self.layer1(x)
		x = self.gelu(x)
		x = self.layer2(x)
		return x


class MultiHeadAttention(nn.Module):

	def __init__(self, hidden_size, attention_dropout_rate, num_heads):
		super(MultiHeadAttention, self).__init__()
		self.num_heads = num_heads
		self.att_size = att_size = hidden_size // num_heads
		self.scale = att_size ** -0.5
		self.linear_q = nn.Linear(hidden_size, num_heads * att_size)
		self.linear_k = nn.Linear(hidden_size, num_heads * att_size)
		self.linear_v = nn.Linear(hidden_size, num_heads * att_size)
		self.att_dropout = nn.Dropout(attention_dropout_rate)
		self.output_layer = nn.Linear(num_heads * att_size, hidden_size)

	def forward(self, q, k, v, attn_bias=None):
		orig_q_size = q.size() # [b, q_len]
		d_k = self.att_size
		d_v = self.att_size
		batch_size = q.size(0)
		# head_i = Attention(Q(W^Q)_i, K(W^K)_i, V(W^V)_i)
		q = self.linear_q(q).view(batch_size, -1, self.num_heads, d_k) # [b, q_len, h, d_k]
		k = self.linear_k(k).view(batch_size, -1, self.num_heads, d_k) # [b, k_len, h, d_k]
		v = self.linear_v(v).view(batch_size, -1, self.num_heads, d_v) # [b, v_len, h, d_v]
		q = q.transpose(1, 2)                  # [b, h, q_len, d_k]
		v = v.transpose(1, 2)                  # [b, h, v_len, d_v], attn == d_v
		k = k.transpose(1, 2).transpose(2, 3)  # [b, h, d_k, k_len]
		# Scaled Dot-Product Attention.
		# Attention(Q, K, V) = softmax((QK^T)/sqrt(d_k))V
		q = q * self.scale
		x = th.matmul(q, k)  # [b, h, q_len, k_len]
		if attn_bias is not None:
			x = x + attn_bias
		x = th.softmax(x, dim=3)
		attn_mat = x
		x = self.att_dropout(x)
		x = x.matmul(v)  # [b, h, q_len, d_v]
		x = x.transpose(1, 2).contiguous()  # [b, q_len, h, d_v]
		x = x.view(batch_size, -1, self.num_heads * d_v) # [b, num_heads * d_v]
		x = self.output_layer(x) # [b, h]
		assert x.size() == orig_q_size
		return x, attn_mat


class EncoderLayer(nn.Module):
	
	def __init__(self, hidden_size, ffn_size, dropout_rate, attention_dropout_rate, num_heads):
		super(EncoderLayer, self).__init__()
		self.self_attention_norm = nn.LayerNorm(hidden_size)
		self.self_attention = MultiHeadAttention(
			hidden_size, attention_dropout_rate, num_heads)
		self.self_attention_dropout = nn.Dropout(dropout_rate)
		self.ffn_norm = nn.LayerNorm(hidden_size)
		self.ffn = FeedForwardNetwork(hidden_size, ffn_size, dropout_rate)
		self.ffn_dropout = nn.Dropout(dropout_rate)

	def forward(self, x, attn_bias=None):
		y = self.self_attention_norm(x)
		y, attn_mat = self.self_attention(y, y, y, attn_bias)
		y = self.self_attention_dropout(y)
		x = x + y
		y = self.ffn_norm(x)
		y = self.ffn(y)
		y = self.ffn_dropout(y)
		x = x + y
		return x, attn_mat
