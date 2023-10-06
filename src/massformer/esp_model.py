import torch as th
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree, softmax
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool, GlobalAttention, Set2Set, GINEConv
import torch.nn.functional as F
from torch_scatter import scatter_add
from torch_geometric.nn.inits import glorot, zeros

from massformer.model import reverse_prediction, mask_prediction_by_mass

# num_atom_type = 27 # including the extra mask tokens
# num_instrument_type = 7 # including the extra mask tokens
NUM_FP_SIZE = -1 #4096 # including the extra mask tokens
NUM_ONTOLOGY_SIZE = -1 #5861
# num_lda_size = 100
# num_bond_type = 5  # including aromatic and self-loop edge, and extra masked tokens


class GINConv(MessagePassing):
    """
    Extension of GIN aggregation to incorporate edge information by concatenation.

    Args:
        emb_dim (int): dimensionality of embeddings for nodes and edges.
        embed_input (bool): whether to embed input or not.


    See https://arxiv.org/abs/1810.00826
    """

    def __init__(self, emb_dim, num_bond_type, aggr="add"):
        super(GINConv, self).__init__()
        # multi-layer perceptron
        self.mlp = th.nn.Sequential(th.nn.Linear(emb_dim, 2 * emb_dim), th.nn.ReLU(),
                                       th.nn.Linear(2 * emb_dim, emb_dim))

        self.edge_embedding1 = th.nn.Embedding(num_bond_type, emb_dim)
        th.nn.init.xavier_uniform_(self.edge_embedding1.weight.data)

        self.conv = GINEConv(nn=self.mlp)

    def forward(self, x, edge_index, edge_attr):
        edge_embeddings = self.edge_embedding1(edge_attr.squeeze(1))
        x = self.conv(x, edge_index=edge_index, edge_attr=edge_embeddings)
        return x


# class GCNConv(MessagePassing):

#     def __init__(self, emb_dim, aggr="add"):
#         super(GCNConv, self).__init__()

#         self.emb_dim = emb_dim
#         self.linear = th.nn.Linear(emb_dim, emb_dim)
#         self.edge_embedding1 = th.nn.Embedding(num_bond_type, emb_dim)

#         th.nn.init.xavier_uniform_(self.edge_embedding1.weight.data)

#         self.aggr = aggr

#     def norm(self, edge_index, num_nodes, dtype):
#         ### assuming that self-loops have been already added in edge_index
#         edge_weight = th.ones((edge_index.size(1),), dtype=dtype,
#                                  device=edge_index.device)
#         row, col = edge_index
#         deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
#         deg_inv_sqrt = deg.pow(-0.5)
#         deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

#         return deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

#     def forward(self, x, edge_index, edge_attr):
#         # add self loops in the edge space
#         edge_index = add_self_loops(edge_index, num_nodes=x.size(0))

#         # add features corresponding to self-loop edges.
#         self_loop_attr = th.zeros(x.size(0), 2)
#         self_loop_attr[:, 0] = 4  # bond type for self-loop edge
#         self_loop_attr = self_loop_attr.to(edge_attr.device).to(edge_attr.dtype)
#         edge_attr = th.cat((edge_attr, self_loop_attr), dim=0)

#         edge_embeddings = self.edge_embedding1(edge_attr[:, 0])

#         norm = self.norm(edge_index, x.size(0), x.dtype)

#         x = self.linear(x)

#         return self.propagate(self.aggr, edge_index, x=x, edge_attr=edge_embeddings, norm=norm)

#     def message(self, x_j, edge_attr, norm):
#         return norm.view(-1, 1) * (x_j + edge_attr)


# class GATConv(MessagePassing):
#     def __init__(self, emb_dim, heads=2, negative_slope=0.2, aggr="add"):
#         super(GATConv, self).__init__()

#         self.aggr = aggr

#         self.emb_dim = emb_dim
#         self.heads = heads
#         self.negative_slope = negative_slope

#         self.weight_linear = th.nn.Linear(emb_dim, heads * emb_dim)
#         self.att = th.nn.Parameter(th.Tensor(1, heads, 2 * emb_dim))

#         self.bias = th.nn.Parameter(th.Tensor(emb_dim))

#         self.edge_embedding1 = th.nn.Embedding(num_bond_type, heads * emb_dim)

#         th.nn.init.xavier_uniform_(self.edge_embedding1.weight.data)

#         self.reset_parameters()

#     def reset_parameters(self):
#         glorot(self.att)
#         zeros(self.bias)

#     def forward(self, x, edge_index, edge_attr):
#         # add self loops in the edge space
#         edge_index = add_self_loops(edge_index, num_nodes=x.size(0))

#         # add features corresponding to self-loop edges.
#         self_loop_attr = th.zeros(x.size(0), 2)
#         self_loop_attr[:, 0] = 4  # bond type for self-loop edge
#         self_loop_attr = self_loop_attr.to(edge_attr.device).to(edge_attr.dtype)
#         edge_attr = th.cat((edge_attr, self_loop_attr), dim=0)

#         edge_embeddings = self.edge_embedding1(edge_attr[:, 0])

#         x = self.weight_linear(x).view(-1, self.heads, self.emb_dim)
#         return self.propagate(self.aggr, edge_index, x=x, edge_attr=edge_embeddings)

#     def message(self, edge_index, x_i, x_j, edge_attr):
#         edge_attr = edge_attr.view(-1, self.heads, self.emb_dim)
#         x_j += edge_attr

#         alpha = (th.cat([x_i, x_j], dim=-1) * self.att).sum(dim=-1)

#         alpha = F.leaky_relu(alpha, self.negative_slope)
#         alpha = softmax(alpha, edge_index[0])

#         return x_j * alpha.view(-1, self.heads, 1)

#     def update(self, aggr_out):
#         aggr_out = aggr_out.mean(dim=1)
#         aggr_out = aggr_out + self.bias

#         return aggr_out


# class GraphSAGEConv(MessagePassing):
#     def __init__(self, emb_dim, aggr="mean"):
#         super(GraphSAGEConv, self).__init__()

#         self.emb_dim = emb_dim
#         self.linear = th.nn.Linear(emb_dim, emb_dim)
#         self.edge_embedding1 = th.nn.Embedding(num_bond_type, emb_dim)

#         th.nn.init.xavier_uniform_(self.edge_embedding1.weight.data)

#         self.aggr = aggr

#     def forward(self, x, edge_index, edge_attr):
#         # add self loops in the edge space
#         edge_index = add_self_loops(edge_index, num_nodes=x.size(0))

#         # add features corresponding to self-loop edges.
#         self_loop_attr = th.zeros(x.size(0), 2)
#         self_loop_attr[:, 0] = 4  # bond type for self-loop edge
#         self_loop_attr = self_loop_attr.to(edge_attr.device).to(edge_attr.dtype)
#         edge_attr = th.cat((edge_attr, self_loop_attr), dim=0)

#         edge_embeddings = self.edge_embedding1(edge_attr[:, 0])

#         x = self.linear(x)

#         return self.propagate(self.aggr, edge_index, x=x, edge_attr=edge_embeddings)

#     def message(self, x_j, edge_attr):
#         return x_j + edge_attr

#     def update(self, aggr_out):
#         return F.normalize(aggr_out, p=2, dim=-1)


class GNN(th.nn.Module):
    """


    Args:
        num_layer (int): the number of GNN layers
        emb_dim (int): dimensionality of embeddings
        JK (str): last, concat, max or sum.
        max_pool_layer (int): the layer from which we use max pool rather than add pool for neighbor aggregation
        drop_ratio (float): dropout rate
        gnn_type: gin, gcn, graphsage, gat

    Output:
        node representations

    """

    def __init__(
            self, 
            num_layer, 
            emb_dim, 
            num_atom_type,
            num_bond_type,
            num_instrument_type,
            JK="last", 
            drop_ratio=0, 
            gnn_type="gin", 
            disable_fingerprint=False
        ):
        super(GNN, self).__init__()
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.num_atom_type = num_atom_type
        self.num_bond_type = num_bond_type
        self.num_instrument_type = num_instrument_type
        self.JK = JK

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        self.x_embedding1 = th.nn.Sequential(
            th.nn.Linear(num_atom_type, emb_dim),
            # th.nn.ReLU(),
            # th.nn.Linear(emb_dim, emb_dim)
        )

        self.x_embedding2 = th.nn.Sequential(
            th.nn.Linear(num_instrument_type, emb_dim),
            # th.nn.ReLU(),
            # th.nn.Linear(emb_dim, emb_dim)
        )

        self.disable_fingerprint = disable_fingerprint
        if not self.disable_fingerprint:
            raise NotImplementedError
            # self.concat_emb_mlp = th.nn.Sequential(
            #     th.nn.ReLU(),
            #     th.nn.Linear(emb_dim * 2, emb_dim),
            # )
            # self.x_embedding3 = th.nn.Sequential(
            #     th.nn.Linear(NUM_FP_SIZE, emb_dim),
            #     th.nn.Dropout(drop_ratio),
            #     th.nn.Linear(emb_dim, emb_dim),
            #     th.nn.ReLU(),
            #     th.nn.Dropout(drop_ratio),
            #     th.nn.Linear(emb_dim, emb_dim),
            # )

        ###List of MLPs
        self.gnns = th.nn.ModuleList()
        for layer in range(num_layer):
            if gnn_type == "gin":
                self.gnns.append(
                    GINConv(
                        emb_dim,
                        num_bond_type,
                        aggr="add"
                    )
                )
            else:
                raise NotImplementedError
            # elif gnn_type == "gcn":
            #     self.gnns.append(GCNConv(emb_dim))
            # elif gnn_type == "gat":
            #     self.gnns.append(GATConv(emb_dim))
            # elif gnn_type == "graphsage":
            #     self.gnns.append(GraphSAGEConv(emb_dim))

        self.batch_norms = th.nn.ModuleList()
        for layer in range(num_layer):
            self.batch_norms.append(th.nn.BatchNorm1d(emb_dim))

    # def forward(self, x, edge_index, edge_attr):
    def forward(self, *argv):
        x, edge_index, edge_attr, instrument, fp = argv[0], argv[1], argv[2], argv[3], argv[4]

        x1 = self.x_embedding1(x[:, :self.num_atom_type])
        x2 = self.x_embedding2(instrument)
        x = x1 + x2

        h_list = [x]
        for layer in range(self.num_layer):
            h = self.gnns[layer](h_list[layer], edge_index, edge_attr)
            h = self.batch_norms[layer](h)
            # h = F.dropout(F.relu(h), self.drop_ratio, training = self.training)
            if layer == self.num_layer - 1:
                # remove relu for the last layer
                h = F.dropout(h, self.drop_ratio, training=self.training)
            else:
                h = F.dropout(F.relu(h), self.drop_ratio, training=self.training)
            h_list.append(h)

        ### Different implementations of Jk-concat
        if self.JK == "concat":
            node_representation = th.cat(h_list, dim=1)
        elif self.JK == "last":
            node_representation = h_list[-1]
        elif self.JK == "max":
            h_list = [h.unsqueeze_(0) for h in h_list]
            node_representation = th.max(th.cat(h_list, dim=0), dim=0)[0]
        elif self.JK == "sum":
            h_list = [h.unsqueeze_(0) for h in h_list]
            node_representation = th.sum(th.cat(h_list, dim=0), dim=0)[0]

        if not self.disable_fingerprint:
            concat_emb = th.cat([node_representation, self.x_embedding3(fp)], dim=-1)
            node_representation = self.concat_emb_mlp(concat_emb)

        return node_representation


class GNN_graphpred(th.nn.Module):
    """
    Extension of GIN to incorporate edge information by concatenation.

    Args:
        num_layer (int): the number of GNN layers
        emb_dim (int): dimensionality of embeddings
        num_tasks (int): number of tasks in multi-task learning scenario
        drop_ratio (float): dropout rate
        JK (str): last, concat, max or sum.
        graph_pooling (str): sum, mean, max, attention, set2set
        gnn_type: gin, gcn, graphsage, gat

    See https://arxiv.org/abs/1810.00826
    JK-net: https://arxiv.org/abs/1806.03536
    """

    def __init__(self, 
                num_layer, 
                emb_dim, 
                num_tasks,
                num_atom_type,
                num_bond_type,
                num_instrument_type,
                num_lda_size,
                JK="last", 
                drop_ratio=0, 
                graph_pooling="mean", 
                gnn_type="gin",
                disable_two_step_pred=False,
                disable_reverse=False,
                disable_fingerprint=False,
                disable_mt_fingerprint=False,
                disable_mt_lda=False,
                disable_mt_ontology=False,
                correlation_mat_rank=5, 
                correlation_type=5,
                mt_lda_weight=0.01, 
                correlation_mix_residual_weight=0.3,
                new_reverse=False,
                prec_mass_offset=0.0):
        super(GNN_graphpred, self).__init__()
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK
        self.emb_dim = emb_dim
        self.num_tasks = num_tasks
        self.new_reverse = new_reverse
        self.prec_mass_offset = prec_mass_offset

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        self.gnn = GNN(
            num_layer, 
            emb_dim,
            num_atom_type,
            num_bond_type,
            num_instrument_type,
            JK=JK, 
            drop_ratio=drop_ratio, 
            gnn_type=gnn_type, 
            disable_fingerprint=disable_fingerprint
        )

        # Different kind of graph pooling
        if graph_pooling == "sum":
            self.pool = global_add_pool
        elif graph_pooling == "mean":
            self.pool = global_mean_pool
        elif graph_pooling == "max":
            self.pool = global_max_pool
        elif graph_pooling == "attention":
            if self.JK == "concat":
                self.pool = GlobalAttention(gate_nn=th.nn.Linear((self.num_layer + 1) * emb_dim, 1))
            else:
                self.pool = GlobalAttention(gate_nn=th.nn.Linear(emb_dim, 1))
        elif graph_pooling[:-1] == "set2set":
            set2set_iter = int(graph_pooling[-1])
            if self.JK == "concat":
                self.pool = Set2Set((self.num_layer + 1) * emb_dim, set2set_iter)
            else:
                self.pool = Set2Set(emb_dim, set2set_iter)
        else:
            raise ValueError("Invalid graph pooling type.")

        # For graph-level binary classification
        if graph_pooling[:-1] == "set2set":
            self.mult = 2
        else:
            self.mult = 1

        gnn_output_emb_size = self.mult * (self.num_layer + 1) * self.emb_dim if self.JK == "concat" else self.mult * self.emb_dim

        self.disable_two_step_pred = disable_two_step_pred
        if not self.disable_two_step_pred:
            raise NotImplementedError

        self.disable_reverse = disable_reverse
        if not self.disable_reverse:
            self.graph_pred_linear_reverse = th.nn.Sequential(th.nn.Linear(gnn_output_emb_size, self.num_tasks))
            self.gate = th.nn.Sequential(th.nn.Linear(gnn_output_emb_size, self.num_tasks), th.nn.Sigmoid())

        self.disable_mt_fingerprint = disable_mt_fingerprint
        if not self.disable_mt_fingerprint:
            raise NotImplementedError

        self.disable_mt_lda = disable_mt_lda
        if not self.disable_mt_lda:
            self.graph_pred_mt_lda = th.nn.Sequential(th.nn.Linear(gnn_output_emb_size, num_lda_size), th.nn.Softmax(dim=-1))

        self.disable_mt_ontology = disable_mt_ontology
        if not self.disable_mt_ontology:
            raise NotImplementedError

        self.correlation_mat_rank = correlation_mat_rank
        if self.correlation_mat_rank > 0:
            self.correlation_mat = th.nn.Parameter(th.randn([correlation_type, self.num_tasks, self.correlation_mat_rank]), requires_grad=True)
            self.correlation_belong = th.nn.Sequential(
                th.nn.Linear(emb_dim, emb_dim),
                th.nn.ReLU(),
                th.nn.Linear(emb_dim, correlation_type),
                th.nn.Softmax(dim=-1)
            )
            self.correlation_type = correlation_type
            self.correlation_mix_residual_weight = correlation_mix_residual_weight

        self.graph_pred_linear = th.nn.Sequential(th.nn.Linear(gnn_output_emb_size, self.num_tasks))

    def from_pretrained(self, model_file):
        raise NotImplementedError
        # self.gnn.load_state_dict(th.load(model_file))

    def forward(self, *argv):
        x, edge_index, edge_attr, batch, instrument, fp, shift = argv[0], argv[1], argv[2], argv[3], argv[4], argv[5], argv[6]

        node_representation = self.gnn(x, edge_index, edge_attr, instrument[batch], None) #fp[batch])

        pred_logit = self.pool(node_representation, batch)
        loss = 0.

        # multi task
        if self.training:
            if not self.disable_mt_fingerprint:
                raise NotImplementedError

            if not self.disable_mt_lda:
                pred_mt_lda = self.graph_pred_mt_lda(pred_logit)

            if not self.disable_mt_ontology:
                raise NotImplementedError
        else:
            pred_mt_lda = None

        pred_val = self.graph_pred_linear(pred_logit)

        if not self.disable_reverse:
            if self.new_reverse:
                pred_val_reverse = reverse_prediction(
                    self.graph_pred_linear_reverse(pred_logit),
                    shift,
                    self.prec_mass_offset)
                gate = self.gate(pred_logit)
                pred_val = pred_val * gate + pred_val_reverse * (1. - gate)
            else:
                shift = shift + 1
                pred_val_reverse = th.flip(self.graph_pred_linear_reverse(pred_logit), dims=[1])
                for i in range(len(shift)):
                    pred_val_reverse[i, :] = pred_val_reverse[i, :].roll(shift[i].item())
                    pred_val_reverse[i, shift[i]:] = 0
                gate = self.gate(pred_logit)
                pred_val = gate * pred_val + (1 - gate) * pred_val_reverse

        if not self.disable_two_step_pred:
            raise NotImplementedError
            # pred_binary = self.graph_binary_linear(pred_logit)
            # pred_val = pred_binary * pred_val

        pred_val = F.softplus(pred_val)

        if self.correlation_mat_rank > 0:
            y_belong = self.correlation_belong(pred_logit).unsqueeze(-1)
            y = pred_val.reshape([1, -1, self.num_tasks])
            y = y @ self.correlation_mat @ self.correlation_mat.transpose(-1, -2)
            y = y.transpose(0, 1)
            y = (y * y_belong).sum(-2)
            y = F.softplus(y)
            pred_val = (1.0 - self.correlation_mix_residual_weight) * y + self.correlation_mix_residual_weight * pred_val

        if not self.disable_reverse and self.new_reverse:
            pred_val = mask_prediction_by_mass(
                pred_val, shift, self.prec_mass_offset,
                mask_value=0.)

        return pred_val, pred_mt_lda

class ESPPredictor(th.nn.Module):

    def __init__(self, num_tasks, num_atom_type, num_bond_type, num_instrument_type, new_reverse, prec_mass_offset, **kwargs):

        super().__init__()
        self.model = GNN_graphpred(
            num_layer=3,
            emb_dim=1024,
            num_tasks=num_tasks,
            num_atom_type=num_atom_type,
            num_bond_type=num_bond_type,
            num_instrument_type=num_instrument_type,
            num_lda_size=100,
            JK="last",
            drop_ratio=0.3,
            graph_pooling="mean",
            gnn_type="gin",
            disable_two_step_pred=True,
            disable_reverse=False,
            disable_fingerprint=True,
            disable_mt_fingerprint=True,
            disable_mt_lda=False,
            disable_mt_ontology=True,
            correlation_mat_rank=100,
            correlation_type=5,
            mt_lda_weight=0.01,
            correlation_mix_residual_weight=0.7,
            new_reverse=new_reverse,
            prec_mass_offset=prec_mass_offset
        )
        
    def forward(self, data, return_lda_pred=False, **kwargs):

        g = data["esp_graph"]
        x, edge_index, edge_attr, batch = g.x, g.edge_index, g.edge_attr, g.batch
        instrument = data["esp_meta"]
        shift = data["prec_mz_idx"]
        fp = mt_ontology_feature = lda_feature = None, None, None
        argv = [x, edge_index, edge_attr, batch, instrument, fp, shift, lda_feature, mt_ontology_feature]
        pred, pred_lda = self.model(*argv)
        output = {"pred": pred}
        if return_lda_pred:
            output["lda_pred"] = pred_lda
        return output


def test_reverse():

    th.manual_seed(420)
    batch_size = 2
    spec_dim = 10 #1000

    pred_fwd = th.rand(batch_size,spec_dim)
    pred_rev = th.rand(batch_size,spec_dim)
    gate = th.rand(batch_size,spec_dim)
    prec_mz_idx = th.randint(0,spec_dim,(batch_size,))
    pred_mz_offset = 1 ## play with this

    # old implementation
    def old(pred_fwd,pred_rev,gate,prec_mz_idx,prec_mz_offset):

        shift = prec_mz_idx+1
        pred_val = pred_fwd
        pred_val_reverse = th.flip(pred_rev, dims=[1])
        for i in range(len(shift)):
            pred_val_reverse[i, :] = pred_val_reverse[i, :].roll(shift[i].item())
            pred_val_reverse[i, shift[i]:] = 0
        pred_val = gate * pred_val + (1 - gate) * pred_val_reverse
        return pred_val

    # new implementation
    def new(pred_fwd,pred_rev,gate,prec_mz_idx,prec_mz_offset):

        ff = pred_fwd
        fr = reverse_prediction(
            pred_rev,
            prec_mz_idx,
            prec_mz_offset)
        fg = gate
        fo = ff * fg + fr * (1. - fg)
        fo = mask_prediction_by_mass(
            fo, prec_mz_idx, prec_mz_offset)
        return fo

    pred_old = old(pred_fwd,pred_rev,gate,prec_mz_idx,pred_mz_offset)
    pred_new = new(pred_fwd,pred_rev,gate,prec_mz_idx,pred_mz_offset)

    print((pred_old == pred_new).all())

    import pdb; pdb.set_trace()


if __name__ == "__main__":

    from massformer.dataset import get_esp_dicts, get_dataloader, data_to_device
    from massformer.data_utils import ELEMENT_LIST

    data_d_ow, model_d_ow, run_d_ow = get_esp_dicts()
    model_d_ow["esp_new_reverse"] = True

    ds, dl_dict, data_d, model_d, run_d = get_dataloader(
        data_d_ow=data_d_ow,
        model_d_ow=model_d_ow,
        run_d_ow=run_d_ow
    )
    # print(run_d["device"])

    dim_d = ds.get_data_dims()

    model = ESPPredictor(
        num_tasks=dim_d["o_dim"], 
        num_atom_type=len(ELEMENT_LIST)+1, 
        num_bond_type=7,
        num_instrument_type=len(ds.prec_type_c2i)+1,
        new_reverse=model_d["esp_new_reverse"],
        prec_mass_offset=model_d["prec_mass_offset"]
    )
    model.train()
    # model.to(run_d["device"])

    train_dl = dl_dict["primary"]["train"]
    batch = next(iter(train_dl))
    # batch = data_to_device(batch,run_d["device"],run_d["non_blocking"])

    output = model(batch,return_lda_pred=run_d["lda_topic_loss"])
    pred = output["pred"]
    targ = batch["spec"]
    spec_loss = -F.cosine_similarity(pred,targ,dim=1)
    mean_spec_loss = th.mean(spec_loss,dim=0)
    # mean_loss.backward()
    print(mean_spec_loss)
    
    lda_pred = output["lda_pred"]
    lda_targ = batch["lda_topic"]
    lda_loss = -F.cosine_similarity(lda_pred,lda_targ,dim=1)
    mean_lda_loss = th.mean(lda_loss,dim=0)
    # mean_lda_loss.backward()
    print(mean_lda_loss)

    mean_loss = mean_spec_loss + run_d["lda_topic_loss_weight"]*mean_lda_loss
    print(mean_loss)
    mean_loss.backward()

    import pdb; pdb.set_trace()

    # test_reverse()