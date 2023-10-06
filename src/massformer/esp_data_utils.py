import dgllife.utils as chemutils
import numpy as np
import torch as th
import torch_geometric

from massformer.data_utils import ELEMENT_LIST


def get_atom_featurizer(feature_mode, element_list):
    atom_mass_fun = chemutils.ConcatFeaturizer([chemutils.atom_mass])
    def atom_type_one_hot(atom):
        return chemutils.atom_type_one_hot(atom, allowable_set=element_list, encode_unknown=True)
    if feature_mode == 'medium':
        atom_featurizer_funs = chemutils.ConcatFeaturizer([
            chemutils.atom_mass,
            atom_type_one_hot,
            chemutils.atom_total_degree_one_hot,
            chemutils.atom_total_num_H_one_hot,
            chemutils.atom_is_aromatic_one_hot,
            chemutils.atom_is_in_ring_one_hot])
    return chemutils.BaseAtomFeaturizer({"h": atom_featurizer_funs, "m": atom_mass_fun})

def get_bond_featurizer(feature_mode, self_loop):
    if feature_mode == 'light':
        return chemutils.BaseBondFeaturizer(featurizer_funcs={'e': chemutils.ConcatFeaturizer([chemutils.bond_type_one_hot])}, self_loop=self_loop)

def get_ms_setting_all_nodes(precursor_type, ce, n_nodes, prec_pool):
    out = th.zeros((n_nodes, len(prec_pool) + 1))
    out[:, prec_pool.index(precursor_type)] = 1.0
    out[:, -1] = ce
    return out

def get_ms_setting(precursor_type, ce, prec_pool):
    out = th.zeros(1, len(prec_pool) + 1)
    out[0, prec_pool.index(precursor_type)] = 1.0
    out[0, -1] = ce
    return out

def esp_preprocess(mol, prec_type, nce, prec_types):
    # TODO: support LDA features
    setting_tensor = get_ms_setting(prec_type, nce, prec_types)
    # fp = np.array([int(x) for x in AllChem.GetMorganFingerprintAsBitVect(mol, radius = 2, nBits = fp_size).ToBitString()])
    g = chemutils.mol_to_bigraph(
        mol,
        node_featurizer=get_atom_featurizer("medium", ELEMENT_LIST),
        edge_featurizer=get_bond_featurizer("light", True),
        add_self_loop=True,
        num_virtual_nodes=0
    )
    setting_tensor_on_nodes = get_ms_setting_all_nodes(prec_type, nce, g.num_nodes(), prec_types)
    g.ndata['h'] = th.cat((g.ndata['h'], setting_tensor_on_nodes), -1)
    pyg_g = convert_graph_dgl_to_torchgeometric(g)
    return pyg_g, setting_tensor

def convert_graph_dgl_to_torchgeometric(graph):
    g_nodes = graph.nodes().long()
    g_edges = th.stack([x for x in graph.edges()],dim=0).long()
    g_edges_f = th.argmax(graph.edata['e'],dim=1,keepdim=True).long()
    g_nodes_f = graph.ndata['h'][g_nodes].float()
    g = torch_geometric.data.Data(
        x=g_nodes_f, 
        edge_index=g_edges, 
        edge_attr=g_edges_f
    )
    return g

def collator(g_list):

    # import pdb; pdb.set_trace()
    return torch_geometric.data.Batch.from_data_list(g_list)