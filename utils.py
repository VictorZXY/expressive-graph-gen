import os
import pickle

import torch
from torch_geometric.utils import degree
from torchdrug import datasets, models

from models.gin import GIN_GCPN
from models.gsn import prepare_GSN_dataset, GSN
from models.pna import PNA


def load_dataset(dataset_dir):
    if os.path.exists(os.path.join(dataset_dir, 'zinc250k.pickle')):
        with open(os.path.join(dataset_dir, 'zinc250k.pickle'), 'rb') as fin:
            dataset = pickle.load(fin)
    else:
        dataset = datasets.ZINC250k(dataset_dir, kekulize=True, node_feature='symbol')
        with open(os.path.join(dataset_dir, 'zinc250k.pickle'), 'wb') as fout:
            pickle.dump(dataset, fout)

    return dataset


def load_GSN_dataset(dataset_dir):
    if os.path.exists(os.path.join(dataset_dir, 'zinc250k_gsn.pickle')):
        with open(os.path.join(dataset_dir, 'zinc250k_gsn.pickle'), 'rb') as fin:
            dataset = pickle.load(fin)
    else:
        dataset = load_dataset(dataset_dir)
        prepare_GSN_dataset(dataset)
        with open(os.path.join(dataset_dir, 'zinc250k_gsn.pickle'), 'wb') as fout:
            pickle.dump(dataset, fout)

    return dataset


def load_GNN(dataset, model_type, gnn_type):
    if gnn_type == 'RGCN':
        gnn = models.RGCN(input_dim=dataset.node_feature_dim,
                          hidden_dims=[128, 128, 128],
                          num_relation=dataset.num_bond_type,
                          batch_norm=True)
    elif gnn_type == 'GIN':
        if model_type == 'GCPN':
            gnn = GIN_GCPN(input_dim=dataset.node_feature_dim,
                           hidden_dims=[128, 128, 128],
                           num_relation=dataset.num_bond_type,
                           batch_norm=True)
        elif model_type == 'GraphAF':
            gnn = models.GIN(input_dim=dataset.node_feature_dim,
                             hidden_dims=[128, 128, 128],
                             batch_norm=True)
        else:
            assert False
    elif gnn_type == 'PNA':
        deg = torch.zeros(10, dtype=torch.long, device='cuda')
        for data in dataset:
            graph = data['graph']
            d = degree(graph.edge_list[:, 1], num_nodes=graph.num_node, dtype=torch.long)
            deg += torch.bincount(d, minlength=deg.numel())

        gnn = PNA(input_dim=dataset.node_feature_dim,
                  hidden_dim=128, num_layer=3,
                  edge_input_dim=dataset.edge_feature_dim,
                  num_relation=dataset.num_bond_type,
                  aggregators=['mean', 'min', 'max', 'std'],
                  scalers=['identity', 'amplification', 'attenuation'],
                  deg=deg, batch_norm=True)
    elif gnn_type == 'GSN':
        gnn = GSN(input_dim=dataset.node_feature_dim,
                  hidden_dim=128, num_layer=3,
                  edge_input_dim=dataset.edge_feature_dim,
                  num_relation=dataset.num_bond_type,
                  batch_norm=True)
    else:
        raise NotImplementedError

    return gnn
