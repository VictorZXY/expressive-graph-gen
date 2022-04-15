import os
import pickle

from torchdrug import datasets, models


def load_dataset(dataset_dir):
    if os.path.exists(os.path.join(dataset_dir, 'zinc250k.pickle')):
        with open(os.path.join(dataset_dir, 'zinc250k.pickle'), 'rb') as fin:
            dataset = pickle.load(fin)
    else:
        dataset = datasets.ZINC250k(dataset_dir, kekulize=True, node_feature='symbol')
        with open(os.path.join(dataset_dir, 'zinc250k.pickle'), 'wb') as fout:
            pickle.dump(dataset, fout)

    return dataset


def load_GNN(dataset, model_type, gnn_type):
    if model_type == 'GCPN':
        if gnn_type == 'RGCN':
            gnn = models.RGCN(input_dim=dataset.node_feature_dim,
                              hidden_dims=[256, 256, 256, 256],
                              num_relation=dataset.num_bond_type,
                              batch_norm=False)
        elif gnn_type == 'GIN':
            gnn = models.GIN(input_dim=dataset.node_feature_dim,
                             hidden_dims=[256, 256, 256, 256],
                             batch_norm=False)
        else:
            raise NotImplementedError
    elif model_type == 'GraphAF':
        if gnn_type == 'RGCN':
            gnn = models.RGCN(input_dim=dataset.node_feature_dim,
                              hidden_dims=[256, 256, 256],
                              num_relation=dataset.num_bond_type,
                              batch_norm=True)
        elif gnn_type == 'GIN':
            gnn = models.GIN(input_dim=dataset.node_feature_dim,
                             hidden_dims=[256, 256, 256],
                             batch_norm=True)
        else:
            raise NotImplementedError
    else:
        raise NotImplemented

    return gnn
