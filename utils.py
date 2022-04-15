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
    if gnn_type == 'RGCN':
        gnn = models.RGCN(input_dim=dataset.node_feature_dim,
                          hidden_dims=[256, 256, 256],
                          num_relation=dataset.num_bond_type,
                          batch_norm=True)
    elif gnn_type == 'GIN':
        if model_type == 'GraphAF':
            gnn = models.GIN(input_dim=dataset.node_feature_dim,
                             hidden_dims=[256, 256, 256],
                             batch_norm=True)
        else:
            assert False
    else:
        raise NotImplementedError

    return gnn
