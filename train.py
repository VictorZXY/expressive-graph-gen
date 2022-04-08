import os
import pickle

from torch import nn, optim
from torchdrug import datasets, models, tasks, core


def load_dataset():
    if os.path.exists('datasets/zinc250k.pickle'):
        with open('datasets/zinc250k.pickle', 'rb') as fin:
            dataset = pickle.load(fin)
    else:
        dataset = datasets.ZINC250k('~/molecule-datasets/', kekulize=True, node_feature='symbol')
        with open('datasets/zinc250k.pickle', 'wb') as fout:
            pickle.dump(dataset, fout)

    return dataset


def pretrain_GCPN(dataset):
    model = models.RGCN(input_dim=dataset.node_feature_dim, hidden_dims=[256, 256, 256, 256],
                        num_relation=dataset.num_bond_type, batch_norm=False)
    task = tasks.GCPNGeneration(model, dataset.atom_types, max_edge_unroll=12, max_node=38, criterion='nll')

    optimizer = optim.Adam(task.parameters(), lr=1e-3)
    solver = core.Engine(task, dataset, None, None, optimizer, gpus=(0,), batch_size=128, log_interval=10)

    solver.train(num_epoch=1)
    solver.save('checkpoints/gcpn_zinc250k_1epoch.pickle')

    return task, optimizer, solver


if __name__ == '__main__':
    dataset = load_dataset()
    task, optimizer, solver = pretrain_GCPN(dataset)
    results = task.generate(num_sample=32, max_resample=5)
    print(results.to_smiles())
