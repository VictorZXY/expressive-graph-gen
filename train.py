import argparse
import os
import pickle

from torch import nn, optim
from torchdrug import datasets, models, tasks, core


def load_dataset(dataset_dir):
    if os.path.exists(os.path.join(dataset_dir, 'zinc250k.pickle')):
        with open(os.path.join(dataset_dir, 'zinc250k.pickle'), 'rb') as fin:
            dataset = pickle.load(fin)
    else:
        dataset = datasets.ZINC250k(dataset_dir, kekulize=True, node_feature='symbol')
        with open(os.path.join(dataset_dir, 'zinc250k.pickle'), 'wb') as fout:
            pickle.dump(dataset, fout)

    return dataset


def pretrain_GCPN(dataset, num_epochs, checkpoint_dir):
    model = models.RGCN(input_dim=dataset.node_feature_dim, hidden_dims=[256, 256, 256, 256],
                        num_relation=dataset.num_bond_type, batch_norm=False)
    task = tasks.GCPNGeneration(model, dataset.atom_types, max_edge_unroll=12, max_node=38, criterion='nll')

    optimizer = optim.Adam(task.parameters(), lr=1e-3)
    solver = core.Engine(task, dataset, None, None, optimizer, gpus=(0,), batch_size=128, log_interval=10)

    solver.train(num_epoch=1)
    solver.save(os.path.join(checkpoint_dir, f'gcpn_zinc250k_{num_epochs}epochs.pickle'))

    return task, optimizer, solver


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--checkpoint_dir', type=str, required=True)
    args = parser.parse_args()

    dataset = load_dataset(args.data_dir)
    task, optimizer, solver = pretrain_GCPN(dataset, num_epochs=1, checkpoint_dir=args.checkpoint_dir)
    results = task.generate(num_sample=32, max_resample=5)
    print(results.to_smiles())
