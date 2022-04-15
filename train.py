import argparse
import os

import torch
from torch import optim
from torchdrug import models, tasks, core
from torchdrug.layers import distribution

from utils import load_dataset, load_GNN


def pretrain_GCPN(dataset, gnn_type, checkpoint_dir, num_epoch=1):
    model = load_GNN(dataset=dataset, model_type='GCPN', gnn_type=gnn_type)

    pretrain_task = tasks.GCPNGeneration(model, dataset.atom_types,
                                         max_edge_unroll=12, max_node=38,
                                         criterion='nll')
    pretrain_optimizer = optim.Adam(pretrain_task.parameters(), lr=1e-3)
    pretrain_solver = core.Engine(pretrain_task, dataset, None, None,
                                  pretrain_optimizer, gpus=(0,),
                                  batch_size=128, log_interval=10)

    pretrain_solver.train(num_epoch=num_epoch)
    if num_epoch > 1:
        pretrain_model_name = f'gcpn_{gnn_type.lower()}_zinc250k_{num_epoch}epochs.pickle'
    else:
        pretrain_model_name = f'gcpn_{gnn_type.lower()}_zinc250k_{num_epoch}epoch.pickle'
    pretrain_solver.save(os.path.join(checkpoint_dir, pretrain_model_name))

    pretrain_results = pretrain_task.generate(num_sample=32, max_resample=5)
    print(pretrain_results.to_smiles())


def pretrain_GraphAF(dataset, gnn_type, checkpoint_dir, num_epoch=10):
    model = load_GNN(dataset=dataset, model_type='GraphAF', gnn_type=gnn_type)

    num_atom_type = dataset.num_atom_type
    num_bond_type = dataset.num_bond_type + 1  # add one class for non-edge

    node_prior = distribution.IndependentGaussian(torch.zeros(num_atom_type),
                                                  torch.ones(num_atom_type))
    edge_prior = distribution.IndependentGaussian(torch.zeros(num_bond_type),
                                                  torch.ones(num_bond_type))
    node_flow = models.GraphAF(model, node_prior, num_layer=12)
    edge_flow = models.GraphAF(model, edge_prior, use_edge=True, num_layer=12)

    pretrain_task = tasks.AutoregressiveGeneration(node_flow, edge_flow,
                                                   max_edge_unroll=12, max_node=38,
                                                   criterion="nll")
    pretrain_optimizer = optim.Adam(pretrain_task.parameters(), lr=1e-3)
    pretrain_solver = core.Engine(pretrain_task, dataset, None, None,
                                  pretrain_optimizer, gpus=(0,),
                                  batch_size=128, log_interval=10)

    pretrain_solver.train(num_epoch=num_epoch)
    if num_epoch > 1:
        pretrain_model_name = f'graphaf_{gnn_type.lower()}_zinc250k_{num_epoch}epochs.pickle'
    else:
        pretrain_model_name = f'graphaf_{gnn_type.lower()}_zinc250k_{num_epoch}epoch.pickle'
    pretrain_solver.save(os.path.join(checkpoint_dir, pretrain_model_name))

    pretrain_results = pretrain_task.generate(num_sample=32)
    print(pretrain_results.to_smiles())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str, required=True)
    parser.add_argument('--gnn_type', type=str, required=True)
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--checkpoint_dir', type=str, required=True)
    args = parser.parse_args()

    dataset = load_dataset(args.data_dir)
    if args.model_type == 'GCPN':
        pretrain_GCPN(dataset=dataset, gnn_type=args.gnn_type,
                      checkpoint_dir=args.checkpoint_dir)
    elif args.model_type == 'GraphAF':
        pretrain_GraphAF(dataset=dataset, gnn_type=args.gnn_type,
                         checkpoint_dir=args.checkpoint_dir)
