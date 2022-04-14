import argparse
import os
import pickle

import torch
from torch import optim
from torchdrug import datasets, models, tasks, core
from torchdrug.layers import distribution


def load_dataset(dataset_dir):
    if os.path.exists(os.path.join(dataset_dir, 'zinc250k.pickle')):
        with open(os.path.join(dataset_dir, 'zinc250k.pickle'), 'rb') as fin:
            dataset = pickle.load(fin)
    else:
        dataset = datasets.ZINC250k(dataset_dir, kekulize=True, node_feature='symbol')
        with open(os.path.join(dataset_dir, 'zinc250k.pickle'), 'wb') as fout:
            pickle.dump(dataset, fout)

    return dataset


def train_GCPN(dataset, gnn_type, checkpoint_dir,
               num_pretrain_epochs=1,
               num_finetune_epochs=10):
    if gnn_type == 'RGCN':
        model = models.RGCN(input_dim=dataset.node_feature_dim,
                            hidden_dims=[256, 256, 256, 256],
                            num_relation=dataset.num_bond_type,
                            batch_norm=False)
    elif gnn_type == 'GIN':
        model = models.GIN(input_dim=dataset.node_feature_dim,
                           hidden_dims=[256, 256, 256, 256],
                           batch_norm=False)
    else:
        raise NotImplementedError

    # Pre-training
    pretrain_task = tasks.GCPNGeneration(model, dataset.atom_types,
                                         max_edge_unroll=12, max_node=38,
                                         criterion='nll')
    pretrain_optimizer = optim.Adam(pretrain_task.parameters(), lr=1e-3)
    pretrain_solver = core.Engine(pretrain_task, dataset, None, None,
                                  pretrain_optimizer, gpus=(0,),
                                  batch_size=128, log_interval=10)

    pretrain_solver.train(num_epoch=num_pretrain_epochs)
    if num_pretrain_epochs > 1:
        pretrain_model_name = f'gcpn_{gnn_type.lower()}_zinc250k_{num_pretrain_epochs}epochs.pickle'
    else:
        pretrain_model_name = f'gcpn_{gnn_type.lower()}_zinc250k_{num_pretrain_epochs}epoch.pickle'
    pretrain_solver.save(os.path.join(checkpoint_dir, pretrain_model_name))

    pretrain_results = pretrain_task.generate(num_sample=32, max_resample=5)
    print(pretrain_results.to_smiles())

    # Fine-tuning
    finetune_task = tasks.GCPNGeneration(model, dataset.atom_types,
                                         max_edge_unroll=12, max_node=38,
                                         task="plogp", criterion="ppo",
                                         reward_temperature=1,
                                         agent_update_interval=3,
                                         gamma=0.9)
    finetune_optimizer = optim.Adam(finetune_task.parameters(), lr=1e-5)
    finetune_solver = core.Engine(finetune_task, dataset, None, None,
                                  finetune_optimizer, gpus=(0,),
                                  batch_size=16, log_interval=10)
    finetune_solver.load(os.path.join(checkpoint_dir, pretrain_model_name),
                         load_optimizer=False)

    finetune_solver.train(num_epoch=num_finetune_epochs)
    if num_pretrain_epochs > 1:
        finetune_model_name = f'gcpn_{gnn_type.lower()}_zinc250k_{num_pretrain_epochs}epochs_finetune.pickle'
    else:
        finetune_model_name = f'gcpn_{gnn_type.lower()}_zinc250k_{num_pretrain_epochs}epoch_finetune.pickle'
    finetune_solver.save(os.path.join(checkpoint_dir, finetune_model_name))

    finetune_results = finetune_task.generate(num_sample=32, max_resample=5)
    print(finetune_results.to_smiles())


def train_GraphAF(dataset, gnn_type, checkpoint_dir,
                  num_pretrain_epochs=10,
                  num_finetune_epochs=10):
    if gnn_type == 'RGCN':
        model = models.RGCN(input_dim=dataset.node_feature_dim,
                            hidden_dims=[256, 256, 256],
                            num_relation=dataset.num_bond_type,
                            batch_norm=True)
    elif gnn_type == 'GIN':
        model = models.GIN(input_dim=dataset.node_feature_dim,
                           hidden_dims=[256, 256, 256],
                           batch_norm=True)
    else:
        raise NotImplementedError

    num_atom_type = dataset.num_atom_type
    num_bond_type = dataset.num_bond_type + 1  # add one class for non-edge

    node_prior = distribution.IndependentGaussian(torch.zeros(num_atom_type),
                                                  torch.ones(num_atom_type))
    edge_prior = distribution.IndependentGaussian(torch.zeros(num_bond_type),
                                                  torch.ones(num_bond_type))
    node_flow = models.GraphAF(model, node_prior, num_layer=12)
    edge_flow = models.GraphAF(model, edge_prior, use_edge=True, num_layer=12)

    # Pre-training
    pretrain_task = tasks.AutoregressiveGeneration(node_flow, edge_flow,
                                                   max_edge_unroll=12, max_node=38,
                                                   criterion="nll")
    pretrain_optimizer = optim.Adam(pretrain_task.parameters(), lr=1e-3)
    pretrain_solver = core.Engine(pretrain_task, dataset, None, None,
                                  pretrain_optimizer, gpus=(0,),
                                  batch_size=128, log_interval=10)

    pretrain_solver.train(num_epoch=num_pretrain_epochs)
    if num_pretrain_epochs > 1:
        pretrain_model_name = f'graphaf_{gnn_type.lower()}_zinc250k_{num_pretrain_epochs}epochs.pickle'
    else:
        pretrain_model_name = f'graphaf_{gnn_type.lower()}_zinc250k_{num_pretrain_epochs}epoch.pickle'
    pretrain_solver.save(os.path.join(checkpoint_dir, pretrain_model_name))

    pretrain_results = pretrain_task.generate(num_sample=32)
    print(pretrain_results.to_smiles())

    # Fine-tuning
    finetune_task = tasks.AutoregressiveGeneration(node_flow, edge_flow,
                                                   max_edge_unroll=12, max_node=38,
                                                   task="plogp", criterion="ppo",
                                                   reward_temperature=20,
                                                   baseline_momentum=0.9,
                                                   agent_update_interval=5,
                                                   gamma=0.9)
    finetune_optimizer = optim.Adam(finetune_task.parameters(), lr=1e-5)
    finetune_solver = core.Engine(finetune_task, dataset, None, None,
                                  finetune_optimizer, gpus=(0,),
                                  batch_size=64, log_interval=10)
    finetune_solver.load(os.path.join(checkpoint_dir, pretrain_model_name),
                         load_optimizer=False)

    finetune_solver.train(num_epoch=num_finetune_epochs)
    if num_pretrain_epochs > 1:
        finetune_model_name = f'graphaf_{gnn_type.lower()}_zinc250k_{num_pretrain_epochs}epochs_finetune.pickle'
    else:
        finetune_model_name = f'graphaf_{gnn_type.lower()}_zinc250k_{num_pretrain_epochs}epoch_finetune.pickle'
    finetune_solver.save(os.path.join(checkpoint_dir, finetune_model_name))

    finetune_results = finetune_task.generate(num_sample=32)
    print(finetune_results.to_smiles())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str, required=True)
    parser.add_argument('--gnn_type', type=str, required=True)
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--checkpoint_dir', type=str, required=True)
    args = parser.parse_args()

    dataset = load_dataset(args.data_dir)
    if args.model_type == 'GCPN':
        train_GCPN(dataset=dataset, gnn_type=args.gnn_type,
                   checkpoint_dir=args.checkpoint_dir)
    elif args.model_type == 'GraphAF':
        train_GraphAF(dataset=dataset, gnn_type=args.gnn_type,
                      checkpoint_dir=args.checkpoint_dir)
