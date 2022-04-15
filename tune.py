import argparse

import torch
from torch import optim
from torchdrug import models, tasks, core
from torchdrug.layers import distribution

from utils import load_dataset, load_GNN


def finetune_GCPN(dataset, gnn_type, pretrained_model_path, num_epoch=10):
    model = load_GNN(dataset=dataset, model_type='GCPN', gnn_type=gnn_type)

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
    finetune_solver.load(pretrained_model_path, load_optimizer=False)

    finetune_solver.train(num_epoch=num_epoch)
    finetune_model_path = pretrained_model_path.replace('.pickle', '_finetune.pickle')
    finetune_solver.save(finetune_model_path)

    finetune_results = finetune_task.generate(num_sample=32, max_resample=5)
    print(finetune_results.to_smiles())


def finetune_GraphAF(dataset, gnn_type, pretrained_model_path, num_epoch=10):
    model = load_GNN(dataset=dataset, model_type='GraphAF', gnn_type=gnn_type)

    num_atom_type = dataset.num_atom_type
    num_bond_type = dataset.num_bond_type + 1  # add one class for non-edge

    node_prior = distribution.IndependentGaussian(torch.zeros(num_atom_type),
                                                  torch.ones(num_atom_type))
    edge_prior = distribution.IndependentGaussian(torch.zeros(num_bond_type),
                                                  torch.ones(num_bond_type))
    node_flow = models.GraphAF(model, node_prior, num_layer=12)
    edge_flow = models.GraphAF(model, edge_prior, use_edge=True, num_layer=12)

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
    finetune_solver.load(pretrained_model_path, load_optimizer=False)

    finetune_solver.train(num_epoch=num_epoch)
    finetune_model_path = pretrained_model_path.replace('.pickle', '_finetune.pickle')
    finetune_solver.save(finetune_model_path)

    finetune_results = finetune_task.generate(num_sample=32)
    print(finetune_results.to_smiles())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str, required=True)
    parser.add_argument('--gnn_type', type=str, required=True)
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--pretrained_model_path', type=str, required=True)
    args = parser.parse_args()

    dataset = load_dataset(args.data_dir)
    if args.model_type == 'GCPN':
        finetune_GCPN(dataset=dataset, gnn_type=args.gnn_type,
                      pretrained_model_path=args.pretrained_model_path)
    elif args.model_type == 'GraphAF':
        finetune_GraphAF(dataset=dataset, gnn_type=args.gnn_type,
                         pretrained_model_path=args.pretrained_model_path)
