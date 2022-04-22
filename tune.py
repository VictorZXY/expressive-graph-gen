import argparse

import torch
from torch import optim
from torchdrug import models, tasks, core
from torchdrug.layers import distribution

from utils import load_dataset, load_GNN


def finetune_GCPN(dataset, gnn_type, task, pretrained_model_path, num_epoch=10):
    model = load_GNN(dataset=dataset, model_type='GCPN', gnn_type=gnn_type)

    if task == 'plogp':
        finetune_task = tasks.GCPNGeneration(model, dataset.atom_types,
                                             max_edge_unroll=12, max_node=38,
                                             task="plogp", criterion="ppo",
                                             reward_temperature=1,
                                             agent_update_interval=3,
                                             gamma=0.9)
    elif task == 'qed':
        finetune_task = tasks.GCPNGeneration(model, dataset.atom_types,
                                             max_edge_unroll=12, max_node=38,
                                             task="qed", criterion=("ppo", "nll"),
                                             reward_temperature=1,
                                             agent_update_interval=3,
                                             gamma=0.9)
    else:
        raise ValueError(f'Unknown task {task}')

    finetune_optimizer = optim.Adam(finetune_task.parameters(), lr=1e-5)
    finetune_solver = core.Engine(finetune_task, dataset, None, None,
                                  finetune_optimizer, gpus=(0,),
                                  batch_size=64, log_interval=200)
    finetune_solver.load(pretrained_model_path, load_optimizer=False)

    finetune_solver.train(num_epoch=num_epoch)
    finetune_model_path = pretrained_model_path.replace('.pickle', f'_finetune_{task}.pickle')
    finetune_solver.save(finetune_model_path)

    finetune_results = finetune_task.generate(num_sample=32)
    print(finetune_results.to_smiles())
    if task == 'plogp':
        print(finetune_task.best_results['Penalized logP'])
    elif task == 'qed':
        print(finetune_task.best_results['QED'])
    else:
        raise ValueError(f'Unknown task {task}')


def finetune_GraphAF(dataset, gnn_type, task, pretrained_model_path, num_epoch=5):
    model = load_GNN(dataset=dataset, model_type='GraphAF', gnn_type=gnn_type)

    num_atom_type = dataset.num_atom_type
    num_bond_type = dataset.num_bond_type + 1  # add one class for non-edge

    node_prior = distribution.IndependentGaussian(torch.zeros(num_atom_type),
                                                  torch.ones(num_atom_type))
    edge_prior = distribution.IndependentGaussian(torch.zeros(num_bond_type),
                                                  torch.ones(num_bond_type))
    node_flow = models.GraphAF(model, node_prior, num_layer=12)
    edge_flow = models.GraphAF(model, edge_prior, use_edge=True, num_layer=12)

    if task == 'plogp':
        finetune_task = tasks.AutoregressiveGeneration(node_flow, edge_flow,
                                                       max_edge_unroll=12, max_node=38,
                                                       task="plogp", criterion="ppo",
                                                       reward_temperature=20,
                                                       baseline_momentum=0.9,
                                                       agent_update_interval=5,
                                                       gamma=0.9)
    elif task == 'qed':
        finetune_task = tasks.AutoregressiveGeneration(node_flow, edge_flow,
                                                       max_edge_unroll=12, max_node=38,
                                                       task="qed", criterion={"ppo": 0.25, "nll": 1.0},
                                                       reward_temperature=10,
                                                       baseline_momentum=0.9,
                                                       agent_update_interval=5,
                                                       gamma=0.9)
    else:
        raise ValueError(f'Unknown task {task}')

    finetune_optimizer = optim.Adam(finetune_task.parameters(), lr=1e-5)
    finetune_solver = core.Engine(finetune_task, dataset, None, None,
                                  finetune_optimizer, gpus=(0,),
                                  batch_size=64, log_interval=200)
    finetune_solver.load(pretrained_model_path, load_optimizer=False)

    finetune_solver.train(num_epoch=num_epoch)
    finetune_model_path = pretrained_model_path.replace('.pickle', f'_finetune_{task}.pickle')
    finetune_solver.save(finetune_model_path)

    finetune_results = finetune_task.generate(num_sample=32)
    print(finetune_results.to_smiles())
    if task == 'plogp':
        print(finetune_task.best_results['Penalized logP'])
    elif task == 'qed':
        print(finetune_task.best_results['QED'])
    else:
        raise ValueError(f'Unknown task {task}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str, required=True)
    parser.add_argument('--gnn_type', type=str, required=True)
    parser.add_argument('--task', type=str, required=True)
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--pretrained_model_path', type=str, required=True)
    parser.add_argument('--num_epoch', type=int, default=-1)
    args = parser.parse_args()

    dataset = load_dataset(args.data_dir)
    if args.num_epoch == -1:
        if args.model_type == 'GCPN':
            finetune_GCPN(dataset=dataset, gnn_type=args.gnn_type, task=args.task,
                          pretrained_model_path=args.pretrained_model_path)
        elif args.model_type == 'GraphAF':
            finetune_GraphAF(dataset=dataset, gnn_type=args.gnn_type, task=args.task,
                             pretrained_model_path=args.pretrained_model_path)
    else:
        if args.model_type == 'GCPN':
            finetune_GCPN(dataset=dataset, gnn_type=args.gnn_type, task=args.task,
                          pretrained_model_path=args.pretrained_model_path,
                          num_epoch=args.num_epoch)
        elif args.model_type == 'GraphAF':
            finetune_GraphAF(dataset=dataset, gnn_type=args.gnn_type, task=args.task,
                             pretrained_model_path=args.pretrained_model_path,
                             num_epoch=args.num_epoch)
