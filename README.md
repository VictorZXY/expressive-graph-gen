# Improving Graph Generative Models via Expressive Graph Neural Networks
Mini-project for my CST Part III Representation Learning on Graphs and Networks (L45) module

Report [here](https://victorzxy.github.io/project/expressive-graph-gen/Expressive-Graph-Generation-Report.pdf)

### Abstract

Graph generation is a very challenging problem that requires predicting an entire graph with multiple nodes and edges from a given label, and is fundamental for many real-world tasks, such as molecular graph generation for drug discovery. A lot of successful methods have been explored on graph generation, including Graph Convolutional Policy Network (GCPN) and GraphAF, but the underlying graph neural network (GNN) structure for graph representation within both works remains untouched, which is Relational Graph Convolutional Network (R-GCN). In this mini-project, I investigate the expressivity of GNNs under the context of the graph generation problem, by replacing R-GCN in GCPN with more expressive GNNs, including Graph Isomorphism Network (GIN), Principal Neighbourhood Aggregation (PNA) and Graph Substructure Network (GSN). Experimental results show that more expressive GNNs can indeed significantly improve GCPN's performance on chemical property optimisation, with the only bottleneck coming from the sensitive nature of the graph generative method.
