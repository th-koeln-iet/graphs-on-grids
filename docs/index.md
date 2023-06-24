# Welcome to Graphs on Grids

**Graphs on Grids** is a high-level framework for working with graph data based on Keras and TensorFlow 2.
It mainly provides specialized layers for graph data but also enables an easy workflow using its preprocessing pipeline.
Note that this pipeline is more specialized for working with power grid data, however it can be adapted
to work with any dataset.

For more information on the usage, check the provided [workflow guide](/usage/workflow).
Additionally, you can find documentation on the different layer types under the _layers_ section.

The framework mainly focuses on node-level tasks such as node regression or classification. For this, the framework 
provides tools to model standard as well as graph time-series problems.

## How graphs are organized in the framework
Within the framework, graphs are organized in a `StaticGraphDataset` class. Each graph in the dataset needs to have
the same structure as in number of nodes, edges and equal edge connectivity (isomorphic graphs).

Each graph is mapped to a `gog.structure.Graph` object and contains its `node_features` and `edge_features`. Additionally,
they contain information about the number of nodes and edges in a graph. Graph objects however
do not contain the adjacency matrix, which is instead saved in the `StaticGraphDataset` class to avoid duplicate information
for each graph in the dataset.

Graphs within the `StaticGraphDataset` class are organized as `GraphList` objects. They behave similarly to lists, only 
that they provide validation for graphs that are appended to it, as well as information about the graphs they contain.
Furthermore, `GraphList` objects can be converted to **numpy** `nd.arrays` or **pandas** `DataFrame` objects.

## Installation
To install Graphs on Grids, run the following commands in a terminal:
```commandline
git clone git@github.com:th-koeln-iet/graphs-on-grids.git
cd graphs-on-grids
pip install -r requirements.txt
```