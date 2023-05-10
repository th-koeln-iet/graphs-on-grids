import os
from typing import List

import numpy as np
import pandas as pd

from gog.structure.graph import StaticGraphs, Graph


def get_dataframe(filecount):
    dataframes = []
    for i in range(1, filecount):
        directory = f"../data/{i}"
        for filename in os.listdir(directory):
            if filename.endswith(".csv") and not filename.startswith("Simulation"):
                f = os.path.join(directory, filename)
                df = pd.read_csv(f)
                split_name = filename.split("CIGRE_Low_Voltage_Mon_monitoratbus")[1]
                node_num = split_name.split("_")[0]
                df["Node"] = node_num
                df["Timestep"] = i
                dataframes.append(df)
    return pd.concat(dataframes, ignore_index=True)


def get_edges():
    adm_data = pd.read_csv("../AdmittanceMatrix_50Hz.csv",
                           delimiter=";",
                           header=None).iloc[1:, 1:-1].astype(complex).reset_index(drop=True)
    adm_data.columns = range(adm_data.columns.size)

    # Umwandlung der Admittanz-Matrix in eine Bool'sche Adjazenzmatrix
    adj_mask = adm_data != 0

    edges = []
    for i in range(len(adj_mask)):
        for j in range(len(adj_mask)):
            if adj_mask.iloc[i, j]:
                # Adjust to file naming convention
                edges.append((i + 1, j + 1))
    return edges


def create_train_test_split(graphs: StaticGraphs, train_size=0.8, random_state=None, shuffle=True):
    graph_list = graphs.graphs
    num_graphs = len(graph_list)
    if shuffle:
        np.random.seed(random_state)
        np.random.shuffle(graph_list)
    last_train_index = int(num_graphs * train_size)
    train, test = graph_list[0:last_train_index], graph_list[last_train_index:num_graphs + 1]
    return train, test


def mask_labels(X_train: List[Graph], X_test: List[Graph], targets: List[str], nodes: List, method: str = "zeros"):
    X_train_mask, X_test_mask = X_train.copy(), X_test.copy()
    return _mask_split(X_train_mask, targets, nodes, method), _mask_split(X_test_mask, targets, nodes, method)


def _mask_split(split: List[Graph], targets: List[str], nodes: List, method: str):
    for instance in split:
        feature_matrix = instance.node_features
        feature_indices = [instance.node_feature_names.index(feature_name) for feature_name in targets]
        for row in feature_matrix:
            if row in nodes:
                if method is "zeros":
                    row[feature_indices] = 0
