import os

import numpy as np
import pandas as pd

from gog.structure.graph import StaticGraphDataset


def data_to_parquet(filecount):
    dataframes = []
    output_path = f"{filecount}_graphs.parquet.gzip"
    for i in range(1, filecount + 1):
        directory = f"data/{i}"
        if not os.path.exists(directory):
            print(f"Directory {directory} not found.")
            continue
        for filename in os.listdir(directory):
            if filename.endswith(".csv") and not filename.startswith("S"):
                f = os.path.join(directory, filename)
                df = pd.read_csv(f)
                split_name = filename.split("CIGRE_Low_Voltage_Mon_monitoratbus")[1]
                node_num = split_name.split("_")[0]
                df["Node"] = int(node_num)
                df["Timestep"] = i
                dataframes.append(df)
        if i % 500 == 0:
            print(f"Handled {i} files")
            if os.path.isfile(output_path):
                pd.concat(dataframes).to_parquet(output_path, engine="fastparquet", index=False, compression="gzip",
                                                 append=True)
            else:
                pd.concat(dataframes).to_parquet(output_path, engine="fastparquet", index=False, compression="gzip")
            dataframes = []


def get_edges():
    adm_data = pd.read_csv("AdmittanceMatrix_50Hz.csv",
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


def create_train_val_test_split(dataset: StaticGraphDataset, train_size=0.65, validation_size=0.15, random_state=None,
                                shuffle=True):
    graph_list = dataset.graphs
    num_graphs = len(graph_list)
    if shuffle:
        np.random.seed(random_state)
        np.random.shuffle(graph_list)
    last_train_index = int(num_graphs * train_size)
    last_val_index = int(last_train_index + num_graphs * validation_size)
    train, val, test = graph_list[0:last_train_index], \
        graph_list[last_train_index + 1: last_val_index], \
        graph_list[last_train_index:num_graphs + 1]
    dataset.set_splits(train=train, val=val, test=test)
    return train, val, test
