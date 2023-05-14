import pandas as pd

from gog.preprocessing import create_train_test_split, mask_labels
from gog.structure.graph import StaticGraphDataset, Graph


class TestPreprocessing:
    @classmethod
    def setup_method(cls):
        feature_names = ["A", "B", "Node"]
        data = [[1, 2, 0], [3, 4, 1], [5, 6, 2], [7, 8, 3], [9, 10, 4]]
        df = pd.DataFrame(columns=feature_names, data=data)
        graphs = []
        for i in range(5):
            g = Graph(df, feature_names[:2])
            graphs.append(g)
        dataset = StaticGraphDataset(edge_list=[[0, 2], [0, 3], [3, 1], [2, 4], [2, 3]], graphs=graphs)
        cls.dataset = dataset

    def test_create_train_test_split(self):
        train, test = create_train_test_split(self.dataset)
        assert len(train) == 4
        assert len(test) == 1

        assert None not in train
        assert None not in test

    def test_mask_labels(self):
        train, test = create_train_test_split(self.dataset)
        targets = ["A"]
        nodes_to_mask = [1, 2]
        train_masked, test_masked = mask_labels(train, test, targets, nodes_to_mask)

        # assert correct length
        assert len(train_masked) == 4
        assert len(test_masked) == 1

        # masked features in selected nodes are actually masked
        for graph in train_masked:
            masked_nodes = graph.node_features[nodes_to_mask]
            for node in masked_nodes:
                assert node[0] == 0

        # original features matrix is unaffected
        for graph in train:
            masked_nodes = graph.node_features[nodes_to_mask]
            for node in masked_nodes:
                assert node[0] != 0
