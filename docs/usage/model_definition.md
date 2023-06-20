Here are some examples on how to define models using the framework's custom layers

## Standard graphs using the Sequential-API
If graphs only contain node features, the simplest way to define a model is through the [Sequential-API](https://www.tensorflow.org/guide/keras/sequential_model).

Here is a minimal example for the Sequential-API. Layers of the package `gog.layers` are not only stackable with each 
other, but can also be combined with other Keras layers such as BatchNormalization or Dropout.

Note however, that these Keras layers may not be necessary since each graph layer uses BatchNormalization and Dropout
when computing new node embeddings. 
```python
import gog
from tensorflow import keras

model = keras.models.Sequential([
        keras.layers.Input((n_nodes, n_features)),
        gog.layers.GraphBase(adjacency_matrix=adj, embedding_size=3),
        keras.layers.BatchNormalization(),
        keras.layers.ReLU(),
        gog.layers.GraphBase(adjacency_matrix=adj, embedding_size=3),
        keras.layers.BatchNormalization(),
        keras.layers.ReLU(),
        gog.layers.FlattenedDenseOutput(n_features)
    ])
```

## Graphs with node and edge features using the Functional-API
Similarly, the [Functional-API](https://www.tensorflow.org/guide/keras/functional_api) can be used to define models.
This is especially useful when **edge features** are present, since the Sequential model is limited to single input layers.
The following example shows how to handle this case by using a simple Python function.

```python
from gog.layers import GraphAttention
from tensorflow import keras

def create_attn_model_edge():
    input_layer_node = keras.layers.Input((n_nodes, n_features_node))
    input_layer_edge = keras.layers.Input((n_edges, n_features_edge))
    gnn = GraphAttention(adjacency_matrix=adj, embedding_size=5)(
        [input_layer_node, input_layer_edge])
    gnn = keras.layers.BatchNormalization()(gnn)
    gnn = keras.layers.ReLU()(gnn)
    gnn = GraphAttention(adjacency_matrix=adj, embedding_size=5)(
        [gnn, input_layer_edge])
    gnn = keras.layers.BatchNormalization()(gnn)
    gnn = keras.layers.ReLU()(gnn)
    gnn = GraphAttention(adjacency_matrix=adj, embedding_size=5)(
        [gnn, input_layer_edge])
    out = keras.layers.Dense(n_features_node)(gnn)
    return keras.models.Model(inputs=[input_layer_node, input_layer_edge], outputs=out)
```
As you can see, the edge features have to be passed to every graph layer, since edge features are used together with the
current node embeddings to compute a weighted adjacency matrix and no edge feature embeddings are created in the process.
This is the **only** step where edge features need to be handled differently by the user.
