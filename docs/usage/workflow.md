This page describes a possible workflow powered by the framework. Make sure to set up the installation before.

## Initializing the dataset

The first step is to convert a tabular dataset from a pandas `DataFrame` object to a `StaticGraphDataset`.
The data format is described in the [documentation](../../structure/#gog.structure.graph.StaticGraphDataset) of the dataset
class.

The code to create an instance of `StaticGraphDataset` may look like this:

```python
import pandas as pd
from gog.structure.graph import StaticGraphDataset

# initialize edge list with some values 
edge_list = ...

# read dataframe data
df = pd.read_csv("some_data.csv")
# use all columns as features
node_feature_names = df.columns.values.tolist()
n_nodes = df["Node"].nunique()

# list of graph instances
graphs = StaticGraphDataset.pandas_to_graphs(df_node_features=df, node_feature_names=node_feature_names, num_nodes=n_nodes)
dataset = StaticGraphDataset(edge_list=edge_list, graphs=graphs)
```

This dataset only contains graphs without edge features. If edge features are present, they are also passed to the 
`pandas_to_graphs` function just like the node features are.

## Preprocessing
With the created `StaticGraphDataset` instance, the preprocessing can be started. Depending on the task and whether
graphs are independent instances from each other or graphs represent a time-series, there are two slighly different
workflows that will be shown.

### Independent graphs

#### Splitting the data
A mandatory step in every ML task, is to split the dataset into a training and test set.
If each graph is to be treated as an individual instance, the method 
[create_train_test_split()](../../preprocessing/#gog.preprocessing.preprocessing.create_train_test_split) can be used as seen
here: 

```python
from gog.preprocessing.preprocessing import create_train_test_split
train, test = create_train_test_split(dataset)
```
#### Scaling the data
An optional step is to scale the data. This is easily done by calling
```python
from gog.preprocessing.preprocessing import apply_scaler
train, test = apply_scaler(dataset)

# if edge features are present you will also need to call the function with the target parameter set to "edge"
train, test = apply_scaler(dataset, target="edge")
```

#### Masking the input data
The next step is to mask the input data. Here we can select which nodes and which features to mask

```python
import numpy as np
from gog.preprocessing.preprocessing import mask_features

features = ["A", "B"]
nodes_to_mask = np.arange(0, 3)
masked_train, masked_test = mask_features(train, test, features, nodes_to_mask)
```

#### Creating a validation set
If you want to have a validation set, to monitor the generalization performance of the model, you can use the in-built
function [create_validation_set()](../../preprocessing/#gog.preprocessing.preprocessing.create_validation_set).

```python
from gog.preprocessing.preprocessing import create_validation_set
X_train, X_val, y_train, y_val = create_validation_set(masked_train, train)
```


#### Converting the data to numpy
In order to be able to train a model with our data, it needs to a format that TensorFlow can process. Luckily, calling
[to_numpy()](../../structure/#gog.structure.graph.GraphList.to_numpy) on any data split, will generate exactly that.

```python
X_train = X_train.to_numpy()
y_train = y_train.to_numpy()

# if you created a validation set, convert it as well
X_val = X_val.to_numpy()
y_val = y_val.to_numpy()
```

### Time-series data
#### Splitting the data
If the data contains time-series information, often times a rolling window has to be created from the data. This task is
combined with splitting the dataset with the function
[create_train_test_split_windowed()](../../preprocessing/#gog.preprocessing.preprocessing.create_train_test_split_windowed)
and can be used like:

```python
from gog.preprocessing.preprocessing import create_train_test_split_windowed
X_train, y_train, X_test, y_test = create_train_test_split_windowed(dataset, window_size=30, len_labels=3)
```

#### Scaling the data
An optional step is to scale the data. This is easily done by calling 
```python
from gog.preprocessing.preprocessing import apply_scaler
train, test = apply_scaler(dataset)

# if edge features are present you will also need to call the function with the target parameter set to "edge"
X_train, y_train, X_test, y_test = apply_scaler(dataset, target="edge")
```

#### Masking the input data
The next step is to mask the input data. Here we can select which nodes and which features to mask

```python
import numpy as np
from gog.preprocessing.preprocessing import mask_features

features = ["A", "B"]
nodes_to_mask = np.arange(0, 3)
X_train, X_test = mask_features(X_train, X_test, features, nodes_to_mask)
```

#### Creating a validation set
If you want to have a validation set, to monitor the generalization performance of the model, you can use the in-built
function [create_validation_set()](../../preprocessing/#gog.preprocessing.preprocessing.create_validation_set).

```python
from gog.preprocessing.preprocessing import create_validation_set
X_train, X_val, y_train, y_val = create_validation_set(X_train, y_train)
```


#### Converting the data to numpy
In order to be able to train a model with our data, it needs to a format that TensorFlow can process. Luckily, calling
[to_numpy()](../../structure/#gog.structure.graph.GraphList.to_numpy) on any data split, will generate exactly that.

```python
X_train = X_train.to_numpy()
y_train = y_train.to_numpy()

# if you created a validation set, convert it as well
X_val = X_val.to_numpy()
y_val = y_val.to_numpy()
```

## Creating and training the model
If you are unsure on how to create a model using the provided graph layers, check the [page](../../usage/model_definition)
where multiple examples are shown. Beware the limitations when using edge features.

Training the model is very straightforward since there are no differences to training any other model in TF/Keras. 
The right data format is ensured by calling `to_numpy()` on the dataset earlier, other differences will be handled
by the layers automatically.

```python
from tensorflow import keras
EPOCHS = 10
LR = 0.001
BATCH_SIZE = 32
optimizer = keras.optimizers.Adam(learning_rate=LR)
loss_fn = keras.losses.MeanSquaredError()

# get keras.Model instance as seen on the model definition docs
model = get_model()
model.compile(optimizer=optimizer, loss=loss_fn)

# optionally print the model summary
print(model.summary())

history = model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_data=(X_val, y_val))
```

## Testing the model
To test the model, we need to convert the test split to a numpy array first. If we have scaled our data,
and want to manually check which values our model is predicting, we can access the scalers from the dataset
instance.

```python
from gog.metrics.metrics import mean_squared_error
X_test = X_test.to_numpy()
y_test = y_test.to_numpy()

y_pred = model.predict(X_test)
# print error scores
print(mean_squared_error(y_true=y_test, y_pred=y_pred))

# check predicted values without scaling
node_scaler = dataset.node_scaler
real_values = node_scaler.inverse_transform(y_pred[0])
print(real_values)
```

