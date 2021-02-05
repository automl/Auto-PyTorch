# Regularization Cocktail


**Source code for ICML submission 2867.**

**Our implementation of regularization cocktails is based on a fork of AutoPyTorch and this fork is part of the submission to ensure full reproducibility.**

Install pytorch: 
https://pytorch.org/

Install Auto-PyTorch:

```sh
$ cat requirements.txt | xargs -n 1 -L 1 pip install
$ python setup.py install
```


## Examples


In a nutshell:

```py
from autoPyTorch import AutoNetClassification

# data and metric imports
import sklearn.model_selection
import sklearn.datasets
import sklearn.metrics
X, y = sklearn.datasets.load_digits(return_X_y=True)
X_train, X_test, y_train, y_test = \
        sklearn.model_selection.train_test_split(X, y, random_state=1)

# running Auto-PyTorch
autoPyTorch = AutoNetClassification("tiny_cs",  # config preset
                                    log_level='info',
                                    max_runtime=300,
                                    min_budget=30,
                                    max_budget=90)

autoPyTorch.fit(X_train, y_train, validation_split=0.3)
y_pred = autoPyTorch.predict(X_test)

print("Accuracy score", sklearn.metrics.accuracy_score(y_test, y_pred))
```

More examples with datasets:

```sh
$ cd examples/

```

## Configuration

How to configure Auto-PyTorch for your needs:

```py

# Print all possible configuration options.
AutoNetClassification().print_help()

# You can use the constructor to configure Auto-PyTorch.
autoPyTorch = AutoNetClassification(log_level='info', max_runtime=300, min_budget=30, max_budget=90)

# You can overwrite this configuration in each fit call.
autoPyTorch.fit(X_train, y_train, log_level='debug', max_runtime=900, min_budget=50, max_budget=150)

# You can use presets to configure the config space.
# Available presets: full_cs, medium_cs (default), tiny_cs.
# These are defined in autoPyTorch/core/presets.
# tiny_cs is recommended if you want fast results with few resources.
# full_cs is recommended if you have many resources and a very high search budget.
autoPyTorch = AutoNetClassification("full_cs")

# Enable or disable components using the Auto-PyTorch config:
autoPyTorch = AutoNetClassification(networks=["resnet", "shapedresnet", "mlpnet", "shapedmlpnet"])

# You can take a look at the search space.
# Each hyperparameter belongs to a node in Auto-PyTorch's ML Pipeline.
# The names of the hyperparameters are prefixed with the name of the node: NodeName:hyperparameter_name.
# If a hyperparameter belongs to a component: NodeName:component_name:hyperparameter_name.
# Call with the same arguments as fit.
autoPyTorch.get_hyperparameter_search_space(X_train, y_train, validation_split=0.3)

# You can configure the search space of every hyperparameter of every component:
from autoPyTorch import HyperparameterSearchSpaceUpdates
search_space_updates = HyperparameterSearchSpaceUpdates()

search_space_updates.append(node_name="NetworkSelector",
                            hyperparameter="shapedresnet:activation",
                            value_range=["relu", "sigmoid"])
search_space_updates.append(node_name="NetworkSelector",
                            hyperparameter="shapedresnet:blocks_per_group",
                            value_range=[2,5],
                            log=False)
autoPyTorch = AutoNetClassification(hyperparameter_search_space_updates=search_space_updates)
```

Enable ensemble building (for featurized data):

```py
from autoPyTorch import AutoNetEnsemble
autoPyTorchEnsemble = AutoNetEnsemble(AutoNetClassification, "tiny_cs", max_runtime=300, min_budget=30, max_budget=90)

```

Disable pynisher if you experience issues when using cuda:

```py
autoPyTorch = AutoNetClassification("tiny_cs", log_level='info', max_runtime=300, min_budget=30, max_budget=90, cuda=True, use_pynisher=False)

```

## License

This program is free software: you can redistribute it and/or modify
it under the terms of the Apache license 2.0 (please see the LICENSE file).

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.

You should have received a copy of the Apache license 2.0
along with this program (see LICENSE file).