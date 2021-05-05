# Auto-PyTorch

Copyright (C) 2021  [AutoML Groups Freiburg and Hannover](http://www.automl.org/)

While early AutoML frameworks focused on optimizing traditional ML pipelines and their hyperparameters, another trend in AutoML is to focus on neural architecture search. To bring the best of these two worlds together, we developed **Auto-PyTorch**, which jointly and robustly optimizes the network architecture and the training hyperparameters to enable fully automated deep learning (AutoDL).

Auto-PyTorch is mainly developed to support tabular data (classification, regression), but can also be applied to image data (classification).
The newest features in Auto-PyTorch for tabular data are described in the paper ["Auto-PyTorch Tabular: Multi-Fidelity MetaLearning for Efficient and Robust AutoDL"](https://arxiv.org/abs/2006.13799) (see below for bibtex ref).

## Alpha Status of Next Release

The upcoming release of Auto-PyTorch will further improve usability, robustness and efficiency by using SMAC as the underlying optimization package, changing the code structure and other improvements. If you would like to give it a try, check out the `refactor` branch.

## Installation

Clone repository

```sh
$ cd install/path
$ git clone https://github.com/automl/Auto-PyTorch.git
$ cd Auto-PyTorch
```
If you want to contribute to this repository switch to our current develop branch

```sh
$ git checkout develop
```

Install pytorch: 
https://pytorch.org/

Install Auto-PyTorch:

```sh
$ cat requirements.txt | xargs -n 1 -L 1 pip install
$ python setup.py install
```


## Examples

Code for the [paper](https://arxiv.org/abs/2006.13799) is available under `examples/ensemble`.

For a detailed tutorial, please refer to the jupyter notebook in https://github.com/automl/Auto-PyTorch/tree/master/examples/basics.

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

## Reference

```bibtex
  @article{zimmer-tpami21a,
  author = {Lucas Zimmer and Marius Lindauer and Frank Hutter},
  title = {Auto-PyTorch Tabular: Multi-Fidelity MetaLearning for Efficient and Robust AutoDL},
  journal = {IEEE Transactions on Pattern Analysis and Machine Intelligence},
  year = {2021},
  note = {IEEE early access; also available under https://arxiv.org/abs/2006.13799},
  pages = {1-12}
}
```

```bibtex
@incollection{mendoza-automlbook18a,
  author    = {Hector Mendoza and Aaron Klein and Matthias Feurer and Jost Tobias Springenberg and Matthias Urban and Michael Burkart and Max Dippel and Marius Lindauer and Frank Hutter},
  title     = {Towards Automatically-Tuned Deep Neural Networks},
  year      = {2018},
  month     = dec,
  editor    = {Hutter, Frank and Kotthoff, Lars and Vanschoren, Joaquin},
  booktitle = {AutoML: Methods, Sytems, Challenges},
  publisher = {Springer},
  chapter   = {7},
  pages     = {141--156}
}
```

## Contact

Auto-PyTorch is developed by the [AutoML Groups of the University of Freiburg and Hannover](http://www.automl.org/).
