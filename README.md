# Auto-PyTorch

Copyright (C) 2021  [AutoML Groups Freiburg and Hannover](http://www.automl.org/)

While early AutoML frameworks focused on optimizing traditional ML pipelines and their hyperparameters, another trend in AutoML is to focus on neural architecture search. To bring the best of these two worlds together, we developed **Auto-PyTorch**, which jointly and robustly optimizes the network architecture and the training hyperparameters to enable fully automated deep learning (AutoDL).

Auto-PyTorch is mainly developed to support tabular data (classification, regression), but can also be applied to image data (classification).
The newest features in Auto-PyTorch for tabular data are described in the paper ["Auto-PyTorch Tabular: Multi-Fidelity MetaLearning for Efficient and Robust AutoDL"](https://arxiv.org/abs/2006.13799) (see below for bibtex ref).

## Alpha Status of Next Release

The upcoming release of Auto-PyTorch will further improve usability, robustness and efficiency by using SMAC as the underlying optimization package, changing the code structure and other improvements. If you would like to give it a try, check out the `development` branch or it's [documentation](https://automl.github.io/Auto-PyTorch/development/).

## Installation

### Pip

```sh
$ cd install/path
$ git clone https://github.com/automl/Auto-PyTorch.git
$ cd Auto-PyTorch
```
If you want to contribute to this repository switch to our current development branch

```sh
$ git checkout development
```

## Contributing

If you want to contribute to Auto-PyTorch, clone the repository and checkout our current development branch

```sh
$ git checkout refactor_development
```


## Examples

For a detailed tutorial, please refer to the jupyter notebook in https://github.com/automl/Auto-PyTorch/tree/master/examples/basics.

In a nutshell:

```py
from autoPyTorch import TODO
```

For ore examples, checkout `examples/`.


## Configuration

### Pipeline configuration

### Search space

### Fitting single configurations


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
