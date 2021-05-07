# Auto-PyTorch

Copyright (C) 2019  [AutoML Group Freiburg](http://www.automl.org/)

This an alpha version of Auto-PyTorch with improved API.
So far, Auto-PyTorch supports tabular data (classification, regression).
We plan to enable image data and time-series data.


Find the documentation [here](https://automl.github.io/Auto-PyTorch/development)


## Installation

### Pip

We recommend using Anaconda for developing as follows:

```sh
# Following commands assume the user is in a cloned directory of Auto-Pytorch
conda create -n autopytorch python=3.8
conda activate autopytorch
conda install gxx_linux-64 gcc_linux-64 swig
cat requirements.txt | xargs -n 1 -L 1 pip install
python setup.py install

```

## Contributing

If you want to contribute to Auto-PyTorch, clone the repository and checkout our current development branch

```sh
$ git checkout development
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

Please refer to the branc `TPAMI.2021.3067763` to reproduce the paper *Auto-PyTorch Tabular: Multi-Fidelity MetaLearning for Efficient and Robust AutoDL*.

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

Auto-PyTorch is developed by the [AutoML Group of the University of Freiburg](http://www.automl.org/).
