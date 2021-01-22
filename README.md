# Auto-PyTorch

Copyright (C) 2019  [AutoML Group Freiburg](http://www.automl.org/)

This an alpha version of Auto-PyTorch.
So far, Auto-PyTorch supports tabular data (classification, regression), image data (classification) and time-series data (TODO).


## Installation

### Pip
```sh
$ pip install autoPyTorch
```

### Manually
```sh
$ cd install/path
$ git clone https://github.com/automl/Auto-PyTorch.git
$ cd Auto-PyTorch
$ cat requirements.txt | xargs -n 1 -L 1 pip install
$ python setup.py install
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

```
@incollection{mendoza-automlbook18a,
  author    = {Hector Mendoza and Aaron Klein and Matthias Feurer and Jost Tobias Springenberg and Matthias Urban and Michael Burkart and Max Dippel and Marius Lindauer and Frank Hutter},
  title     = {Towards Automatically-Tuned Deep Neural Networks},
  year      = {2018},
  month     = dec,
  editor    = {Hutter, Frank and Kotthoff, Lars and Vanschoren, Joaquin},
  booktitle = {AutoML: Methods, Sytems, Challenges},
  publisher = {Springer},
  chapter   = {7},
  pages     = {141--156},
  note      = {To appear.},
}
```

**Note**: Previously, the name of the project was AutoNet. Since this was too generic, we changed the name to AutoPyTorch. AutoNet 2.0 in the reference mention above is indeed AutoPyTorch.


## Contact

Auto-PyTorch is developed by the [AutoML Group of the University of Freiburg](http://www.automl.org/).
