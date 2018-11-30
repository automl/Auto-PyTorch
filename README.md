# AutoNet

### Installation

Clone repository

```sh
$ cd install/path
$ git clone https://bitbucket.org/aadfreiburg/autonet.git
$ cd autonet
```
If you want to contribute to this repository switch to our current develop branch

```sh
$ git checkout develop
```

Install pytorch: 
https://pytorch.org/
 
Install autonet

```sh
$ python setup.py
```


### Examples

In a nutshell:

```py
1: from autonet import AutoNetClassification
2: 
3: autonet = AutoNetClassification()
4: autonet.fit(X_train, Y_train)
5: Y_pred = autonet.predict(X_test)
```

More examples with datasets:

```sh
$ cd examples/
```
