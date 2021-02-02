************
Auto-PyTorch
************

.. role:: bash(code)
    :language: bash

.. role:: python(code)
    :language: python

*Auto-PyTorch* is an automated machine learning toolkit based on PyTorch:

    >>> import autoPyTorch
    >>> cls = autoPyTorch.api.tabular_classification.TabularClassificationTask()
    >>> cls.search(X_train, y_train)
    >>> predictions = cls.predict(X_test)

*Auto-PyTorch* frees a machine learning user from algorithm selection and
hyperparameter tuning. It leverages recent advantages in *Bayesian
optimization*, *meta-learning* and *ensemble construction*. 
Learn more about *Auto-PyTorch* by reading our paper
`Auto-PyTorch Tabular: Multi-Fidelity MetaLearning for Efficient and Robust AutoDL <https://arxiv.org/abs/2006.13799>`_
.

Example
*******

Manual
******

* :ref:`installation`
* :ref:`manual`
* :ref:`api`
* :ref:`extending`


License
*******
*Auto-PyTorch* is licensed the same way as *scikit-learn*,
namely the 3-clause BSD license.

Citing Auto-PyTorch 
*******************

If you use *Auto-PyTorch* in a scientific publication, we would appreciate a
reference to the following paper:


 `Auto-PyTorch Tabular: Multi-Fidelity MetaLearning for Efficient and Robust AutoDL
 <https://arxiv.org/abs/2006.13799>`_,

 Bibtex entry::

     @article{zimmer2020auto,
        title={Auto-pytorch tabular: Multi-fidelity metalearning for efficient and robust autodl},
        author={Zimmer, Lucas and Lindauer, Marius and Hutter, Frank},
        journal={arXiv preprint arXiv:2006.13799},
        year={2020}
     }

Contributing
************

We appreciate all contribution to *Auto-PyTorch*, from bug reports and
documentation to new features. If you want to contribute to the code, you can
pick an issue from the `issue tracker <https://github.com/automl/Auto-PyTorch/issues>`_
which is marked with `Needs contributer`.

.. note::

    To avoid spending time on duplicate work or features that are unlikely to
    get merged, it is highly advised that you contact the developers
    by opening a `github issue <https://github
    .com/automl/Auto-PyTorch/issues>`_ before starting to work.

When developing new features, please create a new branch from the development
branch. When to submitting a pull request, make sure that all tests are
still passing.
