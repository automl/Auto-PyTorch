:orphan:

.. _manual:

======
Manual
======

This manual shows how to get started with Auto-PyTorch. We recommend going over the examples first.
There are additional recommendations on how to interact with the API, further below in this manual.
However, you are welcome to contribute to this documentation but starting a Pull-Request.

In a nutshell, Auto-PyTorch searches for the best ensemble of both traditional machine learning models and neural networks for a given dataset. It does so via the `search()` method of the different supported task. Currently we support Tabular classification and Tabular Regression. We plan to also support image processing.

Examples
========
* `Classification <examples/tabular/20_basics/example_tabular_classification.html>`_
* `Regression <examples/tabular/20_basics/example_tabular_regression.html>`_
* `Customizing the search space <examples/tabular/40_advanced/example_custom_configuration_space.html>`_
* `Changing the resampling strategy <examples/tabular/40_advanced/example_resampling_strategy.html>`_
* `Visualizing the results <examples/tabular/40_advanced/example_visualization.html>`_

Resource Allocation
===================

Auto-PyTorch allows to control the maximum allowed resident set memory that an estimator can use. By providing the `memory_limit` argument to the `search()` method, one can make sure that neither the individual machine learning models fitted by SMAC nor the final ensemble consume more than `memory_limit` megabytes.

Additionally, one can control the allocated time to search for a model, via the argument `total_walltime_limit` to the `search()` method. The later controls how much time SMAC can search for new configurations to solve the problem at hand. The more time is allocated, the better the final estimator will be.

Ensemble Building Process
=========================

Auto-PyTorch uses ensemble selection by `Caruana et al. (2004) <https://dl.acm.org/doi/pdf/10.1145/1015330.1015432>`_
to build an ensemble based on the models’ prediction for the validation set. The following hyperparameters control how the ensemble is constructed:

* ``ensemble_size`` determines the maximal size of the ensemble. If it is set to zero, no ensemble will be constructed.
* ``ensemble_nbest`` allows the user to directly specify the number of models considered for the ensemble.  This hyperparameter can be an integer *n*, such that only the best *n* models are used in the final ensemble. If a float between 0.0 and 1.0 is provided, ``ensemble_nbest`` would be interpreted as a fraction suggesting the percentage of models to use in the ensemble building process (namely, if ensemble_nbest is a float, library pruning is implemented as described in `Caruana et al. (2006) <https://dl.acm.org/doi/10.1109/ICDM.2006.76>`_).
* ``max_models_on_disc`` defines the maximum number of models that are kept on the disc, as a mechanism to control the amount of disc space consumed by *auto-sklearn*. Throughout the automl process, different individual models are optimized, and their predictions (and other metadata) is stored on disc. The user can set the upper bound on how many models are acceptable to keep on disc, yet this variable takes priority in the definition of the number of models used by the ensemble builder (that is, the minimum of ``ensemble_size``, ``ensemble_nbest`` and ``max_models_on_disc`` determines the maximal amount of models used in the ensemble). If set to None, this feature is disabled.

Inspecting the results
======================

Auto-PyTorch allows users to inspect the training results and statistics. The following example shows how different statistics can be printed for the inspection.

>>> from autoPyTorch.api.tabular_classification import TabularClassificationTask
>>> automl = TabularClassificationTask()
>>> automl.fit(X_train, y_train)
>>> automl.show_models()
