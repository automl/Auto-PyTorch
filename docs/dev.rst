:orphan:

.. _dev:

=======================
Developer Documentation
=======================

This documentation summarizes how the AutoPyTorch code works, and it is meant to guide developers
on how to best contribute to it.

AutoPyTorch relies on the `SMAC <https://automl.github.io/SMAC3/master/>`_ library to build individual models,
which are later ensembled together using ensemble selection by `Caruana et al. (2004) <https://dl.acm.org/doi/pdf/10.1145/1015330.1015432>`_.
Therefore, there are two main parts of the code: `AutoMLSMBO`, which is our interface to the SMAC package, and
`EnsembleBuilder` which opportunistically builds an ensemble of the individual algorithms found by SMAC, at fixed intervals.
The following sections provides details regarding this two main blocks of code.

Building Individual Models
==========================

AutoPyTorch relies on Scikit-Learn `Pipeline <https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html>`_ to build an individual algorithm.
In other words, each of the individual models fitted by SMAC are (and comply) with Scikit-Learn pipeline and framework. For example, when a pipeline is fitted,
we use pickle to save it to disk as stated `here <https://scikit-learn.org/stable/modules/model_persistence.html>`_. SMAC runs an optimization loop that proposes new
configurations based on bayesian optimization, which comply with the package `ConfigSpace <https://automl.github.io/ConfigSpace/master/>`_. These configurations are
translated to a pipeline configuration, fitted and saved to disc using the function evaluator `ExecuteTaFuncWithQueue`. The latter is basically a worker that that
reads a dataset from disc, fits a pipeline, and collect the performance result which is communicated back to the main process via a Queue. This worker manages
resources using `Pynisher <https://github.com/automl/pynisher>`_, and it usually does so by creating a new process.

The Scikit-learn pipeline inherits from the `BaseEstimator <https://scikit-learn.org/stable/modules/generated/sklearn.base.BaseEstimator.html>`_, which implies that we have to honor the `Scikit-Learn development Guidelines <https://scikit-learn.org/stable/developers/develop.html>`_. Of particular interest is that any estimator must define as attributes, the arguments that the class constructor receives (see `get_params and set_params` from the above documentation).

Regarding multiprocessing, AutoPyTorch and SMAC work with `Dask.distributed <https://distributed.dask.org/en/latest/>`_. We only submits jobs to Dask up to the number of 
workers, and wait for a worker to be available before continuing.

At the end of a SMAC runs, the results will be available in the `temporary_directory` provided to the API, in particular inside of the `<temporary_directory>/smac3-output/run_<SEED>/` directory. One can debug
the performance of the individual models using the file `runhistory.json` located in this area. Every individual model will be stored in `<temporary_directory>/.autoPyTorch/runs`. 
In this later directory we store the fitted model (during cross-validation we store a single Voting Classifier/Regressor, which is the soft voting outcome of k-Fold cross-validation), the Out-Of-Fold
predictions that are used to build an ensemble, and also the test predictions of this model in question.

Building the ensemble model
===========================

At every smac iteration, we submit a callback to create an ensemble in the case new models are written to disk. If no new models are available, no ensemble selection 
is triggered. We use the OutOfFold predictions to build an ensemble via `EnsembleSelection`. This process is also submitted to Dask. Every new ensemble that is fitted is also
written to disk, where this object is mainly a container that specifies the weights one should use, to join individual model predictions.

The AutoML Part
===============

The ensemble builder and the individual model constructions are both regulated by the `BaseTask`. This entity fundamentally calls the aforementioned task, and wait until
the time resource is exhausted.

We also rely on the `ConfigSpace <https://automl.github.io/ConfigSpace/master/index.html>`_ package to build a configuration space and sample configurations from it. A configuration in this context, determines the content of a pipeline (for example, that the final estimator will be a MLP, or that it will have PCA as preprocessing).The set of valid configurations is determined by the configuration space. The configuration space is build using the dataset characteristics, like type
of features (categorical, numerical) or the target type (classification, regression).
