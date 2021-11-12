.. _dev:

=======================
Developer Documentation
=======================

This document summarizes how the AutoPyTorch code works and is meant as a guide for the developers looking to contribute.

AutoPyTorch relies on the `SMAC <https://automl.github.io/SMAC3/master/>`_ library to build individual models.
Note that SMAC runs an optimization loop that proposes new configurations
based on Bayesian optimization, which comply with
the package `ConfigSpace <https://automl.github.io/ConfigSpace/master/>`_.
Individual models evaluated during the optimization are later ensembled together
using ensemble selection by `Caruana et al. (2004) <https://dl.acm.org/doi/pdf/10.1145/1015330.1015432>`_.

In other words, there are two main parts of the code:

#. `AutoMLSMBO`: Interface to SMAC
#. `EnsembleBuilder`: Build an ensemble of the individual algorithms found by SMAC at fixed intervals

The ensemble builder and the individual model constructions are both regulated by the `BaseTask`.
It fundamentally calls the aforementioned task and waits until the time resource is exhausted.

The following sections provide details regarding these two main blocks of code.

Building Individual Models
==========================

AutoPytorch first preprocesses a given dataset and then it starts the training of individual algorithm.
The preprocessing and training rely on Scikit-Learn `Pipeline <https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html>`_.
Each of the individual models fitted by SMAC is (and comply) with Scikit-Learn pipeline and framework.

The Scikit-learn pipeline inherits from the `BaseEstimator <https://scikit-learn.org/stable/modules/generated/sklearn.base.BaseEstimator.html>`_,
which implies that we have to honor the `Scikit-Learn development Guidelines <https://scikit-learn.org/stable/developers/develop.html>`_.
Particularly, the arguments to the class constructor of any estimator must be defined as attributes of the class
(see `get_params and set_params` from the above documentation).

Pipeline of individual models
-----------------------------
A pipeline consists of various steps each consisting of either an `autoPyTorchChoice` or an `autoPyTorchComponent` both implemented with `Scikit-learn Base Estimator Guidelines <https://scikit-learn.org/stable/developers/develop.html#rolling-your-own-estimator>`_. 

These steps include:

#. `Imputation <https://github.com/automl/Auto-PyTorch/tree/development/autoPyTorch/pipeline/components/preprocessing/tabular_preprocessing/imputation>`_
#. `Encoding <https://github.com/automl/Auto-PyTorch/tree/development/autoPyTorch/pipeline/components/preprocessing/tabular_preprocessing/encoding>`_
#. `Scaling <https://github.com/automl/Auto-PyTorch/tree/development/autoPyTorch/pipeline/components/preprocessing/tabular_preprocessing/scaling>`_
#. `Feature preprocessing <https://github.com/automl/Auto-PyTorch/tree/development/autoPyTorch/pipeline/components/preprocessing/tabular_preprocessing/feature_preprocessing>`_
#. `Algorithm setup <https://github.com/automl/Auto-PyTorch/tree/development/autoPyTorch/pipeline/components/setup>`_
#. `Training <https://github.com/automl/Auto-PyTorch/tree/development/autoPyTorch/pipeline/components/training>`_

In the case of tabular classification/regression,
the training data is preprocessed using scikit-learn.compose.ColumnTransformer
on a per-column basis.
The data preprocessing is dynamically created depending on the dataset properties.
For example, on a dataset that only contains float-type features,
no one-hot encoding is needed.
Additionally, we wrap the ColumnTransformer via TabularColumnTransformer class
to support torchvision transformation and
handle column-reordering.
Note that column-reordering shifts categorical columns to the earlier indices
and it is activated only if one uses a ColumnTransformer.

Training of individual models
-----------------------------

Auto-PyTorch can fit 3 types of pipelines:

#. Dummy pipeline: Use sklearn.dummy to construct an estimator that predicts using simple rules such as most frequent class
#. Traditional machine learning pipelines: Use LightGBM, CatBoost, RandomForest, ExtraTrees, K-Nearest-Neighbors, and SupportVectorMachines
#. PyTorch neural networks: Neural architecture search of backbones (feature extraction) and network heads (for the final prediction)

Note that dummy pipeline is used as a baseline to define the worst-performing model that can be fit
and traditional machine learning pipelines are critical for small-sized datasets.
A pipeline might also contain additional training components
like learning rate scheduler, optimizers,
and data loaders required to perform the neural architecture search.

After the training (fitting a pipeline), we use pickle to save it
to disk as stated `here <https://scikit-learn.org/stable/modules/model_persistence.html>`_.

Optimization of pipeline
------------------------

To optimize pipeline, we use SMAC as mentioned earlier.
Given a configuration, AutoPytorch pipeline fits a pipeline and 
and finally saves to disc using the function evaluator `ExecuteTaFuncWithQueue`.
`ExecuteTaFuncWithQueue` is basically a worker that reads a dataset from disc,
fits a pipeline, and collects the performance result,
which is communicated back to the main process via a Queue.
This worker manages resources using `Pynisher <https://github.com/automl/pynisher>`_,
and it usually does so by creating a new process with a restricted memory
(`memory_limit` API argument)
and time constraints (`func_eval_time_limit_secs` API argument).

To speed up the search, AutoPyTorch and SMAC use 
`Dask.distributed <https://distributed.dask.org/en/latest/>`_
multiprocessing scheme.
We only submits jobs to Dask.distributed.Client up to the number of workers,
and wait for a worker to be available
before continuing searching for more pipelines.

At the end of SMAC, the results will be available in the `temporary_directory` provided to the API run,
in particular inside of the `<temporary_directory>/smac3-output/run_<SEED>/`
directory.
One can debug the performance of the individual models using the file `runhistory.json`
located in the same directory.
Every individual model will be stored in `<temporary_directory>/.autoPyTorch/runs`. 
In this `runs` directory, we store:

#. Fitted model
#. Test predictions of the model
#. Out-Of-Fold predictions (that are used to build an ensemble)

Note that we store a single Voting Classifier/Regressor,
which is the soft voting outcome of k-Fold cross-validation during cross-validation.

Building the ensemble model
===========================

At every SMAC iteration, we submit a callback to create an ensemble
in the case new models are written to disk.
If no new models are available, no ensemble selection is triggered.
We use the Out-Of-Fold predictions to build an ensemble via `EnsembleSelection`.
This process is also submitted to Dask.
Every new fitted ensemble is also written to disk,
where this object is mainly a container that specifies the weights one should use,
to join individual model predictions.

Search Space
============

We also rely on the
`ConfigSpace package <https://automl.github.io/ConfigSpace/master/index.html>`_
to build a configuration space and sample configurations from it.
In this context, a configuration determines the content of a pipeline.
For example, the choice of model such as MLP, random forest or
whether the pipeline has PCA as preprocessing can be elements of a configuration.
The set of valid configurations is specified by the configuration space.
The configuration space changes by the dataset characteristics,
like type of features (categorical, numerical) or
the target type (classification, regression).
