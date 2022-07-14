:orphan:

.. _manual:

======
Manual
======

This manual shows how to get started with Auto-PyTorch. We recommend going over the examples first.
There are additional recommendations on how to interact with the API, further below in this manual.
However, you are welcome to contribute to this documentation by making a Pull-Request.

The searching starts by calling `search()` function of each supported task.
Currently, we are supporting Tabular classification and Tabular Regression.
We expand the support to image processing tasks in the future.

Examples
========
* `Classification <examples/20_basics/example_tabular_classification.html>`_
* `Regression <examples/20_basics/example_tabular_regression.html>`_
* `Forecasting <examples/20_basic/example_time_series_forecasting.html>`_
* `Customizing the search space <examples/40_advanced/example_custom_configuration_space.html>`_
* `Changing the resampling strategy <examples/40_advanced/example_resampling_strategy.html>`_
* `Visualizing the results <examples/40_advanced/example_visualization.html>`_

Data validation
===============
For **tabular tasks**, *Auto-PyTorch* uses a feature and target validator on the input feature set and target set respectively.

The feature validator checks whether the data is supported by *Auto-PyTorch* or not. Additionally, a sklearn column transformer
is also used which imputes and ordinally encodes the categorical columns of the dataset. This ensures
that no unseen category is found while fitting the data.

The target validator applies a label encoder on the target column.

For **time series forecasting tasks**, besides the functions described above, time series forecasting validators will also
check the information specify for time series forecasting tasks: it checks

* The index of the series that each data point belongs to
* If the dataset is uni-variant (only targets information is contained in the datasets)
* The sample frequency of the datasets
* The static features in the dataset, i.e., features that contain only one value within each series

Time Series forecasting validator then transforms the features and targets into a `pd.DataFrame <https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html>`_
whose index is applied to identify the series that the time step belongs to.

Data Preprocessing
==================
The **tabular preprocessing pipeline** in *Auto-PyTorch* consists of

#. `Imputation <https://github.com/automl/Auto-PyTorch/tree/development/autoPyTorch/pipeline/components/preprocessing/tabular_preprocessing/imputation>`_
#. `Encoding <https://github.com/automl/Auto-PyTorch/tree/development/autoPyTorch/pipeline/components/preprocessing/tabular_preprocessing/encoding>`_
        Choice of `OneHotEncoder` or no encoding.
#. `Scaling <https://github.com/automl/Auto-PyTorch/tree/development/autoPyTorch/pipeline/components/preprocessing/tabular_preprocessing/scaling>`_
        Choice of `MinMaxScaler`, `Normalizer`, `StandardScaler` and no scaling.
#. `Feature preprocessing <https://github.com/automl/Auto-PyTorch/tree/development/autoPyTorch/pipeline/components/preprocessing/tabular_preprocessing/feature_preprocessing>`_
        Choice of `FastICA`, `KernelPCA`, `RandomKitchenSinks`, `Nystroem`, `PolynomialFeatures`, `PowerTransformer`, `TruncatedSVD`,

Along with the choices, their corresponding hyperparameters are also tuned. A sklearn ColumnTransformer is
created which includes a categorical pipeline and a numerical pipeline. These pipelines are made up of the 
relevant preprocessors chosen in the previous steps. The column transformer is compatible with `torchvision transforms <https://pytorch.org/vision/stable/transforms.html>`_
and is therefore passed to the DataLoader.

**time series forecasting pipeline** has two sorts of setup:

- Uni-variant model only requires target transformations. They include *1:
    #. `Target Imputation <https://github.com/automl/Auto-PyTorch/tree/development/autoPyTorch/pipeline/components/preprocessing/time_series_preprocessing/imputation/>`_
        Choice of `linear`, `nearest`, `constant_zero`, `bfill` and `ffill`
- Multi-variant model contains target transformations (see above) and feature transformation. They include
    #. `Imputation <https://github.com/automl/Auto-PyTorch/tree/development/autoPyTorch/pipeline/components/preprocessing/time_series_preprocessing/imputation>`_
         Choice of `linear`, `nearest`, `constant_zero`, `bfill` and `ffill`
    #. `Scaling <https://github.com/automl/Auto-PyTorch/tree/development/autoPyTorch/pipeline/components/preprocessing/time_series_preprocessing/scaling>`_
        Choice of `standard`, `min_max`, `max_abs`, `mean_abs`, or no transformation *2
    #. `Encoding <https://github.com/automl/Auto-PyTorch/tree/development/autoPyTorch/pipeline/components/preprocessing/time_series_preprocessing/encoding>`_
        Choice of `OneHotEncoder` or no encoding.

*1 Target scaling is considered as part of `setup <https://github.com/automl/Auto-PyTorch/tree/development/autoPyTorch/pipeline/components/setup>`_ and the transform is done within each batch iteration

*2 Scaling is transformed within each series

Resource Allocation
===================

*Auto-PyTorch* allows to control the maximum allowed resident set size memory (RSS) that an estimator can use. 
By providing the `memory_limit` argument to the `search()` method, one can make sure that neither the individual 
machine learning models fitted by SMAC nor the final ensemble consume more than `memory_limit` megabytes.

Additionally, one can control the allocated time to search for a model via the argument `total_walltime_limit` 
to the `search()` method. This argument controls the total time SMAC can use to search for new configurations. 
The more time is allocated, the better the final estimator will be.

Ensemble Building Process
=========================

*Auto-PyTorch* uses ensemble selection by `Caruana et al. (2004) <https://dl.acm.org/doi/pdf/10.1145/1015330.1015432>`_
to build an ensemble based on the models’ prediction for the validation set. The following hyperparameters control how the ensemble is constructed:

* ``ensemble_size`` 
        determines the maximal size of the ensemble. If it is set to zero, no ensemble will be constructed.
* ``ensemble_nbest`` 
        allows the user to directly specify the number of models considered for the ensemble. When an integer 
        is provided for this hyperparameter, the final ensemble chooses each predictor from only the best n models. 
        If a float between 0.0 and 1.0 is provided, ``ensemble_nbest`` would be interpreted as a fraction suggesting 
        the percentage of models to use in the ensemble building process (namely, if ensemble_nbest is a float, 
        library pruning is implemented as described in `Caruana et al. (2006) <https://dl.acm.org/doi/10.1109/ICDM.2006.76>`_). 
        For example, if 10 candidates are available for the ensemble building process and the hyper-parameter is `ensemble_nbest==0.7``, 
        we build an ensemble by taking the best 7 models among the original 10 candidate models.
* ``max_models_on_disc`` 
        defines the maximum number of models that are kept on the disc, as a mechanism to control the amount of disc space 
        consumed by Auto-PyTorch. Throughout the automl process, different individual models are optimized, and their 
        predictions (and other metadata) are stored on disc. The user can set the upper bound on how many models are 
        acceptable to keep on disc, yet this variable takes priority in the definition of the number of models used by 
        the ensemble builder (that is, the minimum of ``ensemble_size``, ``ensemble_nbest`` and ``max_models_on_disc`` 
        determines the maximal amount of models used in the ensemble). If set to None, this feature is disabled.

Inspecting the results
======================

Auto-PyTorch allows users to inspect the training results and statistics. The following example shows how different statistics can be printed for the inspection.

>>> from autoPyTorch.api.tabular_classification import TabularClassificationTask
>>> automl = TabularClassificationTask()
>>> automl.fit(X_train, y_train)
>>> automl.show_models()

Parallel computation
====================

In it's default mode, *Auto-PyTorch* already uses two cores. The first one is used for model building, the second for building an ensemble every time a new machine learning model has finished training.

Nevertheless, *Auto-PyTorch* also supports parallel Bayesian optimization via the use of `Dask.distributed  <https://distributed.dask.org/>`_. By providing the arguments ``n_jobs`` to the estimator construction, one can control the number of cores available to *Auto-PyTorch* (As shown in the Example :ref:`sphx_glr_examples_40_advanced_example_parallel_n_jobs.py`). When multiple cores are available, *Auto-PyTorch* will create a worker per core, and use the  available workers to both search for better machine learning models as well as building  an ensemble with them until the time resource is exhausted.

**Note:** *Auto-PyTorch* requires all workers to have access to a shared file system for storing training data and models.

*Auto-PyTorch* employs `threadpoolctl <https://github.com/joblib/threadpoolctl/>`_ to control the number of threads employed by scientific libraries like numpy or scikit-learn. This is done exclusively during the building procedure of models, not during inference. In particular, *Auto-PyTorch* allows each pipeline to use at most 1 thread during training. At predicting and scoring time this limitation is not enforced by *Auto-PyTorch*. You can control the number of resources
employed by the pipelines by setting the following variables in your environment, prior to running *Auto-PyTorch*:

.. code-block:: shell-session

    $ export OPENBLAS_NUM_THREADS=1
    $ export MKL_NUM_THREADS=1
    $ export OMP_NUM_THREADS=1


For further information about how scikit-learn handles multiprocessing, please check the `Parallelism, resource management, and configuration <https://scikit-learn.org/stable/computing/parallelism.html>`_ documentation from the library.
