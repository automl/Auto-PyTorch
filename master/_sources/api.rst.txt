:orphan:

.. _api:

APIs
****

============
Main modules
============

~~~~~~~~~~~~~~
Classification
~~~~~~~~~~~~~~

.. autoclass:: autoPyTorch.api.tabular_classification.TabularClassificationTask
    :members:
    :inherited-members: search, refit, predict, score

~~~~~~~~~~~~~~
Regression
~~~~~~~~~~~~~~

.. autoclass:: autoPyTorch.api.tabular_regression.TabularRegressionTask
    :members:
    :inherited-members: search, refit, predict, score


=========
Pipelines
=========

~~~~~~~~~~~~~~~~~~~~~~
Tabular Classification
~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: autoPyTorch.pipeline.tabular_classification.TabularClassificationPipeline
    :members:

.. autoclass:: autoPyTorch.pipeline.traditional_tabular_classification.TraditionalTabularClassificationPipeline
    :members:

~~~~~~~~~~~~~~~~~~
Tabular Regression
~~~~~~~~~~~~~~~~~~

.. autoclass:: autoPyTorch.pipeline.tabular_regression.TabularRegressionPipeline
    :members:

.. autoclass:: autoPyTorch.pipeline.traditional_tabular_regression.TraditionalTabularRegressionPipeline
    :members:

=================
Steps in Pipeline
=================


~~~~~~~~~~~~~~~~~~~~
autoPyTorchComponent
~~~~~~~~~~~~~~~~~~~~

.. autoclass:: autoPyTorch.pipeline.components.base_component.autoPyTorchComponent
    :members:

~~~~~~~~~~~~~~~~~
autoPyTorchChoice
~~~~~~~~~~~~~~~~~

.. autoclass:: autoPyTorch.pipeline.components.base_choice.autoPyTorchChoice
    :members: