from typing import Any, Dict, Optional, Union

from ConfigSpace.configuration_space import ConfigurationSpace

import numpy as np

import pandas as pd

from scipy.sparse import spmatrix

import torch

from autoPyTorch.automl_common.common.utils.backend import Backend
from autoPyTorch.datasets.base_dataset import BaseDatasetPropertiesType
from autoPyTorch.pipeline.components.base_component import autoPyTorchComponent
from autoPyTorch.utils.common import FitRequirement
from autoPyTorch.pipeline.components.preprocessing.base_preprocessing import autoPyTorchPreprocessingComponent


class autoPyTorchTargetPreprocessingComponent(autoPyTorchPreprocessingComponent):
    """
     Provides abstract interface for preprocessing algorithms in AutoPyTorch.
    """
    def __init__(self) -> None:
        autoPyTorchComponent.__init__()
        self.add_fit_requirements([
            FitRequirement('y_train',
                           (pd.DataFrame, ),
                           user_defined=True, dataset_property=False),
            FitRequirement('backend',
                           (Backend, ),
                           user_defined=True, dataset_property=False)])
