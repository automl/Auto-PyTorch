from abc import abstractstaticmethod
from typing import Any, Dict, Optional, Tuple, Union

from ConfigSpace.configuration_space import ConfigurationSpace

from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

from autoPyTorch.datasets.base_dataset import BaseDatasetPropertiesType
from autoPyTorch.pipeline.components.setup.base_setup import autoPyTorchSetupComponent
from autoPyTorch.pipeline.components.training.trainer.base_trainer import StepIntervalUnit, StepIntervalUnitChoices
from autoPyTorch.utils.common import FitRequirement, HyperparameterSearchSpace


class BaseLRComponent(autoPyTorchSetupComponent):
    """Provide an abstract interface for schedulers
    in Auto-Pytorch"""

    def __init__(self, step_unit: Union[str, StepIntervalUnit]) -> None:
        super().__init__()
        self.scheduler = None  # type: Optional[_LRScheduler]
        if isinstance(step_unit, str) and not hasattr(StepIntervalUnit, step_unit):
            raise ValueError('step_unit must either {}, but got {}'.format(
                StepIntervalUnitChoices,
                step_unit
            ))

        self.step_unit = step_unit

        self.add_fit_requirements([
            FitRequirement('optimizer', (Optimizer,), user_defined=False, dataset_property=False)])

    def transform(self, X: Dict[str, Any]) -> Dict[str, Any]:
        """
        Adds the scheduler into the fit dictionary 'X' and returns it.
        Args:
            X (Dict[str, Any]): 'X' dictionary
        Returns:
            (Dict[str, Any]): the updated 'X' dictionary
        """

        # This processing is an ad-hoc handling of the dependencies because of ConfigSpace and unittest
        step_unit = getattr(StepIntervalUnit, self.step_unit) if isinstance(self.step_unit, str) else self.step_unit

        X.update(
            lr_scheduler=self.scheduler,
            step_unit=step_unit
        )
        return X

    def get_scheduler(self) -> _LRScheduler:
        """Return the underlying scheduler object.
        Returns:
            scheduler : the underlying scheduler object
        """
        assert self.scheduler is not None, "No scheduler was fit"
        return self.scheduler

    def __str__(self) -> str:
        """ Allow a nice understanding of what components where used """
        string = self.scheduler.__class__.__name__
        return string
