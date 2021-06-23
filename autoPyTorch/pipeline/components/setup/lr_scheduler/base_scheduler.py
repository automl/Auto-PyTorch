from typing import Any, Dict, Optional, Union

from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

from autoPyTorch.pipeline.components.setup.base_setup import autoPyTorchSetupComponent
from autoPyTorch.pipeline.components.training.trainer.base_trainer import StepIntervalUnit, StepIntervalUnitChoices
from autoPyTorch.utils.common import FitRequirement


class BaseLRComponent(autoPyTorchSetupComponent):
    """Provide an abstract interface for schedulers
    in Auto-Pytorch"""

    def __init__(self, step_interval: Union[str, StepIntervalUnit]):
        super().__init__()
        self.scheduler = None  # type: Optional[_LRScheduler]

        if isinstance(step_interval, str) and step_interval not in StepIntervalUnitChoices:
            raise ValueError('step_interval must be either {}, but got {}.'.format(
                StepIntervalUnitChoices,
                step_interval
            ))

        self.step_interval = step_interval if isinstance(step_interval, str) else step_interval.name

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
        step_interval = getattr(StepIntervalUnit, self.step_interval)

        X.update(
            lr_scheduler=self.scheduler,
            step_interval=step_interval
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
