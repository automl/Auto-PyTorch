from typing import Any, Dict, Optional

from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

from autoPyTorch.pipeline.components.setup.base_setup import autoPyTorchSetupComponent
from autoPyTorch.utils.common import FitRequirement


class BaseLRComponent(autoPyTorchSetupComponent):
    """Provide an abstract interface for schedulers
    in Auto-Pytorch"""

    def __init__(self) -> None:
        super().__init__()
        self.scheduler = None  # type: Optional[_LRScheduler]

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
        X.update({'lr_scheduler': self.scheduler})
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
        info = vars(self)
        # Remove unwanted info
        info.pop('scheduler', None)
        info.pop('random_state', None)
        string += " (" + str(info) + ")"
        return string
