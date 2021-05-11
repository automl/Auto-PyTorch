from typing import Any, Dict, Optional

import numpy as np

from autoPyTorch.pipeline.components.base_component import autoPyTorchComponent


class autoPyTorchSetupComponent(autoPyTorchComponent):
    """Provide an abstract interface for schedulers
    in Auto-Pytorch"""

    def __init__(self, random_state: Optional[np.random.RandomState] = None) -> None:
        super(autoPyTorchSetupComponent, self).__init__(random_state=random_state)

    def transform(self, X: Dict[str, Any]) -> Dict[str, Any]:
        """
        Adds the fitted component into the fit dictionary 'X' and returns it.
        Args:
            X (Dict[str, Any]): 'X' dictionary
        Returns:
            (Dict[str, Any]): the updated 'X' dictionary
        """
        raise NotImplementedError()
