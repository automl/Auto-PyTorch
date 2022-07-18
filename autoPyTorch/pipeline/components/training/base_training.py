from typing import Any, Dict, Optional

import numpy as np

from autoPyTorch.pipeline.components.base_component import autoPyTorchComponent


class autoPyTorchTrainingComponent(autoPyTorchComponent):
    """Provide an abstract interface for training nodes
    in Auto-Pytorch"""

    def __init__(self, random_state: Optional[np.random.RandomState] = None) -> None:
        super(autoPyTorchTrainingComponent, self).__init__(random_state=random_state)

    def transform(self, X: Dict) -> Dict:
        """The transform function calls the transform function of the
        underlying model and returns the transformed array.

        Args:
            X (Dict): input features

        Returns:
            Dict: Transformed features
        """
        raise NotImplementedError()

    def check_requirements(self, X: Dict[str, Any], y: Any = None) -> None:
        """
        A mechanism in code to ensure the correctness of the fit dictionary
        It recursively makes sure that the children and parent level requirements
        are honored before fit.

        Args:
            X (Dict[str, Any]): Dictionary with fitted parameters. It is a message passing
                mechanism, in which during a transform, a components adds relevant information
                so that further stages can be properly fitted
        """
        pass
