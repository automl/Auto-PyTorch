from typing import Any, Dict, Optional, Tuple, Type

from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import (
    CategoricalHyperparameter
)

import numpy as np

from autoPyTorch.pipeline.components.setup.traditional_ml.base_model import BaseModelComponent
from autoPyTorch.pipeline.components.setup.traditional_ml.classifier_models import (
    BaseClassifier, get_available_classifiers)


class TabularClassifier(BaseModelComponent):
    """
    Implementation of a dynamic model, that consists of a classifier and a head
    """

    def __init__(
            self,
            random_state: Optional[np.random.RandomState] = None,
            **kwargs: Any
    ):
        super().__init__(
            random_state=random_state,
        )
        self.config = kwargs
        self._classifiers = get_available_classifiers()

    @staticmethod
    def get_properties(dataset_properties: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        return {
            "shortname": "TabularClassifier",
            "name": "TabularClassifier",
        }

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties: Optional[Dict[str, str]] = None,
                                        **kwargs: Any) -> ConfigurationSpace:
        cs = ConfigurationSpace()
        classifiers: Dict[str, Type[BaseClassifier]] = get_available_classifiers()
        # Remove knn classifier if data is all categorical
        if dataset_properties is not None and len(dataset_properties['numerical_columns']) == 0:
            del classifiers['knn_classifier']
        classifier_hp = CategoricalHyperparameter("classifier", choices=classifiers.keys())
        cs.add_hyperparameters([classifier_hp])

        return cs

    def build_model(self, input_shape: Tuple[int, ...], output_shape: Tuple[int, ...]) -> BaseClassifier:
        """
        This method returns a classifier, that is dynamically built using
        a self.config that is model specific, and contains the additional
        configuration hyperparameters to build a domain specific model
        """
        classifier_name = self.config["classifier"]
        Classifier = self._classifiers[classifier_name]

        classifier = Classifier()

        return classifier

    def __str__(self) -> str:
        """ Allow a nice understanding of what components where used """
        info = vars(self)
        # Remove unwanted info
        info.pop('random_state', None)
        info.pop('fit_output', None)
        info.pop('config', None)
        return f"TabularClassifier: {self.model.name if self.model is not None else None} ({str(info)})"
