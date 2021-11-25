from typing import Any, Dict, List, Optional, Tuple, Type, Union

from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import (
    CategoricalHyperparameter
)

import numpy as np

from autoPyTorch.pipeline.base_pipeline import BaseDatasetPropertiesType
from autoPyTorch.pipeline.components.setup.traditional_ml.base_model import BaseModelComponent
from autoPyTorch.pipeline.components.setup.traditional_ml.traditional_learner import (
    BaseTraditionalLearner, get_available_traditional_learners)


class TabularTraditionalModel(BaseModelComponent):
    """
    Implementation of a dynamic model, that consists of a learner and a head
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
        self._traditional_learners = get_available_traditional_learners()

    @staticmethod
    def get_properties(
        dataset_properties: Optional[Dict[str, BaseDatasetPropertiesType]] = None
    ) -> Dict[str, Union[str, bool]]:
        return {
            "shortname": "TabularTraditionalModel",
            "name": "Tabular Traditional Model",
        }

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties: Optional[Dict[str, BaseDatasetPropertiesType]] = None,
                                        **kwargs: Any) -> ConfigurationSpace:
        cs = ConfigurationSpace()
        traditional_learners: Dict[str, Type[BaseTraditionalLearner]] = get_available_traditional_learners()
        # Remove knn if data is all categorical

        if dataset_properties is not None:
            numerical_columns = dataset_properties['numerical_columns'] \
                if isinstance(dataset_properties['numerical_columns'], List) else []
            if len(numerical_columns) == 0:
                del traditional_learners['knn']
        learner_hp = CategoricalHyperparameter("traditional_learner", choices=traditional_learners.keys())
        cs.add_hyperparameters([learner_hp])

        return cs

    def build_model(self, input_shape: Tuple[int, ...], output_shape: Tuple[int, ...],
                    logger_port: int, task_type: str, output_type: str, optimize_metric: Optional[str] = None
                    ) -> BaseTraditionalLearner:
        """
        This method returns a traditional learner, that is dynamically
        built using a self.config that is model specific, and contains
        the additional configuration hyperparameters to build a domain
        specific model
        """
        learner_name = self.config["traditional_learner"]
        Learner = self._traditional_learners[learner_name]

        learner = Learner(random_state=self.random_state, logger_port=logger_port,
                          task_type=task_type, output_type=output_type, optimize_metric=optimize_metric)

        return learner

    def __str__(self) -> str:
        """ Allow a nice understanding of what components where used """
        return f"TabularTraditionalModel: {self.model.name if self.model is not None else None}"
