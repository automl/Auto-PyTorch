import collections
import os
from typing import Dict, Optional

import numpy as np

from autoPyTorch.constants import FORECASTING_BUDGET_TYPE, STRING_TO_TASK_TYPES
from autoPyTorch.datasets.base_dataset import BaseDatasetPropertiesType
from autoPyTorch.pipeline.components.base_component import (
    ThirdPartyComponents,
    autoPyTorchComponent,
    find_components
)
from autoPyTorch.pipeline.components.setup.forecasting_target_scaling.base_target_scaler import BaseTargetScaler
from autoPyTorch.pipeline.components.training.metrics.utils import get_metrics
from autoPyTorch.pipeline.components.training.trainer import TrainerChoice
from autoPyTorch.pipeline.components.training.trainer.base_trainer import BudgetTracker
from autoPyTorch.pipeline.components.training.trainer.forecasting_trainer.forecasting_base_trainer import (
    ForecastingBaseTrainerComponent
)
from autoPyTorch.utils.common import (
    FitRequirement,
    get_device_from_fit_dictionary
)

trainer_directory = os.path.split(__file__)[0]
_trainers = find_components(__package__,
                            trainer_directory,
                            ForecastingBaseTrainerComponent)
_addons = ThirdPartyComponents(ForecastingBaseTrainerComponent)


def add_trainer(trainer: ForecastingBaseTrainerComponent) -> None:
    _addons.add_component(trainer)


class ForecastingTrainerChoice(TrainerChoice):
    def __init__(self,
                 dataset_properties: Dict[str, BaseDatasetPropertiesType],
                 random_state: Optional[np.random.RandomState] = None
                 ):
        super().__init__(dataset_properties=dataset_properties, random_state=random_state)
        assert self._fit_requirements is not None
        self._fit_requirements.extend([FitRequirement("target_scaler", (BaseTargetScaler,),
                                                      user_defined=False, dataset_property=False),
                                       FitRequirement("window_size", (int,), user_defined=False,
                                                      dataset_property=False)])

    def get_budget_tracker(self, X: Dict) -> BudgetTracker:
        if 'epochs' in X:
            max_epochs = X['epochs']
        elif X['budget_type'] in FORECASTING_BUDGET_TYPE:
            max_epochs = 50
        else:
            max_epochs = None
        return BudgetTracker(
            budget_type='epochs' if X['budget_type'] in FORECASTING_BUDGET_TYPE else X['budget_type'],
            max_runtime=X['runtime'] if 'runtime' in X else None,
            max_epochs=max_epochs,
        )

    def prepare_trainer(self, X: Dict) -> None:
        assert self.choice is not None
        # Support additional user metrics
        metrics = get_metrics(dataset_properties=X['dataset_properties'])
        if 'additional_metrics' in X:
            metrics.extend(get_metrics(dataset_properties=X['dataset_properties'], names=X['additional_metrics']))
        if 'optimize_metric' in X and X['optimize_metric'] not in [m.name for m in metrics]:
            metrics.extend(get_metrics(dataset_properties=X['dataset_properties'], names=[X['optimize_metric']]))
        if hasattr(X['y_train'], "to_numpy"):
            labels = X['y_train'].to_numpy()[X['backend'].load_datamanager().splits[X['split_id']][0]]
        else:
            labels = X['y_train'][X['backend'].load_datamanager().splits[X['split_id']][0]]

        self.choice.prepare(
            model=X['network'],
            metrics=metrics,
            criterion=X['loss'],
            budget_tracker=self.budget_tracker,
            optimizer=X['optimizer'],
            device=get_device_from_fit_dictionary(X),
            metrics_during_training=X['metrics_during_training'],
            scheduler=X['lr_scheduler'],
            task_type=STRING_TO_TASK_TYPES[X['dataset_properties']['task_type']],
            labels=labels,
            step_interval=X['step_interval'],
            window_size=X['window_size'],
            dataset_properties=X['dataset_properties'],
            target_scaler=X['target_scaler'],
            backcast_loss_ratio=X.get('backcast_loss_ratio', 0.0)
        )

    def get_components(self) -> Dict[str, autoPyTorchComponent]:
        """Returns the available trainer components

        Args:
            None

        Returns:
            Dict[str, autoPyTorchComponent]: all components available
                as choices for learning rate scheduling
        """
        components: Dict[str, autoPyTorchComponent] = collections.OrderedDict()
        components.update(_trainers)
        components.update(_addons.components)
        return components
