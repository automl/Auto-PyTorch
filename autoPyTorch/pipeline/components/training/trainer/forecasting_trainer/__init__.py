import collections
import os

from typing import Any, Dict, List, Optional, Tuple, cast

from autoPyTorch.pipeline.components.training.trainer.forecasting_trainer.forecasting_base_trainer import (
    ForecastingBaseTrainerComponent,
)

from autoPyTorch.constants import STRING_TO_TASK_TYPES

from autoPyTorch.pipeline.components.base_component import (
    ThirdPartyComponents,
    autoPyTorchComponent,
    find_components,
)
from autoPyTorch.utils.common import FitRequirement, get_device_from_fit_dictionary
from autoPyTorch.pipeline.components.training.losses import get_loss
from autoPyTorch.pipeline.components.training.metrics.utils import get_metrics
from autoPyTorch.pipeline.components.preprocessing.time_series_preprocessing.forecasting_target_scaling. \
    base_target_scaler import BaseTargetScaler

from autoPyTorch.utils.common import get_device_from_fit_dictionary

trainer_directory = os.path.split(__file__)[0]
_trainers = find_components(__package__,
                            trainer_directory,
                            ForecastingBaseTrainerComponent)
_addons = ThirdPartyComponents(ForecastingBaseTrainerComponent)


def add_trainer(trainer: ForecastingBaseTrainerComponent) -> None:
    _addons.add_component(trainer)


from autoPyTorch.pipeline.components.training.trainer import TrainerChoice


class ForecastingTrainerChoice(TrainerChoice):
    @property
    def _fit_requirements(self) -> Optional[List[FitRequirement]]:
        fit_requirements = super()._fit_requirements
        fit_requirements.append(FitRequirement("target_scaler", (BaseTargetScaler,),
                                               user_defined=False, dataset_property=False))
        return fit_requirements

    def prepare_trainer(self, X):
        # Support additional user metrics
        metrics = get_metrics(dataset_properties=X['dataset_properties'])
        if 'additional_metrics' in X:
            metrics.extend(get_metrics(dataset_properties=X['dataset_properties'], names=X['additional_metrics']))
        if 'optimize_metric' in X and X['optimize_metric'] not in [m.name for m in metrics]:
            metrics.extend(get_metrics(dataset_properties=X['dataset_properties'], names=[X['optimize_metric']]))

        additional_losses = X['additional_losses'] if 'additional_losses' in X else None

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
            labels=X['y_train'][X['backend'].load_datamanager().splits[X['split_id']][0]],
            step_interval=X['step_interval'],
            dataset_properties=X['dataset_properties'],
            target_scaler=X['target_scaler'],
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
