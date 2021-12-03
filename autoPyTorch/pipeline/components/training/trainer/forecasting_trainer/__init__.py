import collections
import os

from typing import Any, Dict, List, Optional, Tuple, cast


from autoPyTorch.pipeline.components.base_component import (
    ThirdPartyComponents,
    autoPyTorchComponent,
    find_components,
)

from autoPyTorch.pipeline.components.training.trainer.forecasting_trainer.forecasting_base_trainer import (
    ForecastingBaseTrainerComponent,
)


trainer_directory = os.path.split(__file__)[0]
_trainers = find_components(__package__,
                            trainer_directory,
                            ForecastingBaseTrainerComponent)
_addons = ThirdPartyComponents(ForecastingBaseTrainerComponent)


def add_trainer(trainer: ForecastingBaseTrainerComponent) -> None:
    _addons.add_component(trainer)


from autoPyTorch.pipeline.components.training.trainer import TrainerChoice


class ForecastingTrainerChoice(TrainerChoice):
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


