from enum import IntEnum

from autoPyTorch.ensemble.ensemble_builder import EnsembleBuilder
from autoPyTorch.ensemble.stacking_ensemble_builder import StackingEnsembleBuilder

class EnsembleSelectionTypes(IntEnum):
    ensemble_selection = 1
    stacking_ensemble = 2


def get_ensemble_builder_class(ensemble_method: int):
    if ensemble_method == EnsembleSelectionTypes.ensemble_selection:
        return EnsembleBuilder
    elif ensemble_method == EnsembleSelectionTypes.stacking_ensemble:
        return StackingEnsembleBuilder