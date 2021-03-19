from typing import Any, Dict, Type, Union

from autoPyTorch.pipeline.components.base_component import (
    ThirdPartyComponents,
)
from autoPyTorch.pipeline.components.setup.traditional_ml.classifier_models.base_classifier import BaseClassifier
from autoPyTorch.pipeline.components.setup.traditional_ml.classifier_models.classifiers import (
    CatboostModel,
    ExtraTreesModel,
    KNNModel,
    LGBModel,
    RFModel,
    SVMModel)

_classifiers = {
    # Sort by more robust models
    # Depending on the allocated time budget, only the
    # top models from this dict are two be fitted.
    # LGBM is the more robust model, with
    # internal measures to prevent crashes, overfit
    # Additionally, it is one of the state of the art
    # methods for tabular prediction.
    # Then follow with catboost for categorical heavy
    # datasets. The other models are complementary and
    # their ordering is not critical
    'lgb': LGBModel,
    'catboost': CatboostModel,
    'random_forest': RFModel,
    'extra_trees': ExtraTreesModel,
    'svm_classifier': SVMModel,
    'knn_classifier': KNNModel,
}
_addons = ThirdPartyComponents(BaseClassifier)


def add_classifier(classifier: BaseClassifier) -> None:
    _addons.add_component(classifier)


def get_available_classifiers() -> Dict[str, Union[Type[BaseClassifier], Any]]:
    classifiers = dict()
    classifiers.update(_classifiers)
    return classifiers
