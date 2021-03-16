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
    # Sort by fit importance
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
