import logging.handlers
import tempfile
from typing import Dict, Optional, Union

from catboost import CatBoostClassifier, CatBoostRegressor, Pool

from lightgbm import LGBMClassifier, LGBMRegressor

import numpy as np

from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    RandomForestClassifier,
    RandomForestRegressor
)
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR

from autoPyTorch.pipeline.base_pipeline import BaseDatasetPropertiesType
from autoPyTorch.pipeline.components.setup.traditional_ml.traditional_learner.base_traditional_learner import \
    BaseTraditionalLearner
from autoPyTorch.pipeline.components.setup.traditional_ml.traditional_learner.utils import (
    AutoPyTorchToCatboostMetrics
)


class LGBModel(BaseTraditionalLearner):

    def __init__(self,
                 task_type: str,
                 output_type: str,
                 optimize_metric: Optional[str] = None,
                 logger_port: int = logging.handlers.DEFAULT_TCP_LOGGING_PORT,
                 random_state: Optional[np.random.RandomState] = None
                 ):
        super(LGBModel, self).__init__(name="lgb",
                                       logger_port=logger_port,
                                       random_state=random_state,
                                       task_type=task_type,
                                       output_type=output_type,
                                       optimize_metric=optimize_metric)

    def _prepare_model(self,
                       X_train: np.ndarray,
                       y_train: np.ndarray
                       ) -> None:

        if self.has_val_set:
            early_stopping = 150 if X_train.shape[0] > 10000 else max(round(150 * 10000 / X_train.shape[0]), 10)
            self.config["early_stopping_rounds"] = early_stopping
        if not self.is_classification:
            self.model = LGBMRegressor(**self.config, random_state=self.random_state)
        else:
            self.num_classes = len(np.unique(y_train)) if len(np.unique(y_train)) != 2 else 1  # this fixes a bug
            self.config["num_class"] = self.num_classes

            self.model = LGBMClassifier(**self.config, random_state=self.random_state)

    def _fit(self, X_train: np.ndarray,
             y_train: np.ndarray,
             X_val: Optional[np.ndarray] = None,
             y_val: Optional[np.ndarray] = None
             ) -> None:
        assert self.model is not None, "No model found. Can't fit without preparing the model"
        eval_set = None
        if self.has_val_set:
            eval_set = [(X_val, y_val)]
        self.model.fit(X_train, y_train, eval_set=eval_set)

    def predict(self, X_test: np.ndarray,
                predict_proba: bool = False,
                preprocess: bool = True) -> np.ndarray:
        assert self.model is not None, "No model found. Can't " \
                                       "predict before fitting. " \
                                       "Call fit before predicting"
        if preprocess:
            X_test = self._preprocess(X_test)

        if predict_proba:
            if not self.is_classification:
                raise ValueError("Can't predict probabilities for a regressor")
            y_pred_proba = self.model.predict_proba(X_test)
            if self.num_classes == 2:
                y_pred_proba = y_pred_proba.transpose()[0:len(X_test)]
            return y_pred_proba

        y_pred = self.model.predict(X_test)
        return y_pred

    @staticmethod
    def get_properties(
        dataset_properties: Optional[Dict[str, BaseDatasetPropertiesType]] = None
    ) -> Dict[str, Union[str, bool]]:
        return {
            'shortname': 'LGBMLearner',
            'name': 'Light Gradient Boosting Machine Learner',
        }


class CatboostModel(BaseTraditionalLearner):

    def __init__(self,
                 task_type: str,
                 output_type: str,
                 optimize_metric: Optional[str] = None,
                 logger_port: int = logging.handlers.DEFAULT_TCP_LOGGING_PORT,
                 random_state: Optional[np.random.RandomState] = None
                 ):
        super(CatboostModel, self).__init__(name="catboost",
                                            logger_port=logger_port,
                                            random_state=random_state,
                                            task_type=task_type,
                                            output_type=output_type,
                                            optimize_metric=optimize_metric)
        self.config["train_dir"] = tempfile.gettempdir()

    def _prepare_model(self,
                       X_train: np.ndarray,
                       y_train: np.ndarray
                       ) -> None:
        if not self.is_classification:
            self.config['eval_metric'] = AutoPyTorchToCatboostMetrics[self.metric.name].value
            # CatBoost Cannot handle a random state object, just the seed
            self.model = CatBoostRegressor(**self.config, random_state=self.random_state.get_state()[1][0])
        else:
            self.config['eval_metric'] = AutoPyTorchToCatboostMetrics[self.metric.name].value
            # CatBoost Cannot handle a random state object, just the seed
            self.model = CatBoostClassifier(**self.config, random_state=self.random_state.get_state()[1][0])

    def _fit(self, X_train: np.ndarray,
             y_train: np.ndarray,
             X_val: Optional[np.ndarray] = None,
             y_val: Optional[np.ndarray] = None
             ) -> None:

        assert self.model is not None, "No model found. Can't fit without preparing the model"
        categoricals = [ind for ind in range(X_train.shape[1]) if isinstance(X_train[0, ind], str)]

        X_train_pooled = Pool(data=X_train, label=y_train, cat_features=categoricals)
        X_val_pooled = None
        if self.has_val_set:
            X_val_pooled = Pool(data=X_val, label=y_val, cat_features=categoricals)
            early_stopping: Optional[int] = 150 if X_train.shape[0] > 10000 else max(
                round(150 * 10000 / X_train.shape[0]), 10)
        else:
            early_stopping = None

        self.model.fit(X_train_pooled,
                       eval_set=X_val_pooled,
                       use_best_model=True,
                       early_stopping_rounds=early_stopping,
                       verbose=False)

    @staticmethod
    def get_properties(
        dataset_properties: Optional[Dict[str, BaseDatasetPropertiesType]] = None
    ) -> Dict[str, Union[str, bool]]:
        return {
            'shortname': 'CBLearner',
            'name': 'Categorical Boosting Learner',
        }


class RFModel(BaseTraditionalLearner):

    def __init__(self,
                 task_type: str,
                 output_type: str,
                 optimize_metric: Optional[str] = None,
                 logger_port: int = logging.handlers.DEFAULT_TCP_LOGGING_PORT,
                 random_state: Optional[np.random.RandomState] = None
                 ):
        super(RFModel, self).__init__(name="random_forest",
                                      logger_port=logger_port,
                                      random_state=random_state,
                                      task_type=task_type,
                                      output_type=output_type,
                                      optimize_metric=optimize_metric)

    def _prepare_model(self,
                       X_train: np.ndarray,
                       y_train: np.ndarray
                       ) -> None:

        self.config["warm_start"] = False
        # TODO: Check if we need to warmstart for regression.
        #  In autogluon, they warm start when usinf daal backend, see
        #  ('https://github.com/awslabs/autogluon/blob/master/tabular/src/autogluon/tabular/models/rf/rf_model.py#L35')
        if not self.is_classification:
            self.model = RandomForestRegressor(**self.config, random_state=self.random_state)
        else:
            self.num_classes = len(np.unique(y_train))
            if self.num_classes > 2:
                self.logger.info("==> Using warmstarting for multiclass")
                self.final_n_estimators = self.config["n_estimators"]
                self.config["n_estimators"] = 8
                self.config["warm_start"] = True
            self.model = RandomForestClassifier(**self.config, random_state=self.random_state)

    def _fit(self, X_train: np.ndarray,
             y_train: np.ndarray,
             X_val: Optional[np.ndarray] = None,
             y_val: Optional[np.ndarray] = None
             ) -> None:
        assert self.model is not None, "No model found. Can't fit without preparing the model"

        self.model.fit(X_train, y_train)
        if self.config["warm_start"]:
            self.model.n_estimators = self.final_n_estimators
            self.model.fit(X_train, y_train)

    @staticmethod
    def get_properties(
        dataset_properties: Optional[Dict[str, BaseDatasetPropertiesType]] = None
    ) -> Dict[str, Union[str, bool]]:
        return {
            'shortname': 'RFLearner',
            'name': 'Random Forest Learner',
        }


class ExtraTreesModel(BaseTraditionalLearner):

    def __init__(self,
                 task_type: str,
                 output_type: str,
                 optimize_metric: Optional[str] = None,
                 logger_port: int = logging.handlers.DEFAULT_TCP_LOGGING_PORT,
                 random_state: Optional[np.random.RandomState] = None
                 ):
        super(ExtraTreesModel, self).__init__(name="extra_trees",
                                              logger_port=logger_port,
                                              random_state=random_state,
                                              task_type=task_type,
                                              output_type=output_type,
                                              optimize_metric=optimize_metric)

    def _prepare_model(self,
                       X_train: np.ndarray,
                       y_train: np.ndarray
                       ) -> None:
        self.config["warm_start"] = False

        if not self.is_classification:
            self.model = ExtraTreesRegressor(**self.config, random_state=self.random_state)
        else:
            self.num_classes = len(np.unique(y_train))
            if self.num_classes > 2:
                self.logger.info("==> Using warmstarting for multiclass")
                self.final_n_estimators = self.config["n_estimators"]
                self.config["n_estimators"] = 8
                self.config["warm_start"] = True

            self.model = ExtraTreesClassifier(**self.config, random_state=self.random_state)

    def _fit(self, X_train: np.ndarray,
             y_train: np.ndarray,
             X_val: Optional[np.ndarray] = None,
             y_val: Optional[np.ndarray] = None) -> None:
        assert self.model is not None, "No model found. Can't fit without preparing the model"
        self.model.fit(X_train, y_train)
        if self.config["warm_start"]:
            self.model.n_estimators = self.final_n_estimators
            self.model.fit(X_train, y_train)

    @staticmethod
    def get_properties(
        dataset_properties: Optional[Dict[str, BaseDatasetPropertiesType]] = None
    ) -> Dict[str, Union[str, bool]]:
        return {
            'shortname': 'ETLearner',
            'name': 'ExtraTreesLearner',
        }


class KNNModel(BaseTraditionalLearner):

    def __init__(self,
                 task_type: str,
                 output_type: str,
                 optimize_metric: Optional[str] = None,
                 logger_port: int = logging.handlers.DEFAULT_TCP_LOGGING_PORT,
                 random_state: Optional[np.random.RandomState] = None
                 ):
        super(KNNModel, self).__init__(name="knn",
                                       logger_port=logger_port,
                                       random_state=random_state,
                                       task_type=task_type,
                                       output_type=output_type,
                                       optimize_metric=optimize_metric)
        self.categoricals: Optional[np.ndarray[bool]] = None

    def _preprocess(self,
                    X: np.ndarray
                    ) -> np.ndarray:

        super(KNNModel, self)._preprocess(X)
        if self.categoricals is None:
            self.categoricals = np.array([isinstance(X[0, ind], str) for ind in range(X.shape[1])])
        X = X[:, ~self.categoricals] if self.categoricals is not None else X

        return X

    def _prepare_model(self,
                       X_train: np.ndarray,
                       y_train: np.ndarray
                       ) -> None:
        if not self.is_classification:
            self.model = KNeighborsRegressor(**self.config)
        else:
            self.num_classes = len(np.unique(y_train))
            # KNN is deterministic, no random seed needed
            self.model = KNeighborsClassifier(**self.config)

    def _fit(self, X_train: np.ndarray,
             y_train: np.ndarray,
             X_val: Optional[np.ndarray] = None,
             y_val: Optional[np.ndarray] = None) -> None:
        assert self.model is not None, "No model found. Can't fit without preparing the model"
        self.model.fit(X_train, y_train)

    @staticmethod
    def get_properties(
        dataset_properties: Optional[Dict[str, BaseDatasetPropertiesType]] = None
    ) -> Dict[str, Union[str, bool]]:
        return {
            'shortname': 'KNNLearner',
            'name': 'K Nearest Neighbors Learner',
        }


class SVMModel(BaseTraditionalLearner):

    def __init__(self,
                 task_type: str,
                 output_type: str,
                 optimize_metric: Optional[str] = None,
                 logger_port: int = logging.handlers.DEFAULT_TCP_LOGGING_PORT,
                 random_state: Optional[np.random.RandomState] = None
                 ):
        super(SVMModel, self).__init__(name="svm",
                                       logger_port=logger_port,
                                       random_state=random_state,
                                       task_type=task_type,
                                       output_type=output_type,
                                       optimize_metric=optimize_metric)

    def _prepare_model(self,
                       X_train: np.ndarray,
                       y_train: np.ndarray
                       ) -> None:
        if not self.is_classification:
            # Does not take random state.
            self.model = SVR(**self.config)
        else:
            self.model = SVC(**self.config, probability=True, random_state=self.random_state)

    def _fit(self, X_train: np.ndarray,
             y_train: np.ndarray,
             X_val: Optional[np.ndarray] = None,
             y_val: Optional[np.ndarray] = None) -> None:
        assert self.model is not None, "No model found. Can't fit without preparing the model"
        self.model.fit(X_train, y_train)

    @staticmethod
    def get_properties(
        dataset_properties: Optional[Dict[str, BaseDatasetPropertiesType]] = None
    ) -> Dict[str, Union[str, bool]]:
        return {
            'shortname': 'SVMLearner',
            'name': 'Support Vector Machine Learner',
        }
