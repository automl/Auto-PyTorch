import tempfile
from typing import Any, Dict, List, Optional, Union

from catboost import CatBoostClassifier, Pool

from lightgbm import LGBMClassifier

import numpy as np

import pandas as pd

from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from autoPyTorch.pipeline.components.setup.traditional_ml.classifier_models.base_classifier import BaseClassifier


def encode_categoricals(X_train: np.ndarray,
                        X_val: Optional[np.ndarray] = None,
                        encode_dicts: Optional[List] = None
                        ) -> Union[np.ndarray, Optional[np.ndarray], Optional[List]]:
    if encode_dicts is None:
        encode_dicts = []
        got_encoded_dicts = False
    else:
        got_encoded_dicts = True

    for ind in range(X_train.shape[1]):
        if isinstance(X_train[0, ind], str):
            uniques = np.unique(X_train[0, :])

            if got_encoded_dicts:
                cat_to_int_dict = encode_dicts[ind]
            else:
                cat_to_int_dict = {val: ind for ind, val in enumerate(uniques)}

            converted_column_train = [cat_to_int_dict[v] for v in X_train[0, :]]
            X_train[0, :] = converted_column_train

            if X_val is not None:
                converted_column_val = [cat_to_int_dict[v] for v in X_val[0, :]]
                X_val[0, :] = converted_column_val

            if not got_encoded_dicts:
                encode_dicts.append(cat_to_int_dict)
    return X_train, X_val, encode_dicts


class LGBModel(BaseClassifier):

    def __init__(self) -> None:
        super(LGBModel, self).__init__(name="lgb")

    def fit(self, X_train: np.ndarray,
            y_train: np.ndarray,
            X_val: np.ndarray,
            y_val: np.ndarray,
            categoricals: np.ndarray = np.array(())) -> Dict[str, Any]:

        results = dict()

        self.num_classes = len(np.unique(y_train)) if len(np.unique(y_train)) != 2 else 1  # this fixes a bug
        self.config["num_class"] = self.num_classes

        early_stopping = 150 if X_train.shape[0] > 10000 else max(round(150 * 10000 / X_train.shape[0]), 10)
        self.config["early_stopping_rounds"] = early_stopping

        self.all_nan = np.all(pd.isnull(X_train), axis=0)
        X_train = X_train[:, ~self.all_nan]
        X_val = X_val[:, ~self.all_nan]

        X_train = np.nan_to_num(X_train)
        X_val = np.nan_to_num(X_val)

        self.model = LGBMClassifier(**self.config)
        self.model.fit(X_train, y_train, eval_set=[(X_val, y_val)])

        pred_train = self.model.predict_proba(X_train)
        pred_val = self.model.predict_proba(X_val)

        results["val_preds"] = pred_val.tolist()
        results["labels"] = y_val.tolist()

        pred_train = np.argmax(pred_train, axis=1)
        pred_val = np.argmax(pred_val, axis=1)

        results["train_score"] = self.metric(y_train, pred_train)
        results["val_score"] = self.metric(y_val, pred_val)

        return results

    def score(self, X_test: np.ndarray, y_test: Union[np.ndarray, List]) -> float:
        y_pred = self.predict(X_test)
        return self.metric(y_test, y_pred)

    def predict(self, X_test: np.ndarray, predict_proba: bool = False) -> np.ndarray:
        X_test = X_test[:, ~self.all_nan]
        X_test = np.nan_to_num(X_test)
        if predict_proba:
            y_pred_proba = self.model.predict_proba(X_test)
            if self.num_classes == 2:
                y_pred_proba = y_pred_proba.transpose()[0:len(X_test)]
            return y_pred_proba

        y_pred = self.model.predict(X_test)
        return y_pred

    @staticmethod
    def get_properties(dataset_properties: Optional[Dict[str, Any]] = None) -> Dict[str, str]:
        return {
            'shortname': 'LGBMClassifier',
            'name': 'LGBMClassifier',
        }


class CatboostModel(BaseClassifier):

    def __init__(self) -> None:
        super(CatboostModel, self).__init__(name="catboost")
        self.config["train_dir"] = tempfile.gettempdir()

    def fit(self, X_train: np.ndarray,
            y_train: np.ndarray,
            X_val: np.ndarray,
            y_val: np.ndarray,
            categoricals: np.ndarray = np.array(())) -> Dict[str, Any]:

        results = dict()

        categoricals = [ind for ind in range(X_train.shape[1]) if isinstance(X_train[0, ind], str)]

        self.all_nan = np.all(pd.isnull(X_train), axis=0)
        X_train = X_train[:, ~self.all_nan]
        X_val = X_val[:, ~self.all_nan]

        X_train = np.nan_to_num(X_train)
        X_val = np.nan_to_num(X_val)

        early_stopping = 150 if X_train.shape[0] > 10000 else max(round(150 * 10000 / X_train.shape[0]), 10)

        X_train_pooled = Pool(data=X_train, label=y_train, cat_features=categoricals)
        X_val_pooled = Pool(data=X_val, label=y_val, cat_features=categoricals)

        self.model = CatBoostClassifier(**self.config)
        self.model.fit(X_train_pooled, eval_set=X_val_pooled, use_best_model=True, early_stopping_rounds=early_stopping)

        pred_train = self.model.predict_proba(X_train)
        pred_val = self.model.predict_proba(X_val)

        results["val_preds"] = pred_val.tolist()
        results["labels"] = y_val.tolist()

        try:
            pred_train = np.argmax(pred_train, axis=1)
            pred_val = np.argmax(pred_val, axis=1)
        except ValueError:
            self.logger.info("==> No probabilities provided in predictions")

        results["train_score"] = self.metric(y_train, pred_train)
        results["val_score"] = self.metric(y_val, pred_val)

        return results

    def score(self, X_test: np.ndarray, y_test: Union[np.ndarray, List]) -> float:
        y_pred = self.predict(X_test)
        return self.metric(y_test, y_pred)

    def predict(self, X_test: np.ndarray, predict_proba: bool = False) -> np.ndarray:
        X_test = X_test[:, ~self.all_nan]
        X_test = np.nan_to_num(X_test)
        if predict_proba:
            return self.model.predict_proba(X_test)
        y_pred = self.model.predict(X_test)
        return y_pred

    @staticmethod
    def get_properties(dataset_properties: Optional[Dict[str, Any]] = None) -> Dict[str, str]:
        return {
            'shortname': 'CatBoostClassifier',
            'name': 'CatBoostClassifier',
        }


class RFModel(BaseClassifier):

    def __init__(self) -> None:
        super(RFModel, self).__init__(name="random_forest")

    def fit(self, X_train: np.ndarray,
            y_train: np.ndarray,
            X_val: np.ndarray,
            y_val: np.ndarray) -> Dict[str, Any]:

        results = dict()

        self.all_nan = np.all(pd.isnull(X_train), axis=0)
        X_train = X_train[:, ~self.all_nan]
        X_val = X_val[:, ~self.all_nan]

        X_train = np.nan_to_num(X_train)
        X_val = np.nan_to_num(X_val)

        self.config["warm_start"] = False
        self.num_classes = len(np.unique(y_train))
        if self.num_classes > 2:
            self.logger.info("==> Using warmstarting for multiclass")
            final_n_estimators = self.config["n_estimators"]
            self.config["n_estimators"] = 8
            self.config["warm_start"] = True

        self.model = RandomForestClassifier(**self.config)

        self.model.fit(X_train, y_train)
        if self.config["warm_start"]:
            self.model.n_estimators = final_n_estimators
            self.model.fit(X_train, y_train)

        pred_val_probas = self.model.predict_proba(X_val)

        pred_train = self.model.predict(X_train)
        pred_val = self.model.predict(X_val)

        results["train_score"] = self.metric(y_train, pred_train)
        results["val_score"] = self.metric(y_val, pred_val)
        results["val_preds"] = pred_val_probas.tolist()
        results["labels"] = y_val.tolist()

        return results

    def score(self, X_test: np.ndarray, y_test: Union[np.ndarray, List]) -> float:
        y_pred = self.predict(X_test)
        return self.metric(y_test, y_pred)

    def predict(self, X_test: np.ndarray, predict_proba: bool = False) -> np.ndarray:
        X_test = X_test[:, ~self.all_nan]
        X_test = np.nan_to_num(X_test)
        if predict_proba:
            return self.model.predict_proba(X_test)
        y_pred = self.model.predict(X_test)
        return y_pred

    @staticmethod
    def get_properties(dataset_properties: Optional[Dict[str, Any]] = None) -> Dict[str, str]:
        return {
            'shortname': 'RandomForestClassifier',
            'name': 'RandomForestClassifier',
        }


class ExtraTreesModel(BaseClassifier):

    def __init__(self) -> None:
        super(ExtraTreesModel, self).__init__(name="extra_trees")

    def fit(self, X_train: np.ndarray,
            y_train: np.ndarray,
            X_val: np.ndarray,
            y_val: np.ndarray) -> Dict[str, Any]:

        results = dict()

        self.all_nan = np.all(pd.isnull(X_train), axis=0)
        X_train = X_train[:, ~self.all_nan]
        X_val = X_val[:, ~self.all_nan]

        X_train = np.nan_to_num(X_train)
        X_val = np.nan_to_num(X_val)

        self.config["warm_start"] = False
        self.num_classes = len(np.unique(y_train))
        if self.num_classes > 2:
            self.logger.info("==> Using warmstarting for multiclass")
            final_n_estimators = self.config["n_estimators"]
            self.config["n_estimators"] = 8
            self.config["warm_start"] = True

        self.model = ExtraTreesClassifier(**self.config)

        self.model.fit(X_train, y_train)
        if self.config["warm_start"]:
            self.model.n_estimators = final_n_estimators
            self.model.fit(X_train, y_train)

        pred_val_probas = self.model.predict_proba(X_val)

        pred_train = self.model.predict(X_train)
        pred_val = self.model.predict(X_val)

        results["train_score"] = self.metric(y_train, pred_train)
        results["val_score"] = self.metric(y_val, pred_val)
        results["val_preds"] = pred_val_probas.tolist()
        results["labels"] = y_val.tolist()

        return results

    def score(self, X_test: np.ndarray, y_test: Union[np.ndarray, List]) -> float:
        y_pred = self.predict(X_test)
        return self.metric(y_test, y_pred)

    def predict(self, X_test: np.ndarray, predict_proba: bool = False) -> np.ndarray:
        X_test = X_test[:, ~self.all_nan]
        X_test = np.nan_to_num(X_test)
        if predict_proba:
            return self.model.predict_proba(X_test)
        y_pred = self.model.predict(X_test)
        return y_pred

    @staticmethod
    def get_properties(dataset_properties: Optional[Dict[str, Any]] = None) -> Dict[str, str]:
        return {
            'shortname': 'ExtraTreesClassifier',
            'name': 'ExtraTreesClassifier',
        }


class KNNModel(BaseClassifier):

    def __init__(self) -> None:
        super(KNNModel, self).__init__(name="knn")

    def fit(self, X_train: np.ndarray,
            y_train: np.ndarray,
            X_val: np.ndarray,
            y_val: np.ndarray) -> Dict[str, Any]:
        results = dict()

        self.all_nan = np.all(pd.isnull(X_train), axis=0)
        X_train = X_train[:, ~self.all_nan]
        X_val = X_val[:, ~self.all_nan]

        X_train = np.nan_to_num(X_train)
        X_val = np.nan_to_num(X_val)

        self.categoricals = np.array([isinstance(X_train[0, ind], str) for ind in range(X_train.shape[1])])
        X_train = X_train[:, ~self.categoricals] if self.categoricals is not None else X_train
        X_val = X_val[:, ~self.categoricals] if self.categoricals is not None else X_val

        self.num_classes = len(np.unique(y_train))

        self.model = KNeighborsClassifier(**self.config)
        self.model.fit(X_train, y_train)

        pred_val_probas = self.model.predict_proba(X_val)

        pred_train = self.model.predict(X_train)
        pred_val = self.model.predict(X_val)

        results["train_score"] = self.metric(y_train, pred_train)
        results["val_score"] = self.metric(y_val, pred_val)
        results["val_preds"] = pred_val_probas.tolist()
        results["labels"] = y_val.tolist()

        return results

    def score(self, X_test: np.ndarray, y_test: Union[np.ndarray, List]) -> float:
        y_pred = self.predict(X_test)
        return self.metric(y_test, y_pred)

    def predict(self, X_test: np.ndarray, predict_proba: bool = False) -> np.ndarray:
        X_test = X_test[:, ~self.all_nan]
        X_test = np.nan_to_num(X_test)
        X_test = X_test[:, ~self.categoricals] if self.categoricals is not None else X_test
        if predict_proba:
            return self.model.predict_proba(X_test)
        y_pred = self.model.predict(X_test)
        return y_pred

    @staticmethod
    def get_properties(dataset_properties: Optional[Dict[str, Any]] = None) -> Dict[str, str]:
        return {
            'shortname': 'KNeighborsClassifier',
            'name': 'KNeighborsClassifier',
        }


class SVMModel(BaseClassifier):

    def __init__(self) -> None:
        super(SVMModel, self).__init__(name="svm")

    def fit(self, X_train: np.ndarray,
            y_train: np.ndarray,
            X_val: np.ndarray,
            y_val: np.ndarray) -> Dict[str, Any]:
        results = dict()

        self.all_nan = np.all(pd.isnull(X_train), axis=0)
        X_train = X_train[:, ~self.all_nan]
        X_val = X_val[:, ~self.all_nan]

        X_train = np.nan_to_num(X_train)
        X_val = np.nan_to_num(X_val)

        self.model = SVC(**self.config, probability=True)

        self.model.fit(X_train, y_train)

        pred_val_probas = self.model.predict_proba(X_val)

        pred_train = self.model.predict(X_train)
        pred_val = self.model.predict(X_val)

        results["train_score"] = self.metric(y_train, pred_train)
        results["val_score"] = self.metric(y_val, pred_val)
        results["val_preds"] = pred_val_probas.tolist()
        results["labels"] = y_val.tolist()

        return results

    def score(self, X_test: np.ndarray, y_test: Union[np.ndarray, List]) -> float:
        y_pred = self.predict(X_test)
        return self.metric(y_test, y_pred)

    def predict(self, X_test: np.ndarray, predict_proba: bool = False) -> np.ndarray:
        X_test = X_test[:, ~self.all_nan]
        X_test = np.nan_to_num(X_test)
        if predict_proba:
            return self.model.predict_proba(X_test)
        y_pred = self.model.predict(X_test)
        return y_pred

    @staticmethod
    def get_properties(dataset_properties: Optional[Dict[str, Any]] = None) -> Dict[str, str]:
        return {
            'shortname': 'SVC',
            'name': 'SVC',
        }
