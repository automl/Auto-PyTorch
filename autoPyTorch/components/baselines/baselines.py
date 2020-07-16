import numpy as np

import pickle
import logging
from sklearn import metrics
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from lightgbm import LGBMClassifier
from catboost import Pool, CatBoostClassifier

from autoPyTorch.components.baselines.rotation_forest import RotationForestClassifier
from autoPyTorch.components.baselines.base_baseline import BaseBaseline


def encode_categoricals(X_train, X_val=None, encode_dicts=None):
    
    if encode_dicts is None:
        encode_dicts = []
        got_encoded_dicts = False
    else:
        got_encoded_dicts = True

    for ind in range(X_train.shape[1]):
        if isinstance(X_train[0, ind], str):
            uniques = np.unique(X_train[0,:])

            if got_encoded_dicts:
                cat_to_int_dict = encode_dicts[ind]
            else:
                cat_to_int_dict = {val:ind for ind,val in enumerate(uniques)}

            converted_column_train = [cat_to_int_dict[v] for v in X_train[0,:]]
            x_train[0,:] = converted_column

            if X_val is not None:
                converted_column_val = [cat_to_int_dict[v] for v in X_val[0,:]]
                x_val[0,:] = converted_column_val

            if not got_encoded_dicts:
                encode_dicts.append(cat_to_int_dict)
    return X_train, X_val, encode_dicts


class LGBBaseline(BaseBaseline):
    
    def __init__(self):
        super(LGBBaseline, self).__init__(name="lgb")

    def fit(self, X_train, y_train, X_val, y_val, categoricals=None):
        results = dict()

        self.num_classes = len(np.unique(y_train))
        self.config["num_class"] = self.num_classes

        self.all_nan = np.all(np.isnan(X_train), axis=0)
        X_train = X_train[:, ~self.all_nan]
        X_val = X_val[:, ~self.all_nan]

        X_train = np.nan_to_num(X_train)
        X_val = np.nan_to_num(X_val)

        early_stopping = 150 if X_train.shape[0]>10000 else max(round(150*10000/X_train.shape[0]), 10)
        self.config["early_stopping_rounds"] = early_stopping

        categoricals = [ind for ind in range(X_train.shape[1]) if isinstance(X_train[0,ind], str)]
        X_train, X_val, self.encode_dicts = encode_categoricals(X_train, X_val, encode_dicts=None)

        self.model = LGBMClassifier(**self.config)
        self.model.fit(X_train, y_train, eval_set=[(X_val, y_val)])

        pred_train = self.model.predict_proba(X_train)
        pred_val = self.model.predict_proba(X_val)

        # This fixes a bug
        if self.num_classes==2:
            pred_train = pred_train.transpose()[0:len(y_train)]
            pred_val = pred_val.transpose()[0:len(y_val)]

        results["val_preds"] = pred_val.tolist()
        results["labels"] = y_val.tolist()

        pred_train = np.argmax(pred_train, axis=1)
        pred_val = np.argmax(pred_val, axis=1)

        results["train_acc"] = metrics.accuracy_score(y_train, pred_train)
        results["train_balanced_acc"] = metrics.balanced_accuracy_score(y_train, pred_train)
        results["val_acc"] = metrics.accuracy_score(y_val, pred_val)
        results["val_balanced_acc"] = metrics.balanced_accuracy_score(y_val, pred_val)
        
        return results

    def refit(self, X_train, y_train):
        # neither self.model.best_iteration_ nor self.model._best_iteration seems to work for lgb
        loss_key = 'multi_logloss' if 'multi_logloss' in self.model.evals_result_['valid_0'].keys() else list(self.model.evals_result_['valid_0'].keys())[0]
        best_iter = int(np.where(self.model.evals_result_['valid_0'][loss_key]==self.model._best_score["valid_0"][loss_key])[0]+1)
        self.config["num_rounds"] = best_iter
        logging.info("==> Refitting with %i iterations" %best_iter)
        
        results = dict()

        self.num_classes = len(np.unique(y_train))
        self.config["num_class"] = self.num_classes

        self.all_nan = np.all(np.isnan(X_train), axis=0)
        X_train = X_train[:, ~self.all_nan]

        X_train = np.nan_to_num(X_train)

        del self.config["early_stopping_rounds"]

        categoricals = [ind for ind in range(X_train.shape[1]) if isinstance(X_train[0,ind], str)]
        X_train, X_val, self.encode_dicts = encode_categoricals(X_train, X_val=None, encode_dicts=None)

        self.model = LGBMClassifier(**self.config)
        self.model.fit(X_train, y_train)

        pred_train = self.model.predict_proba(X_train)

        # This fixes a bug
        if self.num_classes==2:
            pred_train = pred_train.transpose()[0:len(y_train)]

        pred_train = np.argmax(pred_train, axis=1)
        results["train_acc"] = metrics.accuracy_score(y_train, pred_train)
        results["train_balanced_acc"] = metrics.balanced_accuracy_score(y_train, pred_train)
        return results

    def score(self, X_test, y_test):
        results = dict()

        y_pred = self.predict(X_test)

        results["test_acc"] = metrics.accuracy_score(y_test, y_pred)
        results["test_balanced_acc"] = metrics.balanced_accuracy_score(y_test, y_pred)
        
        return results

    def predict(self, X_test, predict_proba=False):
        X_test = X_test[:, ~self.all_nan]
        X_test = np.nan_to_num(X_test)
        X_test, _, _ = encode_categoricals(X_test, encode_dicts=self.encode_dicts)
        
        if predict_proba:
            y_pred_proba = self.model.predict_proba(X_test)
            if self.num_classes==2:
                y_pred_proba = y_pred_proba.transpose()[0:len(X_test)]
            return y_pred_proba
        
        y_pred = self.model.predict(X_test)
        if self.num_classes==2:
            y_pred = y_pred.transpose()[0:len(X_test)]
        y_pred = np.argmax(y_pred, axis=1)
        return y_pred


class CatboostBaseline(BaseBaseline):

    def __init__(self):
        super(CatboostBaseline, self).__init__(name="catboost")

    def fit(self, X_train, y_train, X_val, y_val, categoricals=None):
        results = dict()

        self.all_nan = np.all(np.isnan(X_train), axis=0)
        X_train = X_train[:, ~self.all_nan]
        X_val = X_val[:, ~self.all_nan]

        X_train = np.nan_to_num(X_train)
        X_val = np.nan_to_num(X_val)

        categoricals = [ind for ind in range(X_train.shape[1]) if isinstance(X_train[0,ind], str)]

        early_stopping = 150 if X_train.shape[0]>10000 else max(round(150*10000/X_train.shape[0]), 10)

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
        except:
            print("==> No probabilities provided in predictions")

        results["train_acc"] = metrics.accuracy_score(y_train, pred_train)
        results["train_balanced_acc"] = metrics.balanced_accuracy_score(y_train, pred_train)
        results["val_acc"] = metrics.accuracy_score(y_val, pred_val)
        results["val_balanced_acc"] = metrics.balanced_accuracy_score(y_val, pred_val)

        return results

    def refit(self, X_train, y_train):
        best_iter = self.model.best_iteration_ + 1 # appearently 0 based
        self.config["iterations"] = best_iter
        logging.info("==> Refitting with %i iterations" %best_iter)
        
        results = dict()

        self.all_nan = np.all(np.isnan(X_train), axis=0)
        X_train = X_train[:, ~self.all_nan]
        X_train = np.nan_to_num(X_train)

        categoricals = [ind for ind in range(X_train.shape[1]) if isinstance(X_train[0,ind], str)]

        early_stopping = 0

        X_train_pooled = Pool(data=X_train, label=y_train, cat_features=categoricals)

        self.model = CatBoostClassifier(**self.config)
        self.model.fit(X_train_pooled, use_best_model=False)

        pred_train = self.model.predict_proba(X_train)

        try:
            pred_train = np.argmax(pred_train, axis=1)
        except:
            print("==> No probabilities provided in predictions")

        results["train_acc"] = metrics.accuracy_score(y_train, pred_train)
        results["train_balanced_acc"] = metrics.balanced_accuracy_score(y_train, pred_train)
        return results

    def score(self, X_test, y_test):
        results = dict()

        y_pred = self.predict(X_test)

        results["test_acc"] = metrics.accuracy_score(y_test, y_pred)
        results["test_balanced_acc"] = metrics.balanced_accuracy_score(y_test, y_pred)

        return results

    def predict(self, X_test, predict_proba=False):
        X_test = X_test[:, ~self.all_nan]
        X_test = np.nan_to_num(X_test)
        if predict_proba:
            return self.model.predict_proba(X_test)
        y_pred = self.model.predict(X_test)
        return y_pred


class RFBaseline(BaseBaseline):
    
    def __init__(self):
        super(RFBaseline, self).__init__(name="random_forest")
        
    def fit(self, X_train, y_train, X_val, y_val):
        results = dict()

        self.all_nan = np.all(np.isnan(X_train), axis=0)
        X_train = X_train[:, ~self.all_nan]
        X_val = X_val[:, ~self.all_nan]

        X_train = np.nan_to_num(X_train)
        X_val = np.nan_to_num(X_val)

        self.config["warm_start"] = False
        self.num_classes = len(np.unique(y_train))
        if self.num_classes>2:
            print("==> Using warmstarting for multiclass")
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

        results["train_acc"] = metrics.accuracy_score(y_train, pred_train)
        results["train_balanced_acc"] = metrics.balanced_accuracy_score(y_train, pred_train)
        results["val_acc"] = metrics.accuracy_score(y_val, pred_val)
        results["val_balanced_acc"] = metrics.balanced_accuracy_score(y_val, pred_val)
        results["val_preds"] = pred_val_probas.tolist()
        results["labels"] = y_val.tolist()

        return results

    def refit(self, X_train, y_train):
        results = dict()

        self.all_nan = np.all(np.isnan(X_train), axis=0)
        X_train = X_train[:, ~self.all_nan]

        X_train = np.nan_to_num(X_train)

        self.config["warm_start"] = False
        self.num_classes = len(np.unique(y_train))
        if self.num_classes>2:
            print("==> Using warmstarting for multiclass")
            final_n_estimators = self.config["n_estimators"]
            self.config["n_estimators"] = 8
            self.config["warm_start"] = True

        self.model = RandomForestClassifier(**self.config)

        self.model.fit(X_train, y_train)
        if self.config["warm_start"]:
            self.model.n_estimators = final_n_estimators
            self.model.fit(X_train, y_train)

        pred_train = self.model.predict(X_train)

        results["train_acc"] = metrics.accuracy_score(y_train, pred_train)
        results["train_balanced_acc"] = metrics.balanced_accuracy_score(y_train, pred_train)
        return results
    
    def score(self, X_test, y_test):
        results = dict()

        y_pred = self.predict(X_test)

        results["test_acc"] = metrics.accuracy_score(y_test, y_pred)
        results["test_balanced_acc"] = metrics.balanced_accuracy_score(y_test, y_pred)
        
        return results

    def predict(self, X_test, predict_proba=False):
        X_test = X_test[:, ~self.all_nan]
        X_test = np.nan_to_num(X_test)
        if predict_proba:
            return self.model.predict_proba(X_test)
        y_pred = self.model.predict(X_test)
        return y_pred


class ExtraTreesBaseline(BaseBaseline):

    def __init__(self):
        super(ExtraTreesBaseline, self).__init__(name="extra_trees")

    def fit(self, X_train, y_train, X_val, y_val):
        results = dict()

        self.all_nan = np.all(np.isnan(X_train), axis=0)
        X_train = X_train[:, ~self.all_nan]
        X_val = X_val[:, ~self.all_nan]

        X_train = np.nan_to_num(X_train)
        X_val = np.nan_to_num(X_val)

        self.config["warm_start"] = False
        self.num_classes = len(np.unique(y_train))
        if self.num_classes>2:
            print("==> Using warmstarting for multiclass")
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

        results["train_acc"] = metrics.accuracy_score(y_train, pred_train)
        results["train_balanced_acc"] = metrics.balanced_accuracy_score(y_train, pred_train)
        results["val_acc"] = metrics.accuracy_score(y_val, pred_val)
        results["val_balanced_acc"] = metrics.balanced_accuracy_score(y_val, pred_val)
        results["val_preds"] = pred_val_probas.tolist()
        results["labels"] = y_val.tolist()

        return results

    def refit(self, X_train, y_train):
        results = dict()

        self.all_nan = np.all(np.isnan(X_train), axis=0)
        X_train = X_train[:, ~self.all_nan]

        X_train = np.nan_to_num(X_train)

        self.config["warm_start"] = False
        self.num_classes = len(np.unique(y_train))
        if self.num_classes>2:
            print("==> Using warmstarting for multiclass")
            final_n_estimators = self.config["n_estimators"]
            self.config["n_estimators"] = 8
            self.config["warm_start"] = True

        self.model = ExtraTreesClassifier(**self.config)

        self.model.fit(X_train, y_train)
        if self.config["warm_start"]:
            self.model.n_estimators = final_n_estimators
            self.model.fit(X_train, y_train)

        pred_train = self.model.predict(X_train)

        results["train_acc"] = metrics.accuracy_score(y_train, pred_train)
        results["train_balanced_acc"] = metrics.balanced_accuracy_score(y_train, pred_train)
        return results

    def score(self, X_test, y_test):
        results = dict()

        y_pred = self.predict(X_test)

        results["test_acc"] = metrics.accuracy_score(y_test, y_pred)
        results["test_balanced_acc"] = metrics.balanced_accuracy_score(y_test, y_pred)

        return results

    def predict(self, X_test, predict_proba=False):
        X_test = X_test[:, ~self.all_nan]
        X_test = np.nan_to_num(X_test)
        if predict_proba:
            return self.model.predict_proba(X_test)
        y_pred = self.model.predict(X_test)
        return y_pred


class RotationForestBaseline(BaseBaseline):

    def __init__(self):
        super(RotationForestBaseline, self).__init__(name="rotation_forest")

    def fit(self, X_train, y_train, X_val, y_val):
        results = dict()

        self.all_nan = np.all(np.isnan(X_train), axis=0)
        X_train = X_train[:, ~self.all_nan]
        X_val = X_val[:, ~self.all_nan]

        X_train = np.nan_to_num(X_train)
        X_val = np.nan_to_num(X_val)

        self.config["warm_start"] = False
        self.num_classes = len(np.unique(y_train))

        self.model = RotationForestClassifier(**self.config)

        self.model.fit(X_train, y_train)

        pred_val_probas = self.model.predict_proba(X_val)

        pred_train = self.model.predict(X_train)
        pred_val = self.model.predict(X_val)

        results["train_acc"] = metrics.accuracy_score(y_train, pred_train)
        results["train_balanced_acc"] = metrics.balanced_accuracy_score(y_train, pred_train)
        results["val_acc"] = metrics.accuracy_score(y_val, pred_val)
        results["val_balanced_acc"] = metrics.balanced_accuracy_score(y_val, pred_val)
        results["val_preds"] = pred_val_probas.tolist()
        results["labels"] = y_val.tolist()

        return results

    def refit(self, X_train, y_train):
        results = dict()
        
        self.all_nan = np.all(np.isnan(X_train), axis=0)
        X_train = X_train[:, ~self.all_nan]

        X_train = np.nan_to_num(X_train)

        self.config["warm_start"] = False
        self.num_classes = len(np.unique(y_train))

        self.model = RotationForestClassifier(**self.config)

        self.model.fit(X_train, y_train)

        pred_train = self.model.predict(X_train)

        results["train_acc"] = metrics.accuracy_score(y_train, pred_train)
        results["train_balanced_acc"] = metrics.balanced_accuracy_score(y_train, pred_train)
        return results

    def score(self, X_test, y_test):
        results = dict()

        y_pred = self.predict(X_test)

        results["test_acc"] = metrics.accuracy_score(y_test, y_pred)
        results["test_balanced_acc"] = metrics.balanced_accuracy_score(y_test, y_pred)

        return results

    def predict(self, X_test, predict_proba=False):
        X_test = X_test[:, ~self.all_nan]
        X_test = np.nan_to_num(X_test)
        if predict_proba:
            return self.model.predict_proba(X_test)
        y_pred = self.model.predict(X_test)
        return y_pred


class KNNBaseline(BaseBaseline):

    def __init__(self):
        super(KNNBaseline, self).__init__(name="knn")

    def fit(self, X_train, y_train, X_val, y_val):
        results = dict()

        self.all_nan = np.all(np.isnan(X_train), axis=0)
        X_train = X_train[:, ~self.all_nan]
        X_val = X_val[:, ~self.all_nan]

        self.categoricals = np.array([isinstance(X_train[0,ind], str) for ind in range(X_train.shape[1])])
        X_train = X_train[:, ~self.categoricals]
        X_val = X_val[:, ~self.categoricals]

        X_train = np.nan_to_num(X_train)
        X_val = np.nan_to_num(X_val)

        self.num_classes = len(np.unique(y_train))
        
        self.model = KNeighborsClassifier(**self.config)
        self.model.fit(X_train, y_train)

        pred_val_probas = self.model.predict_proba(X_val)

        pred_train = self.model.predict(X_train)
        pred_val = self.model.predict(X_val)

        results["train_acc"] = metrics.accuracy_score(y_train, pred_train)
        results["train_balanced_acc"] = metrics.balanced_accuracy_score(y_train, pred_train)
        results["val_acc"] = metrics.accuracy_score(y_val, pred_val)
        results["val_balanced_acc"] = metrics.balanced_accuracy_score(y_val, pred_val)
        results["val_preds"] = pred_val_probas.tolist()
        results["labels"] = y_val.tolist()

        return results

    def refit(self, X_train, y_train):
        results = dict()

        self.all_nan = np.all(np.isnan(X_train), axis=0)
        X_train = X_train[:, ~self.all_nan]

        self.categoricals = np.array([isinstance(X_train[0,ind], str) for ind in range(X_train.shape[1])])
        X_train = X_train[:, ~self.categoricals]

        X_train = np.nan_to_num(X_train)

        self.num_classes = len(np.unique(y_train))

        self.model = KNeighborsClassifier(**self.config)
        self.model.fit(X_train, y_train)

        pred_train = self.model.predict(X_train)

        results["train_acc"] = metrics.accuracy_score(y_train, pred_train)
        results["train_balanced_acc"] = metrics.balanced_accuracy_score(y_train, pred_train)
        return results

    def score(self, X_test, y_test):
        results = dict()

        y_pred = self.predict(X_test)

        results["test_acc"] = metrics.accuracy_score(y_test, y_pred)
        results["test_balanced_acc"] = metrics.balanced_accuracy_score(y_test, y_pred)

        return results

    def predict(self, X_test, predict_proba=False):
        X_test = X_test[:, ~self.all_nan]
        X_test = X_test[:, ~self.categoricals]
        X_test = np.nan_to_num(X_test)
        if predict_proba:
            return self.model.predict_proba(X_test)
        y_pred = self.model.predict(X_test)
        return y_pred


class SVMBaseline(BaseBaseline):

    def __init__(self):
        super(SVMBaseline, self).__init__(name="svm")

    def fit(self, X_train, y_train, X_val, y_val):
        results = dict()

        self.model = SVC(**self.config)

        self.all_nan = np.all(np.isnan(X_train), axis=0)
        X_train = X_train[:, ~self.all_nan]
        X_val = X_val[:, ~self.all_nan]

        self.model.fit(X_train, y_train)

        pred_val_probas = self.model.predict_proba(X_val)

        pred_train = self.model.predict(X_train)
        pred_val = self.model.predict(X_val)

        results["train_acc"] = metrics.accuracy_score(y_train, pred_train)
        results["train_balanced_acc"] = metrics.balanced_accuracy_score(y_train, pred_train)
        results["val_acc"] = metrics.accuracy_score(y_val, pred_val)
        results["val_balanced_acc"] = metrics.balanced_accuracy_score(y_val, pred_val)
        results["val_preds"] = pred_val_probas.tolist()
        results["labels"] = y_val.tolist()

        return results

    def score(self, X_test, y_test):
        results = dict()

        y_pred = self.predict(X_test)

        results["test_acc"] = metrics.accuracy_score(y_test, y_pred)
        results["test_balanced_acc"] = metrics.balanced_accuracy_score(y_test, y_pred)

        return results

    def predict(self, X_test, predict_proba=False):
        X_test = X_test[:, ~self.all_nan]
        X_test = np.nan_to_num(X_test)
        if predict_proba:
            return self.model.predict_proba(X_test)
        y_pred = self.model.predict(X_test)
        return y_pred
