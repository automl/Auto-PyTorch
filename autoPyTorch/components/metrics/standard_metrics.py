import sklearn.metrics as metrics
import numpy as np

# classification metrics


def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)


def auc_metric(y_true, y_pred):
    return (2 * metrics.roc_auc_score(y_true, y_pred) - 1)


# multilabel metric
def multilabel_accuracy(y_true, y_pred):
    return np.mean(y_true == (y_pred > 0.5))

# regression metric
def mean_distance(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))
