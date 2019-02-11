import sklearn.metrics as metrics
import numpy as np

# classification metrics
def accuracy(y_pred, y_true):
    return np.mean((undo_ohe(y_true) == undo_ohe(y_pred))) * 100

def auc_metric(y_pred, y_true):
    return (2 * metrics.roc_auc_score(y_true, y_pred) - 1) * 100


# multilabel metric
def multilabel_accuracy(y_pred, y_true):
    return np.mean(y_true == (y_pred > 0.5)) * 100

# regression metric
def mean_distance(y_pred, y_true):
    return np.mean(np.abs(y_true - y_pred))

def undo_ohe(y):
    if len(y.shape) == 1:
        return(y)
    return np.argmax(y, axis=1)