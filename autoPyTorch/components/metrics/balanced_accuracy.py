import numpy as np
import scipy as sp

from sklearn.metrics.classification import _check_targets, type_of_target


def balanced_accuracy(y_pred, y_true):
    return _balanced_accuracy(np.argmax(y_true, axis=1), np.argmax(y_pred, axis=1)) * 100


def _balanced_accuracy(solution, prediction):
    y_type, solution, prediction = _check_targets(solution, prediction)

    if y_type not in ["binary", "multiclass", 'multilabel-indicator']:
        raise ValueError("{0} is not supported".format(y_type))

    if y_type == 'binary':
        # Do not transform into any multiclass representation
        max_value = max(np.max(solution), np.max(prediction))
        min_value = min(np.min(solution), np.min(prediction))
        if max_value == min_value:
            return 1.0
        solution = (solution - min_value) / (max_value - min_value)
        prediction = (prediction - min_value) / (max_value - min_value)

    elif y_type == 'multiclass':
        # Need to create a multiclass solution and a multiclass predictions
        max_class = int(np.max((np.max(solution), np.max(prediction))))
        solution_binary = np.zeros((len(solution), max_class + 1))
        prediction_binary = np.zeros((len(prediction), max_class + 1))
        for i in range(len(solution)):
            solution_binary[i, int(solution[i])] = 1
            prediction_binary[i, int(prediction[i])] = 1
        solution = solution_binary
        prediction = prediction_binary

    elif y_type == 'multilabel-indicator':
        solution = solution.toarray()
        prediction = prediction.toarray()
    else:
        raise NotImplementedError('bac_metric does not support task type %s'
                                  % y_type)

    fn = np.sum(np.multiply(solution, (1 - prediction)), axis=0,
                dtype=float)
    tp = np.sum(np.multiply(solution, prediction), axis=0, dtype=float)
    # Bounding to avoid division by 0
    eps = 1e-15
    tp = sp.maximum(eps, tp)
    pos_num = sp.maximum(eps, tp + fn)
    tpr = tp / pos_num  # true positive rate (sensitivity)

    if y_type in ('binary', 'multilabel-indicator'):
        tn = np.sum(np.multiply((1 - solution), (1 - prediction)),
                    axis=0, dtype=float)
        fp = np.sum(np.multiply((1 - solution), prediction), axis=0,
                    dtype=float)
        tn = sp.maximum(eps, tn)
        neg_num = sp.maximum(eps, tn + fp)
        tnr = tn / neg_num  # true negative rate (specificity)
        bac = 0.5 * (tpr + tnr)
    elif y_type == 'multiclass':
        label_num = solution.shape[1]
        bac = tpr
    else:
        raise ValueError(y_type)

    return np.mean(bac)  # average over all classes