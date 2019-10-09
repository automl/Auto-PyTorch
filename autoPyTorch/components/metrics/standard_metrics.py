import sklearn.metrics as metrics
import numpy as np

# classification metrics
def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred) * 100

def auc_metric(y_true, y_pred):
    return (2 * metrics.roc_auc_score(y_true, y_pred) - 1)

def cross_entropy(y_true, y_pred):
    if y_true==1:
        return -np.log(y_pred)
    else:
        return -np.log(1-y_pred)

def top1(y_pred, y_true):
    return topN(y_pred, y_true, 1)

def top3(y_pred, y_true):
    return topN(y_pred, y_true, 3)
    
def top5(y_pred, y_true):
    if y_pred.shape[1] < 5:
        return -1
    return topN(y_pred, y_true, 5)

def topN(output, target, topk):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        batch_size = target.size(0)

        _, pred = output.topk(topk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        correct_k = correct[:topk].view(-1).float().sum(0, keepdim=True)
        return correct_k.mul_(100.0 / batch_size).item()


# multilabel metrics
def multilabel_accuracy(y_true, y_pred):
    return np.mean(y_true == (y_pred > 0.5))


# regression metrics
def mean_distance(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))
