import sklearn.metrics as metrics
import time
import torch
import numpy as np

# classification metrics
def accuracy(y_pred, y_true):
    return metrics.accuracy_score(undo_ohe(y_true).cpu(), undo_ohe(y_pred).cpu()) * 100

def cross_entropy(y_pred, y_true):
    y_pred = y_pred.cpu()
    y_true = undo_ohe(y_true).cpu()
    #try:
    #    loss = metrics.log_loss(y_true, y_pred, labels=np.arange(max(y_pred))
    #except:
    #    print(y_pred, y_true)
    return metrics.log_loss(y_true, y_pred, labels=np.arange(len(y_pred[0])))

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

def auc_metric(y_true, y_pred):
    return (2 * metrics.roc_auc_score(y_true.cpu().numpy(), y_pred.cpu().numpy()) - 1) * 100


# multilabel metric
def multilabel_accuracy(y_true, y_pred):
    return (y_true.long() == (y_pred > 0.5).long()).float().mean().item() * 100

# regression metric
def mean_distance(y_true, y_pred):
    return (y_true - y_pred).abs().mean().item()

def undo_ohe(y):
    if len(y.shape) == 1:
        return y
    return y.max(1)[1]
