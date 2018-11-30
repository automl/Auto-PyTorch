import sklearn.metrics as metrics

# classification metrics
def accuracy(y_true, y_pred):
    return (y_true.max(1)[1] == y_pred.max(1)[1]).float().mean().item() * 100

def auc_metric(y_true, y_pred):
    return (2 * metrics.roc_auc_score(y_true.cpu().numpy(), y_pred.cpu().numpy()) - 1) * 100


# multilabel metric
def multilabel_accuracy(y_true, y_pred):
    return (y_true.long() == (y_pred > 0.5).long()).float().mean().item() * 100

# regression metric
def mean_distance(y_true, y_pred):
    return (y_true - y_pred).abs().mean().item()