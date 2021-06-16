from enum import Enum


class AutoPyTorchToCatboostMetrics(Enum):
    mean_absolute_error = "MAE"
    root_mean_squared_error = "RMSE"
    mean_squared_log_error = "MSLE"
    r2 = "R2"
    accuracy = "Accuracy"
    balanced_accuracy = "BalancedAccuracy"
    f1 = "F1"
    roc_auc = "AUC"
    precision = "Precision"
    recall = "Recall"
    log_loss = "Logloss"
