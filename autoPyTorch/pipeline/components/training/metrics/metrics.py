from functools import partial


import sklearn.metrics

from smac.utils.constants import MAXINT

from autoPyTorch.pipeline.components.training.metrics.base import make_metric

# Standard regression scores
mean_absolute_error = make_metric('mean_absolute_error',
                                  sklearn.metrics.mean_absolute_error,
                                  optimum=0,
                                  worst_possible_result=MAXINT,
                                  greater_is_better=False)
mean_squared_error = make_metric('mean_squared_error',
                                 sklearn.metrics.mean_squared_error,
                                 optimum=0,
                                 worst_possible_result=MAXINT,
                                 greater_is_better=False,
                                 squared=True)
root_mean_squared_error = make_metric('root_mean_squared_error',
                                      sklearn.metrics.mean_squared_error,
                                      optimum=0,
                                      worst_possible_result=MAXINT,
                                      greater_is_better=False,
                                      squared=False)
mean_squared_log_error = make_metric('mean_squared_log_error',
                                     sklearn.metrics.mean_squared_log_error,
                                     optimum=0,
                                     worst_possible_result=MAXINT,
                                     greater_is_better=False, )
median_absolute_error = make_metric('median_absolute_error',
                                    sklearn.metrics.median_absolute_error,
                                    optimum=0,
                                    worst_possible_result=MAXINT,
                                    greater_is_better=False)
r2 = make_metric('r2',
                 sklearn.metrics.r2_score,
                 worst_possible_result=-MAXINT)

# Standard Classification Scores
accuracy = make_metric('accuracy',
                       sklearn.metrics.accuracy_score)
balanced_accuracy = make_metric('balanced_accuracy',
                                sklearn.metrics.balanced_accuracy_score)
f1 = make_metric('f1',
                 sklearn.metrics.f1_score)

# Score functions that need decision values
roc_auc = make_metric('roc_auc', sklearn.metrics.roc_auc_score, needs_threshold=True)
average_precision = make_metric('average_precision',
                                sklearn.metrics.average_precision_score,
                                needs_threshold=True)
precision = make_metric('precision',
                        sklearn.metrics.precision_score)
recall = make_metric('recall',
                     sklearn.metrics.recall_score)

# Score function for probabilistic classification
log_loss = make_metric('log_loss',
                       sklearn.metrics.log_loss,
                       optimum=0,
                       worst_possible_result=MAXINT,
                       greater_is_better=False,
                       needs_proba=True)

REGRESSION_METRICS = dict()
for scorer in [mean_absolute_error, mean_squared_error, root_mean_squared_error,
               mean_squared_log_error, median_absolute_error, r2]:
    REGRESSION_METRICS[scorer.name] = scorer

CLASSIFICATION_METRICS = dict()

for scorer in [accuracy, balanced_accuracy, roc_auc, average_precision,
               log_loss]:
    CLASSIFICATION_METRICS[scorer.name] = scorer

for name, metric in [('precision', sklearn.metrics.precision_score),
                     ('recall', sklearn.metrics.recall_score),
                     ('f1', sklearn.metrics.f1_score)]:
    globals()[name] = make_metric(name, metric)
    CLASSIFICATION_METRICS[name] = globals()[name]
    for average in ['macro', 'micro', 'samples', 'weighted']:
        qualified_name = '{0}_{1}'.format(name, average)
        globals()[qualified_name] = make_metric(qualified_name,
                                                partial(metric,
                                                        pos_label=None,
                                                        average=average))
        CLASSIFICATION_METRICS[qualified_name] = globals()[qualified_name]
