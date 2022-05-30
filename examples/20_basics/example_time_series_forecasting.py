"""
======================
Time Series Forecasting
======================

The following example shows how to fit a sample forecasting model
with AutoPyTorch. This is only a dummmy example because of the limited size of the dataset.
Thus, it could be possible that the AutoPyTorch model does not perform as well as a dummy predictor
"""
import os
import tempfile as tmp
import warnings
import copy

os.environ['JOBLIB_TEMP_FOLDER'] = tmp.gettempdir()
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

warnings.simplefilter(action='ignore', category=UserWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)

from sktime.datasets import load_longley
targets, features = load_longley()

forecasting_horizon = 3

# each series represent an element in the List
# we take the last forecasting_horizon  as test targets. The itme before that as training targets
# Normally the value to be forecasted should follow the training sets
y_train = [targets[: -forecasting_horizon]]
y_test = [targets[-forecasting_horizon:]]

# same for features. For uni-variant models, X_train, X_test can be omitted
X_train = [features[: -forecasting_horizon]]
# Here x_test indicates the 'known future features': they are the features known previously, features that are unknown
# could be replaced with NAN or zeros (which will not be used by our networks). If no feature is known beforehand,
# we could also omit X_test
known_future_features = list(features.columns)
X_test = [features[-forecasting_horizon:]]

start_times = [targets.index.to_timestamp()[0]]
freq = '1Y'

from autoPyTorch.api.time_series_forecasting import TimeSeriesForecastingTask
############################################################################
# Build and fit a forecaster
# ==========================
api = TimeSeriesForecastingTask()

############################################################################
# Search for an ensemble of machine learning algorithms
# =====================================================
api.search(
    X_train=X_train,
    y_train=copy.deepcopy(y_train),
    X_test=X_test,
    optimize_metric='mean_MASE_forecasting',
    n_prediction_steps=forecasting_horizon,
    memory_limit=None,
    freq=freq,
    start_times=start_times,
    func_eval_time_limit_secs=50,
    total_walltime_limit=60,
    min_num_test_instances=1000,  # proxy validation sets. This only works for the tasks with more than 1000 series
    known_future_features=known_future_features,
)


from autoPyTorch.datasets.time_series_dataset import TimeSeriesSequence

test_sets = []

# We could construct test sets from scratch
for feature, future_feature, target, start_time in zip(X_train, X_test,y_train, start_times):
    test_sets.append(
        TimeSeriesSequence(X=feature.values,
                           Y=target.values,
                           X_test=future_feature.values,
                           start_time=start_time,
                           is_test_set=True,
                           # additional information required to construct a new time series sequence
                           **api.dataset.sequences_builder_kwargs
                           )
    )
# Alternatively, if we only want to forecast the value after the X_train, we could directly ask datamanager to
# generate a test set:
# test_sets2 = api.dataset.generate_test_seqs()

pred = api.predict(test_sets)
