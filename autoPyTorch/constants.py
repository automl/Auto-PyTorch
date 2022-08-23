TABULAR_CLASSIFICATION = 1
IMAGE_CLASSIFICATION = 2
TABULAR_REGRESSION = 3
IMAGE_REGRESSION = 4
TIMESERIES_FORECASTING = 5

REGRESSION_TASKS = [TABULAR_REGRESSION, IMAGE_REGRESSION]
CLASSIFICATION_TASKS = [TABULAR_CLASSIFICATION, IMAGE_CLASSIFICATION]
FORECASTING_TASKS = [TIMESERIES_FORECASTING]  # TODO extend FORECASTING TASKS to Classification and regression tasks

TABULAR_TASKS = [TABULAR_CLASSIFICATION, TABULAR_REGRESSION]
IMAGE_TASKS = [IMAGE_CLASSIFICATION, IMAGE_REGRESSION]
TIMESERIES_TASKS = [TIMESERIES_FORECASTING]
TASK_TYPES = REGRESSION_TASKS + CLASSIFICATION_TASKS + FORECASTING_TASKS

TASK_TYPES_TO_STRING = \
    {TABULAR_CLASSIFICATION: 'tabular_classification',
     IMAGE_CLASSIFICATION: 'image_classification',
     TABULAR_REGRESSION: 'tabular_regression',
     IMAGE_REGRESSION: 'image_regression',
     TIMESERIES_FORECASTING: 'time_series_forecasting'}

STRING_TO_TASK_TYPES = \
    {'tabular_classification': TABULAR_CLASSIFICATION,
     'image_classification': IMAGE_CLASSIFICATION,
     'tabular_regression': TABULAR_REGRESSION,
     'image_regression': IMAGE_REGRESSION,
     'time_series_forecasting': TIMESERIES_FORECASTING}

# Output types have been defined as in scikit-learn type_of_target
# (https://scikit-learn.org/stable/modules/generated/sklearn.utils.multiclass.type_of_target.html)
BINARY = 10
CONTINUOUSMULTIOUTPUT = 11
MULTICLASS = 12
CONTINUOUS = 13
MULTICLASSMULTIOUTPUT = 14

OUTPUT_TYPES = [BINARY, CONTINUOUSMULTIOUTPUT, MULTICLASS, CONTINUOUS]

OUTPUT_TYPES_TO_STRING = \
    {BINARY: 'binary',
     CONTINUOUSMULTIOUTPUT: 'continuous-multioutput',
     MULTICLASS: 'multiclass',
     CONTINUOUS: 'continuous',
     MULTICLASSMULTIOUTPUT: 'multiclass-multioutput'}

STRING_TO_OUTPUT_TYPES = \
    {'binary': BINARY,
     'continuous-multioutput': CONTINUOUSMULTIOUTPUT,
     'multiclass': MULTICLASS,
     'continuous': CONTINUOUS,
     'multiclass-multioutput': MULTICLASSMULTIOUTPUT}

CLASSIFICATION_OUTPUTS = [BINARY, MULTICLASS, MULTICLASSMULTIOUTPUT]
REGRESSION_OUTPUTS = [CONTINUOUS, CONTINUOUSMULTIOUTPUT]

ForecastingDependenciesNotInstalledMSG = "Additional dependencies must be installed to work with time series " \
                                         "forecasting tasks! Please run \n pip install autoPyTorch[forecasting] \n to "\
                                         "install the corresponding dependencies!"

# This value is applied to ensure numerical stability: Sometimes we want to rescale some values: value / scale.
# We make the scale value to be 1 if it is smaller than this value to ensure that the scaled value will not resutl in
# overflow
VERY_SMALL_VALUE = 1e-12

# The constant values for time series forecasting comes from
# https://github.com/rakshitha123/TSForecasting/blob/master/experiments/deep_learning_experiments.py
# seasonality map, maps a frequency value to a number
FORECASTING_BUDGET_TYPE = ('resolution', 'num_seq', 'num_sample_per_seq')

SEASONALITY_MAP = {
    "1min": [1440, 10080, 525960],
    "10min": [144, 1008, 52596],
    "30min": [48, 336, 17532],
    "1H": [24, 168, 8766],
    "1D": 7,
    "1W": 365.25 / 7,
    "1M": 12,
    "1Q": 4,
    "1Y": 1
}

# To avoid that we get a sequence that is too long to be fed to a network
MAX_WINDOW_SIZE_BASE = 500

# AutoPyTorch optionally allows network inference or metrics calculation for the following datasets
OPTIONAL_INFERENCE_CHOICES = ('test',)
