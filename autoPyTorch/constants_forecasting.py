# The cosntant values for time series forecasting comes from
# https://github.com/rakshitha123/TSForecasting/blob/master/experiments/deep_learning_experiments.py
# seasonality map, maps a frequency value to a number

FORECASTING_BUDGET_TYPE = ['resolution', 'num_seq', 'num_sample_per_seq']

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

MAX_WINDOW_SIZE_BASE = 500
