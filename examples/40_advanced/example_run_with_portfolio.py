"""
============================================
Tabular Classification with Greedy Portfolio
============================================

The following example shows how to fit a sample classification model
with AutoPyTorch using the greedy portfolio
"""
import os
import tempfile as tmp
import warnings

os.environ['JOBLIB_TEMP_FOLDER'] = tmp.gettempdir()
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

warnings.simplefilter(action='ignore', category=UserWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)

import sklearn.datasets
import sklearn.model_selection

from autoPyTorch.api.tabular_classification import TabularClassificationTask


############################################################################
# Data Loading
# ============
X, y = sklearn.datasets.fetch_openml(data_id=40981, return_X_y=True, as_frame=True)
X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
    X,
    y,
    random_state=42,
)

############################################################################
# Build and fit a classifier
# ==========================
api = TabularClassificationTask(
    seed=42,
)

############################################################################
# Search for an ensemble of machine learning algorithms
# =====================================================
api.search(
    X_train=X_train,
    y_train=y_train,
    X_test=X_test.copy(),
    y_test=y_test.copy(),
    optimize_metric='accuracy',
    total_walltime_limit=300,
    func_eval_time_limit_secs=50,
    # Setting this option to "greedy"
    # will make smac run the configurations
    # present in 'autoPyTorch/configs/greedy_portfolio.json'
    portfolio_selection="greedy"
)

############################################################################
# Print the final ensemble performance
# ====================================
y_pred = api.predict(X_test)
score = api.score(y_pred, y_test)
print(score)
# Print the final ensemble built by AutoPyTorch
print(api.show_models())

# Print statistics from search
print(api.sprint_statistics())
