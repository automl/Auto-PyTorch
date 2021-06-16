"""
======================
Tabular Regression
======================

The following example shows how to fit a sample regression model
with AutoPyTorch
"""
import os
import tempfile as tmp
import warnings

import sklearn.datasets
import sklearn.model_selection

os.environ['JOBLIB_TEMP_FOLDER'] = tmp.gettempdir()
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

warnings.simplefilter(action='ignore', category=UserWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)

from autoPyTorch.api.tabular_regression import TabularRegressionTask


############################################################################
# Data Loading
# ============
X, y = sklearn.datasets.fetch_openml(name='boston', return_X_y=True, as_frame=True)
X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
    X,
    y,
    random_state=1,
)

############################################################################
# Build and fit a regressor
# ==========================
api = TabularRegressionTask()

############################################################################
# Search for an ensemble of machine learning algorithms
# =====================================================
api.search(
    X_train=X_train,
    y_train=y_train,
    X_test=X_test.copy(),
    y_test=y_test.copy(),
    optimize_metric='r2',
    total_walltime_limit=300,
    func_eval_time_limit_secs=50,
)

############################################################################
# Print the final ensemble performance
# ====================================
print(api.run_history, api.trajectory)
y_pred = api.predict(X_test)

# Rescale the Neural Network predictions into the original target range
score = api.score(y_pred, y_test)

print(score)
# Print the final ensemble built by AutoPyTorch
print(api.show_models())
