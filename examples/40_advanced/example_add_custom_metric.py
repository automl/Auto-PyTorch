"""
======================
Tabular Classification
======================

The following example shows how to optimize
AutoPyTorch on a custom metric
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
from autoPyTorch.metrics import CLASSIFICATION_METRICS
from autoPyTorch.pipeline.components.training.metrics.base import make_metric
from autoPyTorch.pipeline.components.training.metrics.utils import add_metric


############################################################################
# Data Loading
# ============
X, y = sklearn.datasets.fetch_openml(data_id=40981, return_X_y=True, as_frame=True)
X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
    X,
    y,
    random_state=1,
)


###############################################################################
# Define and add custom score function
# ====================================
def score_function(y_test, y_pred):
    return sum(y_pred==y_test) / y_pred.shape[0]

print("#"*80)
print(f"Current metrics available for classification: {list(CLASSIFICATION_METRICS.keys())}")
custom_metric = make_metric(name="custom_metric", score_func=score_function, worst_possible_result=0, greater_is_better=True)

add_metric(metric=custom_metric, task_type="tabular_classification")
print("#"*80)
print(f"Metrics available for classification after adding custom metric: {list(CLASSIFICATION_METRICS.keys())}")


############################################################################
# Build and fit a classifier
# ==========================
api = TabularClassificationTask(
    # To maintain logs of the run, you can uncomment the
    # Following lines
    # temporary_directory='./tmp/autoPyTorch_example_tmp_01',
    # output_directory='./tmp/autoPyTorch_example_out_01',
    # delete_tmp_folder_after_terminate=False,
    # delete_output_folder_after_terminate=False,
    seed=42,
)

######################################################################################
# Search for an ensemble of machine learning algorithms optimised on the custom metric
# ====================================================================================
api.search(
    X_train=X_train,
    y_train=y_train,
    X_test=X_test.copy(),
    y_test=y_test.copy(),
    dataset_name='Australian',
    optimize_metric='custom_metric',
    total_walltime_limit=300,
    func_eval_time_limit_secs=50,
    memory_limit=None,
)

############################################################################
# Print the final ensemble performance
# ====================================

y_pred = api.predict(X_test)
score = api.score(y_pred, y_test)
print(score)

# Print statistics from search
print(api.sprint_statistics())

# Print the final ensemble built by AutoPyTorch
print(api.show_models())
