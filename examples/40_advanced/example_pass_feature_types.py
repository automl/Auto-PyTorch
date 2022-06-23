"""
=====================================================
Tabular Classification with user passed feature types
=====================================================

The following example shows how to pass feature typesfor datasets which are in 
numpy format (also works for dataframes and lists) fit a sample classification 
model with AutoPyTorch.

AutoPyTorch relies on column dtypes for intepreting the feature types. But they 
can be misinterpreted for example, when dataset is passed as a numpy array, all 
the data is interpreted as numerical if it's dtype is int or float. However, the 
categorical values could have been encoded as integers.

Passing feature types helps AutoPyTorch interpreting them correctly as well as
validates the dataset by checking the dtype of the columns for any incompatibilities.
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

import openml
import sklearn.model_selection

from autoPyTorch.api.tabular_classification import TabularClassificationTask


############################################################################
# Data Loading
# ============
task = openml.tasks.get_task(task_id=146821)
dataset = task.get_dataset()
X, y, categorical_indicator, _ = dataset.get_data(
    dataset_format='array',
    target=dataset.default_target_attribute,
)
X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
    X,
    y,
    random_state=1,
)

feat_types = ["numerical" if not indicator else "categorical" for indicator in categorical_indicator]

# 
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

############################################################################
# Search for an ensemble of machine learning algorithms
# =====================================================
api.search(
    X_train=X_train,
    y_train=y_train,
    X_test=X_test.copy(),
    y_test=y_test.copy(),
    dataset_name='Australian',
    optimize_metric='accuracy',
    total_walltime_limit=100,
    func_eval_time_limit_secs=50,
    feat_types=feat_types,
    enable_traditional_pipeline=False
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
