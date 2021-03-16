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


if __name__ == '__main__':

    ############################################################################
    # Data Loading
    # ============
    X, y = sklearn.datasets.fetch_openml(name='boston', return_X_y=True, as_frame=True)
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
        X,
        y,
        random_state=1,
    )

    # Scale the regression targets to have zero mean and unit variance.
    # This is important for Neural Networks since predicting large target values would require very large weights.
    # One can later rescale the network predictions like this: y_pred = y_pred_scaled * y_train_std + y_train_mean
    y_train_mean = y_train.mean()
    y_train_std = y_train.std()

    y_train_scaled = (y_train - y_train_mean) / y_train_std
    y_test_scaled = (y_test - y_train_mean) / y_train_std

    ############################################################################
    # Build and fit a regressor
    # ==========================
    api = TabularRegressionTask(
        temporary_directory='./tmp/autoPyTorch_example_tmp_02',
        output_directory='./tmp/autoPyTorch_example_out_02',
        # To maintain logs of the run, set the next two as False
        delete_tmp_folder_after_terminate=True,
        delete_output_folder_after_terminate=True
    )

    ############################################################################
    # Search for an ensemble of machine learning algorithms
    # =====================================================
    api.search(
        X_train=X_train,
        y_train=y_train_scaled,
        X_test=X_test.copy(),
        y_test=y_test_scaled.copy(),
        optimize_metric='r2',
        total_walltime_limit=300,
        func_eval_time_limit=50,
        enable_traditional_pipeline=False,
    )

    ############################################################################
    # Print the final ensemble performance
    # ====================================
    print(api.run_history, api.trajectory)
    y_pred_scaled = api.predict(X_test)

    # Rescale the Neural Network predictions into the original target range
    y_pred = y_pred_scaled * y_train_std + y_train_mean
    score = api.score(y_pred, y_test)

    print(score)
    # Print the final ensemble built by AutoPyTorch
    print(api.show_models())
