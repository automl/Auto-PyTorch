"""
=====================================================
Tabular Classification with Post-Hoc Ensemble Fitting
=====================================================

The following example shows how to fit a sample classification model
and create an ensemble post-hoc with AutoPyTorch
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


if __name__ == '__main__':

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
        ensemble_size=0,
        seed=42,
    )

    ############################################################################
    # Search for the best neural network
    # ==================================
    api.search(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test.copy(),
        y_test=y_test.copy(),
        optimize_metric='accuracy',
        total_walltime_limit=100,
        func_eval_time_limit_secs=50
    )

    ############################################################################
    # Print the final performance of the incumbent neural network
    # ===========================================================
    print(api.run_history, api.trajectory)
    y_pred = api.predict(X_test)
    score = api.score(y_pred, y_test)
    print(score)

    ############################################################################
    # Fit an ensemble with the neural networks fitted during the search
    # =================================================================

    api.fit_ensemble(ensemble_size=5,
                     # Set the enable_traditional_pipeline=True
                     # to also include traditional models
                     # in the ensemble
                     enable_traditional_pipeline=False)
    # Print the final ensemble built by AutoPyTorch
    y_pred = api.predict(X_test)
    score = api.score(y_pred, y_test)
    print(score)
    print(api.show_models())
    api._cleanup()