"""
======================
Tabular Classification
======================
"""
import typing
import warnings

warnings.simplefilter(action='ignore', category=UserWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)

import sklearn.datasets
import sklearn.model_selection

from autoPyTorch.api.tabular_classification import TabularClassificationTask
from autoPyTorch.datasets.tabular_dataset import TabularDataset


# Get the training data for tabular classification
def get_data_to_train() -> typing.Tuple[typing.Any, typing.Any, typing.Any, typing.Any]:
    """
    This function returns a fit dictionary that within itself, contains all
    the information to fit a pipeline
    """

    # Get the training data for tabular classification
    # Move to Australian to showcase numerical vs categorical
    X, y = sklearn.datasets.fetch_openml(data_id=40981, return_X_y=True, as_frame=True)
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
        X,
        y,
        random_state=1,
    )

    return X_train, X_test, y_train, y_test


if __name__ == '__main__':
    # Get data to train
    X_train, X_test, y_train, y_test = get_data_to_train()

    # Create a datamanager for this toy problem
    datamanager = TabularDataset(
        X=X_train, Y=y_train,
        X_test=X_test, Y_test=y_test)

    api = TabularClassificationTask(delete_tmp_folder_after_terminate=False,)
    api.search(dataset=datamanager, optimize_metric='accuracy', total_walltime_limit=500, func_eval_time_limit=150)
    print(api.run_history, api.trajectory)
    y_pred = api.predict(X_test)
    score = api.score(y_pred, y_test)
    print(score)
