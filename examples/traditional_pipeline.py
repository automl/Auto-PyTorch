"""
======================
Tabular Classification
======================
"""
import typing

import sklearn.datasets
import sklearn.model_selection

from autoPyTorch.datasets.tabular_dataset import TabularDataset
from autoPyTorch.pipeline.traditional_tabular_classification import TraditionalTabularClassificationPipeline
from autoPyTorch.utils.backend import create
from autoPyTorch.utils.pipeline import get_dataset_requirements


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

    backend = create(temporary_directory='./tmp/example_trad_clf_1_tmp',
                     output_directory='./tmp/example_trad_clf_1_out',
                     delete_tmp_folder_after_terminate=False)
    backend.save_datamanager(datamanager)
    info = {'task_type': datamanager.task_type,
            'output_type': datamanager.output_type,
            'issparse': datamanager.issparse,
            'numerical_columns': datamanager.numerical_columns,
            'categorical_columns': datamanager.categorical_columns}
    dataset_requirements = get_dataset_requirements(info=info)
    dataset_properties = datamanager.get_dataset_properties(dataset_requirements)
    pipeline = TraditionalTabularClassificationPipeline(dataset_properties=dataset_properties)

    split_id = 0
    X = dict({'dataset_properties': dataset_properties,
              'backend': backend,
              'X_train': datamanager.train_tensors[0],
              'y_train': datamanager.train_tensors[1],
              'X_test': datamanager.test_tensors[0] if datamanager.test_tensors is not None else None,
              'y_test': datamanager.test_tensors[1] if datamanager.test_tensors is not None else None,
              'train_indices': datamanager.splits[split_id][0],
              'val_indices': datamanager.splits[split_id][1],
              'split_id': split_id,
              'job_id': 0
              })

    # Configuration space
    pipeline_cs = pipeline.get_hyperparameter_search_space()
    print("Pipeline CS:\n", '_' * 40, f"\n{pipeline_cs}")
    config = pipeline_cs.sample_configuration()
    print("Pipeline Random Config:\n", '_' * 40, f"\n{config}")
    pipeline.set_hyperparameters(config)

    # Fit the pipeline
    print("Fitting the pipeline...")
    pipeline.fit(X)

    # Showcase some components of the pipeline
    print(pipeline)

    predictions = pipeline.predict(X_test.to_numpy())
    print(predictions)
