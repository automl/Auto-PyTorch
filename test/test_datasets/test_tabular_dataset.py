import typing
import unittest

import numpy as np

import pandas as pd

import sklearn.datasets
import sklearn.model_selection

from autoPyTorch.datasets.tabular_dataset import DataTypes, TabularDataset
from autoPyTorch.utils.backend import create
from autoPyTorch.utils.pipeline import get_dataset_requirements


class DataFrameTest(unittest.TestCase):
    def runTest(self):
        df = pd.DataFrame([['a', 0.1, 1], ['b', 0.2, np.nan]])
        target_df = pd.Series([1, 2])
        ds = TabularDataset(df, target_df)
        self.assertEqual(ds.data_types, [DataTypes.String, DataTypes.Float, DataTypes.Canonical])
        self.assertEqual(set(ds.itovs[2]), {np.nan, 1})
        self.assertEqual(set(ds.itovs[0]), {np.nan, 'a', 'b'})

        self.assertEqual(ds.vtois[0]['a'], 1)
        self.assertEqual(ds.vtois[0][np.nan], 0)
        self.assertEqual(ds.vtois[0][pd._libs.NaT], 0)
        self.assertEqual(ds.vtois[0][pd._libs.missing.NAType()], 0)
        self.assertTrue((ds.nan_mask == np.array([[0, 0, 0], [0, 0, 1]], dtype=np.bool)).all())


class NumpyArrayTest(unittest.TestCase):
    def runTest(self):
        matrix = np.array([(0, 0.1, 1), (1, np.nan, 3)], dtype='f4, f4, i4')
        target_df = pd.Series([1, 2])
        ds = TabularDataset(matrix, target_df)
        self.assertEqual(ds.data_types, [DataTypes.Canonical, DataTypes.Float, DataTypes.Canonical])
        self.assertEqual(set(ds.itovs[2]), {np.nan, 1, 3})

        self.assertEqual(ds.vtois[0][1], 2)
        self.assertEqual(ds.vtois[0][np.nan], 0)
        self.assertEqual(ds.vtois[0][pd._libs.NaT], 0)
        self.assertEqual(ds.vtois[0][pd._libs.missing.NAType()], 0)
        self.assertTrue((ds.nan_mask == np.array([[0, 0, 0], [0, 1, 0]], dtype=np.bool)).all())


def get_data_to_train() -> typing.Dict[str, typing.Any]:
    """
    This function returns a fit dictionary that within itself, contains all
    the information needed
    """

    # Get the training data for tabular classification
    # Move to Australian to showcase numerical vs categorical
    X, y = sklearn.datasets.fetch_openml(data_id=40981, return_X_y=True, as_frame=True)
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
        X,
        y,
        random_state=1,
        test_size=0.2,
    )
    # Fit the pipeline
    fit_dictionary = {
        'X_train': X_train,
        'y_train': y_train,
        'X_test': X_test,
        'y_test': y_test,
    }

    return fit_dictionary


class TabularDatasetTest(unittest.TestCase):

    def test_get_dataset_properties(self):
        # Get data to train
        fit_dictionary = get_data_to_train()

        # Build a repository with random fitted models
        try:
            backend = create(temporary_directory='/tmp/autoPyTorch_ensemble_test_tmp',
                             output_directory='/tmp/autoPyTorch_ensemble_test_out',
                             delete_tmp_folder_after_terminate=False)
        except Exception:
            self.assertRaises(FileExistsError)
            return unittest.skip("File already exists")

        fit_dictionary['backend'] = backend

        # Create the directory structure
        backend._make_internals_directory()

        # Create a datamanager for this toy problem
        datamanager = TabularDataset(
            X=fit_dictionary['X_train'], Y=fit_dictionary['y_train'],
            X_test=fit_dictionary['X_test'], Y_test=fit_dictionary['y_test'],
        )
        backend.save_datamanager(datamanager)

        datamanager = backend.load_datamanager()
        info = datamanager.get_required_dataset_info()
        dataset_requirements = get_dataset_requirements(info)

        dataset_properties = datamanager.get_dataset_properties(dataset_requirements)

        self.assertIsInstance(dataset_properties, dict)
        for dataset_requirement in dataset_requirements:
            self.assertIn(dataset_requirement.name, dataset_properties.keys())
            self.assertIsInstance(dataset_properties[dataset_requirement.name], dataset_requirement.supported_types)


if __name__ == '__main__':
    unittest.main()
