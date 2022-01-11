import multiprocessing
import os
import queue
import shutil
import sys
import unittest
import unittest.mock

from ConfigSpace import Configuration
import ConfigSpace

import numpy as np

from sklearn.base import BaseEstimator

from smac.tae import StatusType

from autoPyTorch.automl_common.common.utils.backend import create
from autoPyTorch.datasets.resampling_strategy import NoResamplingStrategyTypes
from autoPyTorch.evaluation.test_evaluator import TestEvaluator
from autoPyTorch.evaluation.utils import read_queue
from autoPyTorch.pipeline.base_pipeline import BasePipeline
from autoPyTorch.pipeline.components.training.metrics.metrics import accuracy

this_directory = os.path.dirname(__file__)
sys.path.append(this_directory)
from evaluation_util import (  # noqa (E402: module level import not at top of file)
    BaseEvaluatorTest,
    get_binary_classification_datamanager,
    get_multiclass_classification_datamanager,
    get_regression_datamanager,
)  # noqa (E402: module level import not at top of file)


class BackendMock(object):
    def load_datamanager(self):
        return get_multiclass_classification_datamanager()


class Dummy(object):
    def __init__(self):
        self.name = 'dummy'


class DummyPipeline(BasePipeline):
    def __init__(self):
        mocked_estimator = unittest.mock.Mock(spec=BaseEstimator)
        self.steps = [('MockStep', mocked_estimator)]
        pass

    def predict_proba(self, X, batch_size=None):
        return np.tile([0.6, 0.4], (len(X), 1))

    def get_additional_run_info(self):
        return {}


class TestTestEvaluator(BaseEvaluatorTest, unittest.TestCase):
    _multiprocess_can_split_ = True

    def setUp(self):
        """
        Creates a backend mock
        """
        tmp_dir_name = self.id()
        self.ev_path = os.path.join(this_directory, '.tmp_evaluations', tmp_dir_name)
        if os.path.exists(self.ev_path):
            shutil.rmtree(self.ev_path)
        os.makedirs(self.ev_path, exist_ok=False)
        dummy_model_files = [os.path.join(self.ev_path, str(n)) for n in range(100)]
        dummy_pred_files = [os.path.join(self.ev_path, str(n)) for n in range(100, 200)]
        dummy_cv_model_files = [os.path.join(self.ev_path, str(n)) for n in range(200, 300)]
        backend_mock = unittest.mock.Mock()
        backend_mock.get_model_dir.return_value = self.ev_path
        backend_mock.get_cv_model_dir.return_value = self.ev_path
        backend_mock.get_model_path.side_effect = dummy_model_files
        backend_mock.get_cv_model_path.side_effect = dummy_cv_model_files
        backend_mock.get_prediction_output_path.side_effect = dummy_pred_files
        backend_mock.temporary_directory = self.ev_path
        self.backend_mock = backend_mock

        self.tmp_dir = os.path.join(self.ev_path, 'tmp_dir')
        self.output_dir = os.path.join(self.ev_path, 'out_dir')

    def tearDown(self):
        if os.path.exists(self.ev_path):
            shutil.rmtree(self.ev_path)

    @unittest.mock.patch('autoPyTorch.pipeline.tabular_classification.TabularClassificationPipeline')
    def test_no_resampling(self, pipeline_mock):
        # Binary iris, contains 69 train samples, 31 test samples
        D = get_binary_classification_datamanager(NoResamplingStrategyTypes.no_resampling)
        pipeline_mock.predict_proba.side_effect = \
            lambda X, batch_size=None: np.tile([0.6, 0.4], (len(X), 1))
        pipeline_mock.side_effect = lambda **kwargs: pipeline_mock
        pipeline_mock.get_additional_run_info.return_value = None
        pipeline_mock.get_default_pipeline_options.return_value = {'budget_type': 'epochs', 'epochs': 10}

        configuration = unittest.mock.Mock(spec=Configuration)
        backend_api = create(self.tmp_dir, self.output_dir, 'autoPyTorch')
        backend_api.load_datamanager = lambda: D
        queue_ = multiprocessing.Queue()

        evaluator = TestEvaluator(backend_api, queue_, configuration=configuration, metric=accuracy, budget=0)
        evaluator.file_output = unittest.mock.Mock(spec=evaluator.file_output)
        evaluator.file_output.return_value = (None, {})

        evaluator.fit_predict_and_loss()

        rval = read_queue(evaluator.queue)
        self.assertEqual(len(rval), 1)
        result = rval[0]['loss']
        self.assertEqual(len(rval[0]), 3)
        self.assertRaises(queue.Empty, evaluator.queue.get, timeout=1)

        self.assertEqual(evaluator.file_output.call_count, 1)
        self.assertEqual(result, 0.5806451612903225)
        self.assertEqual(pipeline_mock.fit.call_count, 1)
        # 2 calls because of train and test set
        self.assertEqual(pipeline_mock.predict_proba.call_count, 2)
        self.assertEqual(evaluator.file_output.call_count, 1)
        # Should be none as no val preds are mentioned
        self.assertIsNone(evaluator.file_output.call_args[0][1])
        # Number of y_test_preds and Y_test should be the same
        self.assertEqual(evaluator.file_output.call_args[0][0].shape[0],
                         D.test_tensors[1].shape[0])
        self.assertEqual(evaluator.pipeline.fit.call_count, 1)

    @unittest.mock.patch.object(TestEvaluator, '_loss')
    def test_file_output(self, loss_mock):

        D = get_regression_datamanager(NoResamplingStrategyTypes.no_resampling)
        D.name = 'test'
        self.backend_mock.load_datamanager.return_value = D
        configuration = unittest.mock.Mock(spec=Configuration)
        configuration.configuration_space = unittest.mock.Mock(spec=ConfigSpace)
        queue_ = multiprocessing.Queue()
        loss_mock.return_value = None

        evaluator = TestEvaluator(self.backend_mock, queue_, configuration=configuration, metric=accuracy, budget=0)

        self.backend_mock.get_model_dir.return_value = True
        evaluator.pipeline = 'model'
        evaluator.Y_optimization = D.train_tensors[1]
        rval = evaluator.file_output(
            D.train_tensors[1],
            None,
            D.test_tensors[1],
        )

        self.assertEqual(rval, (None, {}))
        # These targets are not saved as Fit evaluator is not used to make an ensemble
        self.assertEqual(self.backend_mock.save_targets_ensemble.call_count, 0)
        self.assertEqual(self.backend_mock.save_numrun_to_dir.call_count, 1)
        self.assertEqual(self.backend_mock.save_numrun_to_dir.call_args_list[-1][1].keys(),
                         {'seed', 'idx', 'budget', 'model', 'cv_model',
                          'ensemble_predictions', 'valid_predictions', 'test_predictions'})
        self.assertIsNotNone(self.backend_mock.save_numrun_to_dir.call_args_list[-1][1]['model'])
        self.assertIsNone(self.backend_mock.save_numrun_to_dir.call_args_list[-1][1]['cv_model'])

        # Check for not containing NaNs - that the models don't predict nonsense
        # for unseen data
        D.test_tensors[1][0] = np.NaN
        rval = evaluator.file_output(
            D.train_tensors[1],
            None,
            D.test_tensors[1],
        )
        self.assertEqual(
            rval,
            (
                1.0,
                {
                    'error':
                    'Model predictions for test set contains NaNs.'
                },
            )
        )

    @unittest.mock.patch('autoPyTorch.pipeline.tabular_classification.TabularClassificationPipeline')
    def test_predict_proba_binary_classification(self, mock):
        D = get_binary_classification_datamanager(NoResamplingStrategyTypes.no_resampling)
        self.backend_mock.load_datamanager.return_value = D
        mock.predict_proba.side_effect = lambda y, batch_size=None: np.array(
            [[0.1, 0.9]] * y.shape[0]
        )
        mock.side_effect = lambda **kwargs: mock
        mock.get_default_pipeline_options.return_value = {'budget_type': 'epochs', 'epochs': 10}
        configuration = unittest.mock.Mock(spec=Configuration)
        queue_ = multiprocessing.Queue()

        evaluator = TestEvaluator(self.backend_mock, queue_, configuration=configuration, metric=accuracy, budget=0)

        evaluator.fit_predict_and_loss()
        Y_test_pred = self.backend_mock.save_numrun_to_dir.call_args_list[0][-1][
            'ensemble_predictions']

        for i in range(7):
            self.assertEqual(0.9, Y_test_pred[i][1])

    def test_get_results(self):
        queue_ = multiprocessing.Queue()
        for i in range(5):
            queue_.put((i * 1, 1 - (i * 0.2), 0, "", StatusType.SUCCESS))
        result = read_queue(queue_)
        self.assertEqual(len(result), 5)
        self.assertEqual(result[0][0], 0)
        self.assertAlmostEqual(result[0][1], 1.0)