import multiprocessing
import os
import queue
import shutil
import sys
import unittest
import unittest.mock
import pytest

from ConfigSpace import Configuration

import numpy as np

from sklearn.base import BaseEstimator

from smac.tae import StatusType

from autoPyTorch.automl_common.common.utils.backend import create
from autoPyTorch.datasets.resampling_strategy import CrossValTypes, NoResamplingStrategyTypes
from autoPyTorch.evaluation.time_series_forecasting_train_evaluator import TimeSeriesForecastingTrainEvaluator
from autoPyTorch.evaluation.utils import read_queue
from autoPyTorch.pipeline.base_pipeline import BasePipeline
from autoPyTorch.pipeline.components.training.metrics.metrics import mean_MASE_forecasting

this_directory = os.path.dirname(__file__)
sys.path.append(this_directory)
from evaluation_util import (  # noqa (E402: module level import not at top of file)
    BaseEvaluatorTest,
    get_binary_classification_datamanager,
    get_multiclass_classification_datamanager,
    get_regression_datamanager,
    get_forecasting_dataset
)  # noqa (E402: module level import not at top of file)

from test_evaluators import TestTrainEvaluator

class BackendMock(object):
    def load_datamanager(self):
        return get_multiclass_classification_datamanager()


class TestTimeSeriesForecastingTrainEvaluator(unittest.TestCase):
    def setUp(self):
        TestTrainEvaluator.setUp(self)

    def tearDown(self):
        TestTrainEvaluator.tearDown(self)

    @unittest.mock.patch('autoPyTorch.pipeline.time_series_forecasting.TimeSeriesForecastingPipeline')
    def test_holdout(self, pipeline_mock):
        pipeline_mock.fit_dictionary = {'budget_type': 'epochs', 'epochs': 50}
        D = get_forecasting_dataset()
        n_prediction_steps = D.n_prediction_steps
        pipeline_mock.predict.side_effect = \
            lambda X, batch_size=None: np.tile([0.], (len(X), n_prediction_steps))
        pipeline_mock.side_effect = lambda **kwargs: pipeline_mock
        pipeline_mock.get_additional_run_info.return_value = None

        configuration = unittest.mock.Mock(spec=Configuration)
        backend_api = create(self.tmp_dir, self.output_dir, prefix='autoPyTorch')
        backend_api.load_datamanager = lambda: D
        queue_ = multiprocessing.Queue()

        evaluator = TimeSeriesForecastingTrainEvaluator(backend_api,
                                                        queue_,
                                                        configuration=configuration,
                                                        metric=mean_MASE_forecasting, budget=0,
                                                        pipeline_config={'budget_type': 'epochs', 'epochs': 50},
                                                        min_num_test_instances=100)
        evaluator.file_output = unittest.mock.Mock(spec=evaluator.file_output)
        evaluator.file_output.return_value = (None, {})

        evaluator.fit_predict_and_loss()

        rval = read_queue(evaluator.queue)

        self.assertEqual(len(rval), 1)
        result = rval[0]['loss']
        self.assertEqual(len(rval[0]), 3)
        self.assertRaises(queue.Empty, evaluator.queue.get, timeout=1)

        self.assertEqual(evaluator.file_output.call_count, 1)
        self.assertEqual(result, 4592.0)
        self.assertEqual(pipeline_mock.fit.call_count, 1)
        # As forecasting inference could be quite expensive, we only allow one validation prediction
        self.assertEqual(pipeline_mock.predict.call_count, 1)

        self.assertEqual(evaluator.file_output.call_count, 1)
        self.assertEqual(evaluator.file_output.call_args[0][0].shape[0], len(D.splits[0][1]) * n_prediction_steps)
        self.assertIsNone(evaluator.file_output.call_args[0][1])
        self.assertIsNone(evaluator.file_output.call_args[0][2])
        self.assertEqual(evaluator.pipeline.fit.call_count, 1)

        res = evaluator.file_output.call_args[0][0].reshape(-1, n_prediction_steps, evaluator.num_targets)
        assert np.all(res == 0.)



    @unittest.mock.patch('autoPyTorch.pipeline.time_series_forecasting.TimeSeriesForecastingPipeline')
    def test_cv(self, pipeline_mock):
        D = get_forecasting_dataset(resampling_strategy=CrossValTypes.time_series_cross_validation)
        assert D.resampling_strategy_args['num_splits'] == 3

        n_prediction_steps = D.n_prediction_steps

        pipeline_mock.predict.side_effect = \
            lambda X, batch_size=None: np.tile([0.], (len(X), n_prediction_steps))

        pipeline_mock.side_effect = lambda **kwargs: pipeline_mock
        pipeline_mock.get_additional_run_info.return_value = None

        configuration = unittest.mock.Mock(spec=Configuration)
        backend_api = create(self.tmp_dir, self.output_dir, prefix='autoPyTorch')
        backend_api.load_datamanager = lambda: D
        queue_ = multiprocessing.Queue()

        evaluator = TimeSeriesForecastingTrainEvaluator(backend_api,
                                                        queue_,
                                                        configuration=configuration,
                                                        metric=mean_MASE_forecasting, budget=0,
                                                        pipeline_config={'budget_type': 'epochs', 'epochs': 50})

        evaluator.file_output = unittest.mock.Mock(spec=evaluator.file_output)
        evaluator.file_output.return_value = (None, {})

        evaluator.fit_predict_and_loss()

        rval = read_queue(evaluator.queue)
        self.assertEqual(len(rval), 1)
        result = rval[0]['loss']
        self.assertEqual(len(rval[0]), 3)
        self.assertRaises(queue.Empty, evaluator.queue.get, timeout=1)

        self.assertEqual(evaluator.file_output.call_count, 1)
        self.assertAlmostEqual(result, 4587.208333333334)
        self.assertEqual(pipeline_mock.fit.call_count, 3)
        # 3 calls because of the 3 times validation evaluations
        self.assertEqual(pipeline_mock.predict.call_count, 3)
        # as the optimisation preds in cv is concatenation of the 5 folds,
        # so it is 5*splits
        self.assertEqual(evaluator.file_output.call_args[0][0].shape[0],
                         3 * len(D.splits[0][1]) * n_prediction_steps, evaluator.file_output.call_args)
        self.assertIsNone(evaluator.file_output.call_args[0][1])
        # we do not have test sets
        self.assertIsNone(evaluator.file_output.call_args[0][2])

        res = evaluator.file_output.call_args[0][0].reshape(-1, n_prediction_steps, evaluator.num_targets)
        assert np.all(res == 0.)

    @unittest.mock.patch('autoPyTorch.pipeline.time_series_forecasting.TimeSeriesForecastingPipeline')
    def test_proxy_val_set(self, pipeline_mock):
        pipeline_mock.fit_dictionary = {'budget_type': 'epochs', 'epochs': 0.1}
        D = get_forecasting_dataset()
        n_prediction_steps = D.n_prediction_steps
        pipeline_mock.predict.side_effect = \
            lambda X, batch_size=None: np.tile([0.], (len(X), n_prediction_steps))
        pipeline_mock.side_effect = lambda **kwargs: pipeline_mock
        pipeline_mock.get_additional_run_info.return_value = None

        configuration = unittest.mock.Mock(spec=Configuration)
        backend_api = create(self.tmp_dir, self.output_dir, prefix='autoPyTorch')
        backend_api.load_datamanager = lambda: D
        queue_ = multiprocessing.Queue()

        evaluator = TimeSeriesForecastingTrainEvaluator(backend_api,
                                                        queue_,
                                                        configuration=configuration,
                                                        metric=mean_MASE_forecasting, budget=0.3,
                                                        pipeline_config={'budget_type': 'epochs', 'epochs': 50},
                                                        min_num_test_instances=1)
        evaluator.file_output = unittest.mock.Mock(spec=evaluator.file_output)
        evaluator.file_output.return_value = (None, {})

        evaluator.fit_predict_and_loss()

        rval = read_queue(evaluator.queue)

        self.assertEqual(len(rval), 1)
        result = rval[0]['loss']

        self.assertEqual(result, 925.2)
        res = evaluator.file_output.call_args[0][0].reshape(-1, n_prediction_steps, evaluator.num_targets)

        n_evaluated_pip_mock = 0

        for i_seq, seq_output in enumerate(res):
            if i_seq % 3 == 0 and n_evaluated_pip_mock < 3:
                n_evaluated_pip_mock += 1
                assert np.all(seq_output == 0.)
            else:
                # predict with dummy predictor
                assert np.all(seq_output == D.datasets[i_seq][-1][0]['past_targets'][-1].numpy())
