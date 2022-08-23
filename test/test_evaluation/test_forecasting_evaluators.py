import multiprocessing
import os
import queue
import sys
import unittest
import unittest.mock

from ConfigSpace import Configuration

import numpy as np

from smac.tae import StatusType

from autoPyTorch.automl_common.common.utils.backend import create
from autoPyTorch.datasets.resampling_strategy import CrossValTypes
from autoPyTorch.evaluation.time_series_forecasting_train_evaluator import TimeSeriesForecastingTrainEvaluator
from autoPyTorch.evaluation.utils import read_queue
from autoPyTorch.pipeline.components.training.metrics.metrics import mean_MASE_forecasting

this_directory = os.path.dirname(__file__)
sys.path.append(this_directory)
from evaluation_util import (  # noqa (E402: module level import not at top of file)
    BaseEvaluatorTest, get_binary_classification_datamanager,
    get_forecasting_dataset, get_multiclass_classification_datamanager,
    get_regression_datamanager)

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
    def test_budget_type_choices(self, pipeline_mock):
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

        budget_value = 0.1

        for budget_type in ['resolution', 'num_seq', 'num_sample_per_seq']:
            evaluator = TimeSeriesForecastingTrainEvaluator(backend_api,
                                                            queue_,
                                                            configuration=configuration,
                                                            metric=mean_MASE_forecasting, budget=0,
                                                            pipeline_options={'budget_type': budget_type,
                                                                              budget_type: 0.1},
                                                            min_num_test_instances=100)
            self.assertTrue('epochs' not in evaluator.fit_dictionary)
            if budget_type == 'resolution':
                self.assertTrue('sample_interval' in evaluator.fit_dictionary)
                self.assertEqual(int(np.ceil(1.0 / budget_value)), evaluator.fit_dictionary['sample_interval'])
            elif budget_type == 'num_seq':
                self.assertTrue('fraction_seq' in evaluator.fit_dictionary)
                self.assertEqual(budget_value, evaluator.fit_dictionary['fraction_seq'])
            if budget_type == 'num_sample_per_seq':
                self.assertTrue('fraction_samples_per_seq' in evaluator.fit_dictionary)
                self.assertEqual(budget_value, evaluator.fit_dictionary['fraction_samples_per_seq'])

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
        configuration.get_dictionary.return_value = {}
        backend_api = create(self.tmp_dir, self.output_dir, prefix='autoPyTorch')
        backend_api.load_datamanager = lambda: D
        queue_ = multiprocessing.Queue()

        evaluator = TimeSeriesForecastingTrainEvaluator(backend_api,
                                                        queue_,
                                                        configuration=configuration,
                                                        metric=mean_MASE_forecasting, budget=0,
                                                        pipeline_options={'budget_type': 'epochs', 'epochs': 50},
                                                        min_num_test_instances=100)
        self.assertTrue('epochs' in evaluator.fit_dictionary)
        evaluator.file_output = unittest.mock.Mock(spec=evaluator.file_output)
        evaluator.file_output.return_value = (None, {})

        evaluator.fit_predict_and_loss()

        rval = read_queue(evaluator.queue)

        self.assertEqual(len(rval), 1)
        result = rval[0]['loss']
        self.assertEqual(len(rval[0]), 3)

        self.assertRaises(queue.Empty, evaluator.queue.get, timeout=1)

        self.assertEqual(evaluator.file_output.call_count, 1)
        self.assertAlmostEqual(result, 4591.5, places=4)
        self.assertEqual(pipeline_mock.fit.call_count, 1)
        # As forecasting inference could be quite expensive, we only allow one opt prediction and test prediction
        self.assertEqual(pipeline_mock.predict.call_count, 2)

        self.assertEqual(evaluator.file_output.call_count, 1)
        self.assertEqual(evaluator.file_output.call_args[0][0].shape[0], len(D.splits[0][1]) * n_prediction_steps)
        self.assertIsNone(evaluator.file_output.call_args[0][1])

        self.assertEqual(evaluator.file_output.call_args[0][2].shape[0],
                         D.test_tensors[1].shape[0])
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
        configuration.get_dictionary.return_value = {}
        backend_api = create(self.tmp_dir, self.output_dir, prefix='autoPyTorch')
        backend_api.load_datamanager = lambda: D
        queue_ = multiprocessing.Queue()

        evaluator = TimeSeriesForecastingTrainEvaluator(backend_api,
                                                        queue_,
                                                        configuration=configuration,
                                                        metric=mean_MASE_forecasting, budget=0,
                                                        pipeline_options={'budget_type': 'epochs', 'epochs': 50})

        evaluator.file_output = unittest.mock.Mock(spec=evaluator.file_output)
        evaluator.file_output.return_value = (None, {})

        evaluator.fit_predict_and_loss()

        rval = read_queue(evaluator.queue)
        self.assertEqual(len(rval), 1)
        result = rval[0]['loss']
        self.assertEqual(len(rval[0]), 3)
        self.assertRaises(queue.Empty, evaluator.queue.get, timeout=1)

        self.assertEqual(evaluator.file_output.call_count, 1)
        self.assertAlmostEqual(result, 4590.06977, places=4)
        self.assertEqual(pipeline_mock.fit.call_count, 3)
        # 3 calls because of the 3 times validation evaluations, however, we only evaluate test target once
        self.assertEqual(pipeline_mock.predict.call_count, 4)
        # as the optimisation preds in cv is concatenation of the 3 folds,
        # so it is 3*splits
        self.assertEqual(evaluator.file_output.call_args[0][0].shape[0],
                         3 * len(D.splits[0][1]) * n_prediction_steps, evaluator.file_output.call_args)
        self.assertIsNone(evaluator.file_output.call_args[0][1])
        # we do not have test sets
        self.assertEqual(evaluator.file_output.call_args[0][2].shape[0],
                         D.test_tensors[1].shape[0])

        res = evaluator.file_output.call_args[0][0].reshape(-1, n_prediction_steps, evaluator.num_targets)
        assert np.all(res == 0.)

    @unittest.mock.patch('autoPyTorch.pipeline.time_series_forecasting.TimeSeriesForecastingPipeline')
    def test_proxy_val_set(self, pipeline_mock):
        pipeline_mock.fit_dictionary = {'budget_type': 'epochs', 'epochs': 0.1}
        D = get_forecasting_dataset(n_prediction_steps=5)
        n_prediction_steps = D.n_prediction_steps
        pipeline_mock.predict.side_effect = \
            lambda X, batch_size=None: np.tile([0.], (len(X), n_prediction_steps))
        pipeline_mock.side_effect = lambda **kwargs: pipeline_mock
        pipeline_mock.get_additional_run_info.return_value = None

        configuration = unittest.mock.Mock(spec=Configuration)
        configuration.get_dictionary.return_value = {}
        backend_api = create(self.tmp_dir, self.output_dir, prefix='autoPyTorch')
        backend_api.load_datamanager = lambda: D
        queue_ = multiprocessing.Queue()

        evaluator = TimeSeriesForecastingTrainEvaluator(backend_api,
                                                        queue_,
                                                        configuration=configuration,
                                                        metric=mean_MASE_forecasting, budget=0.3,
                                                        pipeline_options={'budget_type': 'epochs', 'epochs': 50},
                                                        min_num_test_instances=1)
        evaluator.file_output = unittest.mock.Mock(spec=evaluator.file_output)
        evaluator.file_output.return_value = (None, {})

        evaluator.fit_predict_and_loss()

        rval = read_queue(evaluator.queue)

        self.assertEqual(len(rval), 1)
        result = rval[0]['loss']

        self.assertAlmostEqual(result, 925.2, places=4)
        res = evaluator.file_output.call_args[0][0].reshape(-1, n_prediction_steps, evaluator.num_targets)

        n_evaluated_pip_mock = 0
        val_split = D.splits[0][1]

        for i_seq, seq_output in enumerate(res):
            if i_seq % 3 == 0 and n_evaluated_pip_mock < 3:
                n_evaluated_pip_mock += 1
                assert np.all(seq_output == 0.)
            else:
                # predict with dummy predictor
                assert np.all(seq_output == D.get_validation_set(val_split[i_seq])[-1][0]['past_targets'][-1].numpy())

    @unittest.mock.patch('autoPyTorch.pipeline.time_series_forecasting.TimeSeriesForecastingPipeline')
    @unittest.mock.patch('multiprocessing.Queue', )
    def test_finish_up(self, pipeline_mock, queue_mock):
        pipeline_mock.fit_dictionary = {'budget_type': 'epochs', 'epochs': 50}

        rs = np.random.RandomState(1)
        D = get_forecasting_dataset(n_prediction_steps=3)

        n_prediction_steps = D.n_prediction_steps

        pipeline_mock.predict.side_effect = \
            lambda X, batch_size=None: np.tile([0.], (len(X), n_prediction_steps))

        pipeline_mock.side_effect = lambda **kwargs: pipeline_mock
        pipeline_mock.get_additional_run_info.return_value = None

        configuration = unittest.mock.Mock(spec=Configuration)
        backend_api = create(self.tmp_dir, self.output_dir, prefix='autoPyTorch')
        backend_api.load_datamanager = lambda: D

        ae = TimeSeriesForecastingTrainEvaluator(backend_api,
                                                 queue_mock,
                                                 configuration=configuration,
                                                 metric=mean_MASE_forecasting, budget=0.3,
                                                 pipeline_options={'budget_type': 'epochs', 'epochs': 50},
                                                 min_num_test_instances=1)

        val_splits = D.splits[0][1]
        mase_val = ae.generate_mase_coefficient_for_validation(val_splits)

        ae.Y_optimization = rs.rand(len(val_splits) * n_prediction_steps, D.num_targets) * mase_val
        predictions_ensemble = rs.rand(len(val_splits) * n_prediction_steps, D.num_targets) * mase_val
        predictions_test = rs.rand(len(D.datasets) * n_prediction_steps, D.num_targets)

        metric_kwargs = {'sp': ae.seasonality,
                         'n_prediction_steps': ae.n_prediction_steps,
                         'mase_coefficient': ae.generate_mase_coefficient_for_test_set()}

        # NaNs in prediction ensemble
        ae.finish_up(
            loss={'mean_MASE_forecasting': 0.1},
            train_loss=None,
            opt_pred=predictions_ensemble,
            valid_pred=None,
            test_pred=predictions_test,
            additional_run_info=None,
            file_output=True,
            status=StatusType.SUCCESS,
            **metric_kwargs
        )
        self.assertTrue('test_loss' in queue_mock.put.call_args[0][0]['additional_run_info'])
