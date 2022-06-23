import multiprocessing
import os
import queue
import shutil
import sys
import unittest
import unittest.mock

from ConfigSpace import Configuration

import numpy as np

import pytest

from sklearn.base import BaseEstimator

from smac.tae import StatusType

from autoPyTorch.automl_common.common.utils.backend import create
from autoPyTorch.datasets.resampling_strategy import CrossValTypes, NoResamplingStrategyTypes
from autoPyTorch.evaluation.abstract_evaluator import EvaluatorParams, FixedPipelineParams
from autoPyTorch.evaluation.evaluator import (
    Evaluator,
    _CrossValidationResultsManager,
)
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


class TestCrossValidationResultsManager(unittest.TestCase):
    def test_update_loss_dict(self):
        cv_results = _CrossValidationResultsManager(3)
        loss_sum_dict = {}
        loss_dict = {'f1': 1.0, 'f2': 2.0}
        cv_results._update_loss_dict(loss_sum_dict, loss_dict, 3)
        assert loss_sum_dict == {'f1': 1.0 * 3, 'f2': 2.0 * 3}
        loss_sum_dict = {'f1': 2.0, 'f2': 1.0}
        cv_results._update_loss_dict(loss_sum_dict, loss_dict, 3)
        assert loss_sum_dict == {'f1': 2.0 + 1.0 * 3, 'f2': 1.0 + 2.0 * 3}

    def test_merge_predictions(self):
        cv_results = _CrossValidationResultsManager(3)
        preds = np.array([])
        assert cv_results._merge_predictions(preds) is None

        for preds_shape in [(10, ), (10, 10, )]:
            preds = np.random.random(preds_shape)
            with pytest.raises(ValueError):
                cv_results._merge_predictions(preds)

        preds = np.array([
            [
                [1.0, 2.0],
                [3.0, 4.0],
                [5.0, 6.0],
            ],
            [
                [7.0, 8.0],
                [9.0, 10.0],
                [11.0, 12.0],
            ]
        ])
        ans = np.array([
            [4.0, 5.0],
            [6.0, 7.0],
            [8.0, 9.0],
        ])
        assert np.allclose(ans, cv_results._merge_predictions(preds))


class TestEvaluator(BaseEvaluatorTest, unittest.TestCase):
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

        self.fixed_params = FixedPipelineParams.with_default_pipeline_config(
            backend=self.backend_mock,
            metric=accuracy,
            seed=0,
            pipeline_config={'budget_type': 'epochs', 'epochs': 50},
            all_supported_metrics=True
        )
        self.eval_params = EvaluatorParams(
            budget=0, configuration=unittest.mock.Mock(spec=Configuration)
        )

        self.tmp_dir = os.path.join(self.ev_path, 'tmp_dir')
        self.output_dir = os.path.join(self.ev_path, 'out_dir')

    def tearDown(self):
        if os.path.exists(self.ev_path):
            shutil.rmtree(self.ev_path)

    def _get_evaluator(self, pipeline_mock, data):
        pipeline_mock.predict_proba.side_effect = \
            lambda X, batch_size=None: np.tile([0.6, 0.4], (len(X), 1))
        pipeline_mock.side_effect = lambda **kwargs: pipeline_mock
        pipeline_mock.get_additional_run_info.return_value = None

        _queue = multiprocessing.Queue()
        backend_api = create(self.tmp_dir, self.output_dir, prefix='autoPyTorch')
        backend_api.load_datamanager = lambda: data

        fixed_params_dict = self.fixed_params._asdict()
        fixed_params_dict.update(backend=backend_api)
        evaluator = Evaluator(
            queue=_queue,
            fixed_pipeline_params=FixedPipelineParams(**fixed_params_dict),
            evaluator_params=self.eval_params
        )
        evaluator._save_to_backend = unittest.mock.Mock(spec=evaluator._save_to_backend)
        evaluator._save_to_backend.return_value = True

        evaluator.evaluate_loss()

        return evaluator

    def _check_results(self, evaluator, ans):
        rval = read_queue(evaluator.queue)
        self.assertEqual(len(rval), 1)
        result = rval[0]['loss']
        self.assertEqual(len(rval[0]), 3)
        self.assertRaises(queue.Empty, evaluator.queue.get, timeout=1)
        self.assertEqual(result, ans)
        self.assertEqual(evaluator._save_to_backend.call_count, 1)

    def _check_whether_save_y_opt_is_correct(self, resampling_strategy, ans):
        backend_api = create(self.tmp_dir, self.output_dir, prefix='autoPyTorch')
        D = get_binary_classification_datamanager(resampling_strategy)
        backend_api.load_datamanager = lambda: D
        fixed_params_dict = self.fixed_params._asdict()
        fixed_params_dict.update(backend=backend_api, save_y_opt=True)
        evaluator = Evaluator(
            queue=multiprocessing.Queue(),
            fixed_pipeline_params=FixedPipelineParams(**fixed_params_dict),
            evaluator_params=self.eval_params
        )
        assert evaluator.fixed_pipeline_params.save_y_opt == ans

    def test_whether_save_y_opt_is_correct_for_no_resampling(self):
        self._check_whether_save_y_opt_is_correct(NoResamplingStrategyTypes.no_resampling, False)

    def test_whether_save_y_opt_is_correct_for_resampling(self):
        self._check_whether_save_y_opt_is_correct(CrossValTypes.k_fold_cross_validation, True)

    def test_evaluate_loss(self):
        D = get_binary_classification_datamanager()
        backend_api = create(self.tmp_dir, self.output_dir, prefix='autoPyTorch')
        backend_api.load_datamanager = lambda: D
        fixed_params_dict = self.fixed_params._asdict()
        fixed_params_dict.update(backend=backend_api)
        evaluator = Evaluator(
            queue=multiprocessing.Queue(),
            fixed_pipeline_params=FixedPipelineParams(**fixed_params_dict),
            evaluator_params=self.eval_params
        )
        evaluator.splits = None
        with pytest.raises(ValueError):
            evaluator.evaluate_loss()

    @unittest.mock.patch('autoPyTorch.pipeline.tabular_classification.TabularClassificationPipeline')
    def test_holdout(self, pipeline_mock):
        D = get_binary_classification_datamanager()
        evaluator = self._get_evaluator(pipeline_mock, D)
        self._check_results(evaluator, ans=0.5652173913043479)

        self.assertEqual(pipeline_mock.fit.call_count, 1)
        # 3 calls because of train, holdout and test set
        self.assertEqual(pipeline_mock.predict_proba.call_count, 3)
        call_args = evaluator._save_to_backend.call_args
        self.assertEqual(call_args[0][0].shape[0], len(D.splits[0][1]))
        self.assertIsNone(call_args[0][1])
        self.assertEqual(call_args[0][2].shape[0], D.test_tensors[1].shape[0])
        self.assertEqual(evaluator.pipelines[0].fit.call_count, 1)

    @unittest.mock.patch('autoPyTorch.pipeline.tabular_classification.TabularClassificationPipeline')
    def test_cv(self, pipeline_mock):
        D = get_binary_classification_datamanager(resampling_strategy=CrossValTypes.k_fold_cross_validation)
        evaluator = self._get_evaluator(pipeline_mock, D)
        self._check_results(evaluator, ans=0.463768115942029)

        self.assertEqual(pipeline_mock.fit.call_count, 5)
        # 15 calls because of the training, holdout and
        # test set (3 sets x 5 folds = 15)
        self.assertEqual(pipeline_mock.predict_proba.call_count, 15)
        call_args = evaluator._save_to_backend.call_args
        # as the optimisation preds in cv is concatenation of the 5 folds,
        # so it is 5*splits
        self.assertEqual(call_args[0][0].shape[0],
                         # Notice this - 1: It is because the dataset D
                         # has shape ((69, )) which is not divisible by 5
                         5 * len(D.splits[0][1]) - 1, call_args)
        self.assertIsNone(call_args[0][1])
        self.assertEqual(call_args[0][2].shape[0],
                         D.test_tensors[1].shape[0])

    @unittest.mock.patch('autoPyTorch.pipeline.tabular_classification.TabularClassificationPipeline')
    def test_no_resampling(self, pipeline_mock):
        D = get_binary_classification_datamanager(NoResamplingStrategyTypes.no_resampling)
        evaluator = self._get_evaluator(pipeline_mock, D)
        self._check_results(evaluator, ans=0.5806451612903225)

        self.assertEqual(pipeline_mock.fit.call_count, 1)
        # 2 calls because of train and test set
        self.assertEqual(pipeline_mock.predict_proba.call_count, 2)
        call_args = evaluator._save_to_backend.call_args
        self.assertIsNone(D.splits[0][1])
        self.assertIsNone(call_args[0][1])
        self.assertEqual(call_args[0][2].shape[0], D.test_tensors[1].shape[0])
        self.assertEqual(evaluator.pipelines[0].fit.call_count, 1)

    @unittest.mock.patch.object(Evaluator, '_loss')
    def test_save_to_backend(self, loss_mock):
        call_counter = 0
        no_resample_counter = 0
        for rs in [None, NoResamplingStrategyTypes.no_resampling]:
            no_resampling = isinstance(rs, NoResamplingStrategyTypes)
            D = get_regression_datamanager() if rs is None else get_regression_datamanager(rs)
            D.name = 'test'
            self.backend_mock.load_datamanager.return_value = D
            _queue = multiprocessing.Queue()
            loss_mock.return_value = None

            evaluator = Evaluator(
                queue=_queue,
                fixed_pipeline_params=self.fixed_params,
                evaluator_params=self.eval_params
            )
            evaluator.y_opt = D.train_tensors[1]
            key_ans = {'seed', 'idx', 'budget', 'model', 'cv_model',
                       'ensemble_predictions', 'valid_predictions', 'test_predictions'}

            for pl in [['model'], ['model2', 'model2']]:
                call_counter += 1
                no_resample_counter += no_resampling
                self.backend_mock.get_model_dir.return_value = True
                evaluator.pipelines = pl
                self.assertTrue(evaluator._save_to_backend(D.train_tensors[1], None, D.test_tensors[1]))
                call_list = self.backend_mock.save_numrun_to_dir.call_args_list[-1][1]

                self.assertEqual(self.backend_mock.save_targets_ensemble.call_count, call_counter - no_resample_counter)
                self.assertEqual(self.backend_mock.save_numrun_to_dir.call_count, call_counter)
                self.assertEqual(call_list.keys(), key_ans)
                self.assertIsNotNone(call_list['model'])
                if len(pl) > 1:  # ==> cross validation
                    # self.assertIsNotNone(call_list['cv_model'])
                    # TODO: Reflect the ravin's opinion
                    pass
                else:  # holdout ==> single thus no cv_model
                    self.assertIsNone(call_list['cv_model'])

            # Check for not containing NaNs - that the models don't predict nonsense
            # for unseen data
            D.train_tensors[1][0] = np.NaN
            self.assertFalse(evaluator._save_to_backend(D.train_tensors[1], None, D.test_tensors[1]))

    @unittest.mock.patch('autoPyTorch.pipeline.tabular_classification.TabularClassificationPipeline')
    def test_predict_proba_binary_classification(self, mock):
        D = get_binary_classification_datamanager()
        self.backend_mock.load_datamanager.return_value = D
        mock.predict_proba.side_effect = lambda y, batch_size=None: np.array(
            [[0.1, 0.9]] * y.shape[0]
        )
        mock.side_effect = lambda **kwargs: mock

        _queue = multiprocessing.Queue()

        evaluator = Evaluator(
            queue=_queue,
            fixed_pipeline_params=self.fixed_params,
            evaluator_params=self.eval_params
        )

        evaluator.evaluate_loss()
        Y_optimization_pred = self.backend_mock.save_numrun_to_dir.call_args_list[0][1][
            'ensemble_predictions']

        for i in range(7):
            self.assertEqual(0.9, Y_optimization_pred[i][1])

    @unittest.mock.patch('autoPyTorch.pipeline.tabular_classification.TabularClassificationPipeline')
    def test_predict_proba_binary_classification_no_resampling(self, mock):
        D = get_binary_classification_datamanager(NoResamplingStrategyTypes.no_resampling)
        self.backend_mock.load_datamanager.return_value = D
        mock.predict_proba.side_effect = lambda y, batch_size=None: np.array(
            [[0.1, 0.9]] * y.shape[0]
        )
        mock.side_effect = lambda **kwargs: mock
        backend_api = create(self.tmp_dir, self.output_dir, prefix='autoPyTorch')
        backend_api.load_datamanager = lambda: D

        fixed_params_dict = self.fixed_params._asdict()
        fixed_params_dict.update(backend=backend_api)

        _queue = multiprocessing.Queue()

        evaluator = Evaluator(
            queue=_queue,
            fixed_pipeline_params=self.fixed_params,
            evaluator_params=self.eval_params
        )
        evaluator.evaluate_loss()
        Y_test_pred = self.backend_mock.save_numrun_to_dir.call_args_list[0][-1][
            'ensemble_predictions']

        for i in range(7):
            self.assertEqual(0.9, Y_test_pred[i][1])

    def test_get_results(self):
        _queue = multiprocessing.Queue()
        for i in range(5):
            _queue.put((i * 1, 1 - (i * 0.2), 0, "", StatusType.SUCCESS))
        result = read_queue(_queue)
        self.assertEqual(len(result), 5)
        self.assertEqual(result[0][0], 0)
        self.assertAlmostEqual(result[0][1], 1.0)

    @unittest.mock.patch('autoPyTorch.pipeline.tabular_classification.TabularClassificationPipeline')
    def test_additional_metrics_during_training(self, pipeline_mock):
        pipeline_mock.fit_dictionary = self.fixed_params.pipeline_config
        # Binary iris, contains 69 train samples, 31 test samples
        D = get_binary_classification_datamanager()
        pipeline_mock.predict_proba.side_effect = \
            lambda X, batch_size=None: np.tile([0.6, 0.4], (len(X), 1))
        pipeline_mock.side_effect = lambda **kwargs: pipeline_mock
        pipeline_mock.get_additional_run_info.return_value = None

        _queue = multiprocessing.Queue()
        backend_api = create(self.tmp_dir, self.output_dir, prefix='autoPyTorch')
        backend_api.load_datamanager = lambda: D

        fixed_params_dict = self.fixed_params._asdict()
        fixed_params_dict.update(backend=backend_api)
        evaluator = Evaluator(
            queue=_queue,
            fixed_pipeline_params=FixedPipelineParams(**fixed_params_dict),
            evaluator_params=self.eval_params
        )
        evaluator._save_to_backend = unittest.mock.Mock(spec=evaluator._save_to_backend)
        evaluator._save_to_backend.return_value = True

        evaluator.evaluate_loss()

        rval = read_queue(evaluator.queue)
        self.assertEqual(len(rval), 1)
        result = rval[0]
        self.assertIn('additional_run_info', result)
        self.assertIn('opt_loss', result['additional_run_info'])
        self.assertGreater(len(result['additional_run_info']['opt_loss'].keys()), 1)
