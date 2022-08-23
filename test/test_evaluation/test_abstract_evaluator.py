# -*- encoding: utf-8 -*-
import os
import shutil
import sys
import unittest
import unittest.mock

import numpy as np

import sklearn.dummy

from smac.tae import StatusType

from autoPyTorch.automl_common.common.utils.backend import Backend, BackendContext
from autoPyTorch.evaluation.abstract_evaluator import AbstractEvaluator
from autoPyTorch.evaluation.utils import DisableFileOutputParameters
from autoPyTorch.pipeline.components.training.metrics.metrics import accuracy

this_directory = os.path.dirname(__file__)
sys.path.append(this_directory)
from evaluation_util import get_multiclass_classification_datamanager  # noqa E402


class AbstractEvaluatorTest(unittest.TestCase):
    _multiprocess_can_split_ = True

    def setUp(self):
        """
        Creates a backend mock
        """
        self.ev_path = os.path.join(this_directory, '.tmp_evaluation')
        if not os.path.exists(self.ev_path):
            os.mkdir(self.ev_path)
        dummy_model_files = [os.path.join(self.ev_path, str(n)) for n in range(100)]
        dummy_pred_files = [os.path.join(self.ev_path, str(n)) for n in range(100, 200)]

        backend_mock = unittest.mock.Mock()
        backend_mock.get_model_dir.return_value = self.ev_path
        backend_mock.get_model_path.side_effect = dummy_model_files
        backend_mock.get_prediction_output_path.side_effect = dummy_pred_files
        backend_mock.temporary_directory = self.ev_path

        D = get_multiclass_classification_datamanager()
        backend_mock.load_datamanager.return_value = D
        self.backend_mock = backend_mock

        self.working_directory = os.path.join(this_directory, '.tmp_%s' % self.id())

    def tearDown(self):
        if os.path.exists(self.ev_path):
            try:
                os.rmdir(self.ev_path)
            except:  # noqa E722
                pass

    def test_finish_up_model_predicts_NaN(self):
        '''Tests by handing in predictions which contain NaNs'''
        rs = np.random.RandomState(1)

        queue_mock = unittest.mock.Mock()
        ae = AbstractEvaluator(backend=self.backend_mock,
                               output_y_hat_optimization=False,
                               queue=queue_mock, metric=accuracy, budget=0,
                               configuration=1)
        ae.Y_optimization = rs.rand(33, 3)
        predictions_ensemble = rs.rand(33, 3)
        predictions_test = rs.rand(25, 3)
        predictions_valid = rs.rand(25, 3)

        # NaNs in prediction ensemble
        predictions_ensemble[5, 2] = np.NaN
        _, loss, _, additional_run_info = ae.finish_up(
            loss={'accuracy': 0.1},
            train_loss={'accuracy': 0.1},
            opt_pred=predictions_ensemble,
            valid_pred=predictions_valid,
            test_pred=predictions_test,
            additional_run_info=None,
            file_output=True,
            status=StatusType.SUCCESS,
        )
        self.assertEqual(loss, 1.0)
        self.assertEqual(additional_run_info,
                         {'error': 'Model predictions for optimization set '
                                   'contains NaNs.'})

        # NaNs in prediction validation
        predictions_ensemble[5, 2] = 0.5
        predictions_valid[5, 2] = np.NaN
        _, loss, _, additional_run_info = ae.finish_up(
            loss={'accuracy': 0.1},
            train_loss={'accuracy': 0.1},
            opt_pred=predictions_ensemble,
            valid_pred=predictions_valid,
            test_pred=predictions_test,
            additional_run_info=None,
            file_output=True,
            status=StatusType.SUCCESS,
        )
        self.assertEqual(loss, 1.0)
        self.assertEqual(additional_run_info,
                         {'error': 'Model predictions for validation set '
                                   'contains NaNs.'})

        # NaNs in prediction test
        predictions_valid[5, 2] = 0.5
        predictions_test[5, 2] = np.NaN
        _, loss, _, additional_run_info = ae.finish_up(
            loss={'accuracy': 0.1},
            train_loss={'accuracy': 0.1},
            opt_pred=predictions_ensemble,
            valid_pred=predictions_valid,
            test_pred=predictions_test,
            additional_run_info=None,
            file_output=True,
            status=StatusType.SUCCESS,
        )
        self.assertEqual(loss, 1.0)
        self.assertEqual(additional_run_info,
                         {'error': 'Model predictions for test set contains '
                                   'NaNs.'})

        self.assertEqual(self.backend_mock.save_predictions_as_npy.call_count, 0)

    def test_disable_file_output(self):
        queue_mock = unittest.mock.Mock()

        rs = np.random.RandomState(1)

        ae = AbstractEvaluator(
            backend=self.backend_mock,
            queue=queue_mock,
            disable_file_output=[DisableFileOutputParameters.all],
            metric=accuracy,
            logger_port=unittest.mock.Mock(),
            budget=0,
            configuration=1
        )
        ae.pipeline = unittest.mock.Mock()
        predictions_ensemble = rs.rand(33, 3)
        predictions_test = rs.rand(25, 3)
        predictions_valid = rs.rand(25, 3)

        loss_, additional_run_info_ = (
            ae.file_output(
                predictions_ensemble,
                predictions_valid,
                predictions_test,
            )
        )

        self.assertIsNone(loss_)
        self.assertEqual(additional_run_info_, {})
        # This function is never called as there is a return before
        self.assertEqual(self.backend_mock.save_numrun_to_dir.call_count, 0)

        for call_count, disable in enumerate(['pipeline', 'pipelines'], start=1):
            ae = AbstractEvaluator(
                backend=self.backend_mock,
                output_y_hat_optimization=False,
                queue=queue_mock,
                disable_file_output=[disable],
                metric=accuracy,
                budget=0,
                configuration=1
            )
            ae.Y_optimization = predictions_ensemble
            ae.pipeline = unittest.mock.Mock()
            ae.pipelines = [unittest.mock.Mock()]

            loss_, additional_run_info_ = (
                ae.file_output(
                    predictions_ensemble,
                    predictions_valid,
                    predictions_test,
                )
            )

            self.assertIsNone(loss_)
            self.assertEqual(additional_run_info_, {})
            self.assertEqual(self.backend_mock.save_numrun_to_dir.call_count, call_count)
            if disable == 'pipeline':
                self.assertIsNone(
                    self.backend_mock.save_numrun_to_dir.call_args_list[-1][1]['model'])
                self.assertIsNotNone(
                    self.backend_mock.save_numrun_to_dir.call_args_list[-1][1]['cv_model'])
            else:
                self.assertIsNotNone(
                    self.backend_mock.save_numrun_to_dir.call_args_list[-1][1]['model'])
                self.assertIsNone(
                    self.backend_mock.save_numrun_to_dir.call_args_list[-1][1]['cv_model'])
            self.assertIsNotNone(
                self.backend_mock.save_numrun_to_dir.call_args_list[-1][1][
                    'ensemble_predictions']
            )
            self.assertIsNotNone(
                self.backend_mock.save_numrun_to_dir.call_args_list[-1][1][
                    'valid_predictions']
            )
            self.assertIsNotNone(
                self.backend_mock.save_numrun_to_dir.call_args_list[-1][1][
                    'test_predictions']
            )

        ae = AbstractEvaluator(
            backend=self.backend_mock,
            output_y_hat_optimization=False,
            queue=queue_mock,
            metric=accuracy,
            disable_file_output=['y_optimization'],
            budget=0,
            configuration=1
        )
        ae.Y_optimization = predictions_ensemble
        ae.pipeline = 'pipeline'
        ae.pipelines = [unittest.mock.Mock()]

        loss_, additional_run_info_ = (
            ae.file_output(
                predictions_ensemble,
                predictions_valid,
                predictions_test,
            )
        )

        self.assertIsNone(loss_)
        self.assertEqual(additional_run_info_, {})

        self.assertIsNone(
            self.backend_mock.save_numrun_to_dir.call_args_list[-1][1][
                'ensemble_predictions']
        )
        self.assertIsNotNone(
            self.backend_mock.save_numrun_to_dir.call_args_list[-1][1][
                'valid_predictions']
        )
        self.assertIsNotNone(
            self.backend_mock.save_numrun_to_dir.call_args_list[-1][1][
                'test_predictions']
        )

    def test_file_output(self):
        shutil.rmtree(self.working_directory, ignore_errors=True)
        os.mkdir(self.working_directory)

        queue_mock = unittest.mock.Mock()

        context = BackendContext(
            prefix='autoPyTorch',
            temporary_directory=os.path.join(self.working_directory, 'tmp'),
            output_directory=os.path.join(self.working_directory, 'out'),
            delete_tmp_folder_after_terminate=True,
            delete_output_folder_after_terminate=True,
        )
        with unittest.mock.patch.object(Backend, 'load_datamanager') as load_datamanager_mock:
            load_datamanager_mock.return_value = get_multiclass_classification_datamanager()

            backend = Backend(context, prefix='autoPyTorch')

            ae = AbstractEvaluator(
                backend=backend,
                output_y_hat_optimization=False,
                queue=queue_mock,
                metric=accuracy,
                budget=0,
                configuration=1
            )
            ae.model = sklearn.dummy.DummyClassifier()

            rs = np.random.RandomState()
            ae.Y_optimization = rs.rand(33, 3)
            predictions_ensemble = rs.rand(33, 3)
            predictions_test = rs.rand(25, 3)
            predictions_valid = rs.rand(25, 3)

            ae.file_output(
                Y_optimization_pred=predictions_ensemble,
                Y_valid_pred=predictions_valid,
                Y_test_pred=predictions_test,
            )

            self.assertTrue(os.path.exists(os.path.join(self.working_directory, 'tmp',
                                                        '.autoPyTorch', 'runs', '1_0_1.0')))

            shutil.rmtree(self.working_directory, ignore_errors=True)

    def test_error_unsupported_budget_type(self):
        shutil.rmtree(self.working_directory, ignore_errors=True)
        os.mkdir(self.working_directory)

        queue_mock = unittest.mock.Mock()

        context = BackendContext(
            prefix='autoPyTorch',
            temporary_directory=os.path.join(self.working_directory, 'tmp'),
            output_directory=os.path.join(self.working_directory, 'out'),
            delete_tmp_folder_after_terminate=True,
            delete_output_folder_after_terminate=True,
        )
        with unittest.mock.patch.object(Backend, 'load_datamanager') as load_datamanager_mock:
            load_datamanager_mock.return_value = get_multiclass_classification_datamanager()

            backend = Backend(context, prefix='autoPyTorch')

            try:
                AbstractEvaluator(
                    backend=backend,
                    output_y_hat_optimization=False,
                    queue=queue_mock,
                    pipeline_options={'budget_type': "error", 'error': 0},
                    metric=accuracy,
                    budget=0,
                    configuration=1)
            except Exception as e:
                self.assertIsInstance(e, ValueError)

            shutil.rmtree(self.working_directory, ignore_errors=True)

    def test_error_unsupported_disable_file_output_parameters(self):
        shutil.rmtree(self.working_directory, ignore_errors=True)
        os.mkdir(self.working_directory)

        queue_mock = unittest.mock.Mock()

        context = BackendContext(
            prefix='autoPyTorch',
            temporary_directory=os.path.join(self.working_directory, 'tmp'),
            output_directory=os.path.join(self.working_directory, 'out'),
            delete_tmp_folder_after_terminate=True,
            delete_output_folder_after_terminate=True,
        )
        with unittest.mock.patch.object(Backend, 'load_datamanager') as load_datamanager_mock:
            load_datamanager_mock.return_value = get_multiclass_classification_datamanager()

            backend = Backend(context, prefix='autoPyTorch')

            try:
                AbstractEvaluator(
                    backend=backend,
                    output_y_hat_optimization=False,
                    queue=queue_mock,
                    metric=accuracy,
                    budget=0,
                    configuration=1,
                    disable_file_output=['model'])
            except Exception as e:
                self.assertIsInstance(e, ValueError)

            shutil.rmtree(self.working_directory, ignore_errors=True)
