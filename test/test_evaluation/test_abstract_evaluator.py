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
from autoPyTorch.evaluation.abstract_evaluator import (
    AbstractEvaluator,
    EvaluationResults,
    EvaluatorParams,
    FixedPipelineParams
)
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
        self.eval_params = EvaluatorParams.with_default_budget(budget=0, configuration=1)
        self.fixed_params = FixedPipelineParams.with_default_pipeline_config(
            backend=self.backend_mock,
            save_y_opt=False,
            metric=accuracy,
            seed=1
        )

        self.working_directory = os.path.join(this_directory, '.tmp_%s' % self.id())

    def tearDown(self):
        if os.path.exists(self.ev_path):
            try:
                os.rmdir(self.ev_path)
            except:  # noqa E722
                pass

    def test_record_evaluation_model_predicts_NaN(self):
        '''Tests by handing in predictions which contain NaNs'''
        rs = np.random.RandomState(1)
        queue_mock = unittest.mock.Mock()
        opt_pred, test_pred, valid_pred = rs.rand(33, 3), rs.rand(25, 3), rs.rand(25, 3)
        ae = AbstractEvaluator(
            queue=queue_mock,
            fixed_pipeline_params=self.fixed_params,
            evaluator_params=self.eval_params
        )
        ae.y_opt = rs.rand(33, 3)

        for inference_name, pred in [('optimization', opt_pred), ('validation', valid_pred), ('test', test_pred)]:
            pred[5, 2] = np.nan
            results = EvaluationResults(
                opt_loss={'accuracy': 0.1},
                train_loss={'accuracy': 0.1},
                opt_pred=opt_pred,
                valid_pred=valid_pred,
                test_pred=test_pred,
                additional_run_info=None,
                status=StatusType.SUCCESS,
            )
            ae.fixed_pipeline_params.backend.save_numrun_to_dir = unittest.mock.Mock()
            ae.record_evaluation(results=results)
            self.assertEqual(ae.fixed_pipeline_params.backend.save_numrun_to_dir.call_count, 0)
            pred[5, 2] = 0.5

        self.assertEqual(self.backend_mock.save_predictions_as_npy.call_count, 0)

    def test_disable_file_output(self):
        queue_mock = unittest.mock.Mock()

        rs = np.random.RandomState(1)
        opt_pred, test_pred, valid_pred = rs.rand(33, 3), rs.rand(25, 3), rs.rand(25, 3)

        fixed_params_dict = self.fixed_params._asdict()

        for call_count, disable in enumerate(['all', 'model', 'cv_model', 'y_opt']):
            fixed_params_dict.update(disable_file_output=[disable])
            ae = AbstractEvaluator(
                queue=queue_mock,
                fixed_pipeline_params=FixedPipelineParams(**fixed_params_dict),
                evaluator_params=self.eval_params
            )
            ae.y_opt = opt_pred
            ae.pipelines = [unittest.mock.Mock()]

            if ae._is_output_possible(opt_pred, valid_pred, test_pred):
                ae._save_to_backend(opt_pred, valid_pred, test_pred)

            self.assertEqual(self.backend_mock.save_numrun_to_dir.call_count, call_count)
            if disable == 'all':
                continue

            call_list = self.backend_mock.save_numrun_to_dir.call_args_list[-1][1]
            if disable == 'model':  # TODO: Check the response from Ravin (add CV version?)
                self.assertIsNone(call_list['model'])
                # self.assertIsNotNone(call_list['cv_model'])
            elif disable == 'cv_model':
                # self.assertIsNotNone(call_list['model'])
                self.assertIsNone(call_list['cv_model'])

            if disable in ('y_opt', 'all'):
                self.assertIsNone(call_list['ensemble_predictions'])
            else:
                self.assertIsNotNone(call_list['ensemble_predictions'])

            self.assertIsNotNone(call_list['valid_predictions'])
            self.assertIsNotNone(call_list['test_predictions'])

    def test_save_to_backend(self):
        shutil.rmtree(self.working_directory, ignore_errors=True)
        os.mkdir(self.working_directory)

        queue_mock = unittest.mock.Mock()
        rs = np.random.RandomState(1)
        opt_pred, test_pred, valid_pred = rs.rand(33, 3), rs.rand(25, 3), rs.rand(25, 3)

        context = BackendContext(
            prefix='autoPyTorch',
            temporary_directory=os.path.join(self.working_directory, 'tmp'),
            output_directory=os.path.join(self.working_directory, 'out'),
            delete_tmp_folder_after_terminate=True,
            delete_output_folder_after_terminate=True,
        )
        with unittest.mock.patch.object(Backend, 'load_datamanager') as load_datamanager_mock:
            load_datamanager_mock.return_value = get_multiclass_classification_datamanager()

            fixed_params_dict = self.fixed_params._asdict()
            fixed_params_dict.update(backend=Backend(context, prefix='autoPyTorch'))

            ae = AbstractEvaluator(
                queue=queue_mock,
                fixed_pipeline_params=FixedPipelineParams(**fixed_params_dict),
                evaluator_params=EvaluatorParams.with_default_budget(choice='dummy', configuration=1)
            )
            ae.model = sklearn.dummy.DummyClassifier()
            ae.y_opt = rs.rand(33, 3)
            ae._save_to_backend(opt_pred=opt_pred, valid_pred=valid_pred, test_pred=test_pred)

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

            try:
                fixed_params_dict = self.fixed_params._asdict()
                fixed_params_dict.update(
                    backend=Backend(context, prefix='autoPyTorch'),
                    pipeline_config={'budget_type': "error", 'error': 0}
                )
                AbstractEvaluator(
                    queue=queue_mock,
                    fixed_pipeline_params=FixedPipelineParams(**fixed_params_dict),
                    evaluator_params=self.eval_params
                )
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

            fixed_params_dict = self.fixed_params._asdict()
            fixed_params_dict.update(
                backend=Backend(context, prefix='autoPyTorch'),
                disable_file_output=['model']
            )

            try:
                AbstractEvaluator(
                    queue=queue_mock,
                    evaluator_params=self.eval_params,
                    fixed_pipeline_params=FixedPipelineParams(**fixed_params_dict)
                )
            except Exception as e:
                self.assertIsInstance(e, ValueError)

            shutil.rmtree(self.working_directory, ignore_errors=True)
