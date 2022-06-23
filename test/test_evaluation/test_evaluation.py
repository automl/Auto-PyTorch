import logging.handlers
import os
import shutil
import sys
import time
import unittest
import unittest.mock

import numpy as np

import pynisher

import pytest

from smac.runhistory.runhistory import RunInfo
from smac.stats.stats import Stats
from smac.tae import StatusType
from smac.utils.constants import MAXINT

from autoPyTorch.evaluation.tae import TargetAlgorithmQuery
from autoPyTorch.pipeline.components.training.metrics.metrics import accuracy, log_loss

this_directory = os.path.dirname(__file__)
sys.path.append(this_directory)
from evaluation_util import get_multiclass_classification_datamanager  # noqa E402


def safe_eval_success_mock(*args, **kwargs):
    queue = kwargs['queue']
    queue.put({'status': StatusType.SUCCESS,
               'loss': 0.5,
               'additional_run_info': ''})


class BackendMock(object):
    def __init__(self):
        self.temporary_directory = './.tmp_evaluation'
        try:
            os.mkdir(self.temporary_directory)
        except:  # noqa 3722
            pass

    def load_datamanager(self):
        return get_multiclass_classification_datamanager()


class EvaluationTest(unittest.TestCase):
    def setUp(self):
        self.datamanager = get_multiclass_classification_datamanager()
        self.tmp = os.path.join(os.getcwd(), '.test_evaluation')
        os.mkdir(self.tmp)
        self.logger_port = logging.handlers.DEFAULT_TCP_LOGGING_PORT
        scenario_mock = unittest.mock.Mock()
        scenario_mock.wallclock_limit = 10
        scenario_mock.algo_runs_timelimit = 1000
        scenario_mock.ta_run_limit = 100
        self.scenario = scenario_mock
        stats = Stats(scenario_mock)
        stats.start_timing()
        self.stats = stats
        self.taq_kwargs = dict(
            backend=BackendMock(),
            seed=1,
            stats=self.stats,
            multi_objectives=["cost"],
            memory_limit=3072,
            metric=accuracy,
            cost_for_crash=accuracy._cost_of_crash,
            abort_on_first_run_crash=False,
            logger_port=self.logger_port,
            pynisher_context='fork'
        )
        config = unittest.mock.Mock(spec=int)
        config.config_id, config.origin = 198, 'MOCK'
        self.runinfo_kwargs = dict(
            config=config,
            instance=None,
            instance_specific=None,
            seed=1,
            capped=False
        )

        try:
            shutil.rmtree(self.tmp)
        except Exception:
            pass

    def tearDown(self):
        try:
            shutil.rmtree(self.tmp)
        except Exception:
            pass

    ############################################################################
    # pynisher tests
    def test_pynisher_memory_error(self):
        def fill_memory():
            a = np.random.random_sample((10000000, 10000000)).astype(np.float64)
            return np.sum(a)

        safe_eval = pynisher.enforce_limits(mem_in_mb=1)(fill_memory)
        safe_eval()
        self.assertEqual(safe_eval.exit_status, pynisher.MemorylimitException)

    def test_pynisher_timeout(self):
        def run_over_time():
            time.sleep(2)

        safe_eval = pynisher.enforce_limits(wall_time_in_s=1,
                                            grace_period_in_s=0)(run_over_time)
        safe_eval()
        self.assertEqual(safe_eval.exit_status, pynisher.TimeoutException)

    ############################################################################
    # Test TargetAlgorithmQuery.run_wrapper()
    @unittest.mock.patch('autoPyTorch.evaluation.tae.eval_fn')
    def test_eval_with_limits_holdout(self, pynisher_mock):
        pynisher_mock.side_effect = safe_eval_success_mock
        ta = TargetAlgorithmQuery(**self.taq_kwargs)
        info = ta.run_wrapper(RunInfo(cutoff=30, **self.runinfo_kwargs))
        self.assertEqual(info[0].config.config_id, 198)
        self.assertEqual(info[1].status, StatusType.SUCCESS, info)
        self.assertEqual(info[1].cost, 0.5)
        self.assertIsInstance(info[1].time, float)

    @unittest.mock.patch('pynisher.enforce_limits')
    def test_cutoff_lower_than_remaining_time(self, pynisher_mock):
        ta = TargetAlgorithmQuery(**self.taq_kwargs)
        self.stats.ta_runs = 1
        ta.run_wrapper(RunInfo(cutoff=30, **self.runinfo_kwargs))
        self.assertEqual(pynisher_mock.call_args[1]['wall_time_in_s'], 4)
        self.assertIsInstance(pynisher_mock.call_args[1]['wall_time_in_s'], int)

    @unittest.mock.patch('pynisher.enforce_limits')
    def test_eval_with_limits_holdout_fail_timeout(self, pynisher_mock):
        m1 = unittest.mock.Mock()
        m2 = unittest.mock.Mock()
        m1.return_value = m2
        pynisher_mock.return_value = m1
        m2.exit_status = pynisher.TimeoutException
        m2.wall_clock_time = 30
        ta = TargetAlgorithmQuery(**self.taq_kwargs)
        info = ta.run_wrapper(RunInfo(cutoff=30, **self.runinfo_kwargs))
        self.assertEqual(info[1].status, StatusType.TIMEOUT)
        self.assertEqual(info[1].cost, 1.0)
        self.assertIsInstance(info[1].time, float)
        self.assertNotIn('exitcode', info[1].additional_info)

    @unittest.mock.patch('pynisher.enforce_limits')
    def test_zero_or_negative_cutoff(self, pynisher_mock):
        ta = TargetAlgorithmQuery(**self.taq_kwargs)
        self.scenario.wallclock_limit = 5
        self.stats.submitted_ta_runs += 1
        run_info, run_value = ta.run_wrapper(RunInfo(cutoff=9, **self.runinfo_kwargs))
        self.assertEqual(run_value.status, StatusType.STOP)

    @unittest.mock.patch('autoPyTorch.evaluation.tae.eval_fn')
    def test_eval_with_limits_holdout_fail_silent(self, pynisher_mock):
        config = unittest.mock.Mock()
        config.config_id, config.origin = 198, 'MOCK'
        runinfo_kwargs = self.runinfo_kwargs.copy()
        runinfo_kwargs['config'] = config
        pynisher_mock.return_value = None
        ta = TargetAlgorithmQuery(**self.taq_kwargs)

        # The following should not fail because abort on first config crashed is false
        info = ta.run_wrapper(RunInfo(cutoff=60, **runinfo_kwargs))
        self.assertEqual(info[1].status, StatusType.CRASHED)
        self.assertEqual(info[1].cost, 1.0)
        self.assertIsInstance(info[1].time, float)
        ans = {
            'configuration_origin': 'MOCK',
            'error': "Result queue is empty",
            'exit_status': '0',
            'exitcode': 0,
            'subprocess_stdout': '',
            'subprocess_stderr': ''
        }
        self.assertTrue(all(ans[key] == info[1].additional_info[key] for key in ans.keys()))

        self.stats.submitted_ta_runs += 1
        info = ta.run_wrapper(RunInfo(cutoff=30, **runinfo_kwargs))
        self.assertEqual(info[1].status, StatusType.CRASHED)
        self.assertEqual(info[1].cost, 1.0)
        self.assertIsInstance(info[1].time, float)
        self.assertTrue(all(ans[key] == info[1].additional_info[key] for key in ans.keys()))

    @unittest.mock.patch('autoPyTorch.evaluation.tae.eval_fn')
    def test_eval_with_limits_holdout_fail_memory_error(self, pynisher_mock):
        pynisher_mock.side_effect = MemoryError
        ta = TargetAlgorithmQuery(**self.taq_kwargs)
        info = ta.run_wrapper(RunInfo(cutoff=30, **self.runinfo_kwargs))
        self.assertEqual(info[1].status, StatusType.MEMOUT)

        # For accuracy, worst possible result is MAXINT
        worst_possible_result = 1
        self.assertEqual(info[1].cost, worst_possible_result)
        self.assertIsInstance(info[1].time, float)
        self.assertNotIn('exitcode', info[1].additional_info)

    @unittest.mock.patch('pynisher.enforce_limits')
    def test_eval_with_limits_holdout_timeout_with_results_in_queue(self, pynisher_mock):
        result_vals = [
            # Test for a succesful run
            {'status': StatusType.SUCCESS, 'loss': 0.5, 'additional_run_info': {}},
            # And a crashed run which is in the queue
            {'status': StatusType.CRASHED, 'loss': 2.0, 'additional_run_info': {}}
        ]
        m1 = unittest.mock.Mock()
        m2 = unittest.mock.Mock()
        m1.return_value = m2
        pynisher_mock.return_value = m1
        m2.exit_status = pynisher.TimeoutException
        m2.wall_clock_time = 30
        ans_loss = [0.5, 1.0]

        for results, ans in zip(result_vals, ans_loss):
            def side_effect(queue, evaluator_params, fixed_pipeline_params):
                queue.put(results)

            m2.side_effect = side_effect

            ta = TargetAlgorithmQuery(**self.taq_kwargs)
            info = ta.run_wrapper(RunInfo(cutoff=30, **self.runinfo_kwargs))
            self.assertEqual(info[1].status, results['status'])
            self.assertEqual(info[1].cost, ans)
            self.assertIsInstance(info[1].time, float)
            self.assertNotIn('exitcode', info[1].additional_info)

    @unittest.mock.patch('autoPyTorch.evaluation.tae.eval_fn')
    def test_eval_with_limits_holdout_2(self, eval_houldout_mock):
        def side_effect(queue, evaluator_params, fixed_pipeline_params):
            queue.put({'status': StatusType.SUCCESS,
                       'loss': 0.5,
                       'additional_run_info': evaluator_params.init_params['instance']})

        eval_houldout_mock.side_effect = side_effect
        ta = TargetAlgorithmQuery(**self.taq_kwargs)
        self.scenario.wallclock_limit = 180
        runinfo_kwargs = self.runinfo_kwargs.copy()
        runinfo_kwargs.update(instance="{'subsample': 30}")
        info = ta.run_wrapper(RunInfo(cutoff=30, **runinfo_kwargs))
        self.assertEqual(info[1].status, StatusType.SUCCESS, info)
        self.assertEqual(len(info[1].additional_info), 2)
        self.assertIn('configuration_origin', info[1].additional_info)
        self.assertEqual(info[1].additional_info['message'], "{'subsample': 30}")

    @unittest.mock.patch('autoPyTorch.evaluation.tae.eval_fn')
    def test_exception_in_target_function(self, eval_holdout_mock):
        eval_holdout_mock.side_effect = ValueError
        ta = TargetAlgorithmQuery(**self.taq_kwargs)
        self.stats.submitted_ta_runs += 1
        info = ta.run_wrapper(RunInfo(cutoff=30, **self.runinfo_kwargs))
        self.assertEqual(info[1].status, StatusType.CRASHED)
        self.assertEqual(info[1].cost, 1.0)
        self.assertIsInstance(info[1].time, float)
        self.assertEqual(info[1].additional_info['error'], 'ValueError()')
        self.assertIn('traceback', info[1].additional_info)
        self.assertNotIn('exitcode', info[1].additional_info)

    def test_silent_exception_in_target_function(self):
        ta = TargetAlgorithmQuery(**self.taq_kwargs)
        ta.pynisher_logger = unittest.mock.Mock()
        self.stats.submitted_ta_runs += 1
        info = ta.run_wrapper(RunInfo(cutoff=3000, **self.runinfo_kwargs))
        self.assertEqual(info[1].status, StatusType.CRASHED, msg=str(info[1].additional_info))
        self.assertEqual(info[1].cost, 1.0)
        self.assertIsInstance(info[1].time, float)
        self.assertIn(
            info[1].additional_info['error'],
            (
                """AttributeError("'BackendMock' object has no attribute """
                """'save_targets_ensemble'",)""",
                """AttributeError("'BackendMock' object has no attribute """
                """'save_targets_ensemble'")""",
                """AttributeError('save_targets_ensemble')"""
                """AttributeError("'BackendMock' object has no attribute """
                """'setup_logger'",)""",
                """AttributeError("'BackendMock' object has no attribute """
                """'setup_logger'")""",
            )
        )
        self.assertNotIn('exitcode', info[1].additional_info)
        self.assertNotIn('exit_status', info[1].additional_info)
        self.assertNotIn('traceback', info[1])

    def test_eval_with_simple_intensification(self):
        taq = TargetAlgorithmQuery(**self.taq_kwargs)
        taq.fixed_pipeline_params = taq.fixed_pipeline_params._replace(budget_type='runtime')
        taq.pynisher_logger = unittest.mock.Mock()

        run_info = RunInfo(cutoff=30, **self.runinfo_kwargs)

        for budget in [0.0, 50.0]:
            # Simple intensification always returns budget = 0
            # Other intensifications return a non-zero value
            self.stats.submitted_ta_runs += 1
            run_info = run_info._replace(budget=budget)
            run_info_out, _ = taq.run_wrapper(run_info)
            self.assertEqual(run_info_out.budget, budget)


@pytest.mark.parametrize("metric,expected", [(accuracy, 1.0), (log_loss, MAXINT)])
def test_cost_of_crash(metric, expected):
    assert metric._cost_of_crash == expected
