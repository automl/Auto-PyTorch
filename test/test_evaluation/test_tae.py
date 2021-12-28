import queue
import unittest.mock

import numpy as np

import pytest

from smac.runhistory.runhistory import RunInfo, RunValue
from smac.tae import StatusType, TAEAbortException

from autoPyTorch.evaluation.tae import (
    PynisherFunctionWrapperLikeType,
    TargetAlgorithmQuery,
    _exception_handling,
    _get_eval_fn,
    _get_logger,
    _process_exceptions
)
from autoPyTorch.metrics import accuracy


def test_pynisher_function_wrapper_like_type_init():
    with pytest.raises(RuntimeError):
        PynisherFunctionWrapperLikeType(lambda: None)


def test_get_eval_fn():
    return_value = 'test_func'
    fn = _get_eval_fn(cost_for_crash=1e9, target_algorithm=lambda: return_value)
    assert fn() == return_value


def test_get_logger():
    name = 'test_logger'
    logger = _get_logger(logger_port=None, logger_name=name)
    assert logger.name == name


@pytest.mark.parametrize('is_anything_exception,ans', (
    (True, StatusType.CRASHED),
    (False, StatusType.SUCCESS)
))
def test_exception_handling(is_anything_exception, ans):
    obj = unittest.mock.Mock()
    obj.exit_status = 1
    info = {
        'loss': 1.0,
        'status': StatusType.SUCCESS,
        'additional_run_info': {}
    }
    q = queue.Queue()
    q.put(info)

    _, status, _, _ = _exception_handling(
        obj=obj,
        queue=q,
        info_msg='dummy',
        info_for_empty={},
        status=StatusType.DONOTADVANCE,
        is_anything_exception=is_anything_exception,
        worst_possible_result=1e9
    )
    assert status == ans


def test_process_exceptions():
    obj = unittest.mock.Mock()
    q = unittest.mock.Mock()
    obj.exit_status = TAEAbortException
    _, _, _, info = _process_exceptions(obj=obj, queue=q, budget=1.0, worst_possible_result=1e9)
    assert info['error'] == 'Your configuration of autoPyTorch did not work'

    obj.exit_status = 0
    info = {
        'loss': 1.0,
        'status': StatusType.DONOTADVANCE,
        'additional_run_info': {}
    }
    q = queue.Queue()
    q.put(info)

    _, status, _, _ = _process_exceptions(obj=obj, queue=q, budget=0, worst_possible_result=1e9)
    assert status == StatusType.SUCCESS
    _, _, _, info = _process_exceptions(obj=obj, queue=q, budget=0, worst_possible_result=1e9)
    assert 'empty' in info.get('error', 'no error')


def _create_taq():
    return TargetAlgorithmQuery(
        backend=unittest.mock.Mock(),
        seed=1,
        metric=accuracy,
        cost_for_crash=accuracy._cost_of_crash,
        abort_on_first_run_crash=True,
        pynisher_context=unittest.mock.Mock()
    )


class TestTargetAlgorithmQuery(unittest.TestCase):
    def test_check_run_info(self):
        taq = _create_taq()
        run_info = unittest.mock.Mock()
        run_info.budget = -1
        with pytest.raises(ValueError):
            taq._check_run_info(run_info)

    def test_cutoff_update_in_run_wrapper(self):
        taq = _create_taq()
        run_info = RunInfo(
            config=unittest.mock.Mock(),
            instance=None,
            instance_specific='dummy',
            seed=0,
            cutoff=8,
            capped=False,
            budget=1,
        )
        run_info._replace()
        taq.stats = unittest.mock.Mock()
        taq.stats.get_remaing_time_budget.return_value = 10

        # remaining_time - 5 < cutoff
        res, _ = taq.run_wrapper(run_info)
        assert res.cutoff == 5

        # flot cutoff ==> round up
        run_info = run_info._replace(cutoff=2.5)
        res, _ = taq.run_wrapper(run_info)
        assert res.cutoff == 3

    def test_add_learning_curve_info(self):
        # add_learning_curve_info is experimental
        taq = _create_taq()
        additional_run_info = {}
        iter = np.arange(1, 6)
        info = [
            RunValue(
                cost=1e9,
                time=1e9,
                status=1e9,
                starttime=1e9,
                endtime=1e9,
                additional_info={
                    'duration': 0.1 * i,
                    'train_loss': 0.2 * i,
                    'loss': 0.3 * i
                }
            )
            for i in iter
        ]
        taq._add_learning_curve_info(
            additional_run_info=additional_run_info,
            info=info
        )

        for i, key in enumerate([
            'learning_curve_runtime',
            'train_learning_curve',
            'learning_curve'
        ]):
            assert key in additional_run_info
            assert np.allclose(additional_run_info[key], 0.1 * iter * (i + 1))
