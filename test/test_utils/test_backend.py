# -*- encoding: utf-8 -*-
import builtins
import unittest
import unittest.mock

import pytest

from autoPyTorch.utils.backend import Backend


class BackendStub(Backend):

    def __init__(self):
        self.__class__ = Backend


##########################################################################################
#                                       Fixtures
##########################################################################################
@pytest.fixture
def backend_stub():
    backend = BackendStub()
    backend.internals_directory = '/'
    return backend


def _setup_load_model_mocks(openMock, pickleLoadMock, seed, idx, budget):
    model_path = '/runs/%s_%s_%s/%s.%s.%s.model' % (seed, idx, budget, seed, idx, budget)
    file_handler = 'file_handler'
    expected_model = 'model'

    fileMock = unittest.mock.MagicMock()
    fileMock.__enter__.return_value = file_handler

    openMock.side_effect = \
        lambda path, flag: fileMock if path == model_path and flag == 'rb' else None
    pickleLoadMock.side_effect = lambda fh: expected_model if fh == file_handler else None

    return expected_model


##########################################################################################
#                                         Tests
##########################################################################################
@unittest.mock.patch('pickle.load')
@unittest.mock.patch('os.path.exists')
def test_load_model_by_seed_and_id(exists_mock, pickleLoadMock, backend_stub):
    exists_mock.return_value = False
    open_mock = unittest.mock.mock_open(read_data='Data')
    with unittest.mock.patch(
        'autoPyTorch.utils.backend.open',
        open_mock,
        create=True,
    ):
        seed = 13
        idx = 17
        budget = 50.0
        expected_model = _setup_load_model_mocks(open_mock,
                                                 pickleLoadMock,
                                                 seed, idx, budget)

        actual_model = backend_stub.load_model_by_seed_and_id_and_budget(
            seed, idx, budget)

        assert expected_model == actual_model


@unittest.mock.patch('pickle.load')
@unittest.mock.patch.object(builtins, 'open')
@unittest.mock.patch('os.path.exists')
def test_loads_models_by_identifiers(exists_mock, openMock, pickleLoadMock, backend_stub):
    exists_mock.return_value = True
    seed = 13
    idx = 17
    budget = 50.0
    expected_model = _setup_load_model_mocks(
        openMock, pickleLoadMock, seed, idx, budget)
    expected_dict = {(seed, idx, budget): expected_model}

    actual_dict = backend_stub.load_models_by_identifiers([(seed, idx, budget)])

    assert isinstance(actual_dict, dict)
    assert expected_dict == actual_dict
