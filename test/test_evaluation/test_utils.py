import pytest

from autoPyTorch.evaluation.utils import DisableFileOutputParameters

def test_disable_file_output_string_no_error():
    disable_file_output = ['pipeline', 'pipelines']
    DisableFileOutputParameters.check_compatibility(disable_file_output=disable_file_output)

def test_disable_file_output_string_error():
    disable_file_output = ['model']
    with pytest.raises(ValueError, match=r"Expected .*? to be in the members (.*?) of"
                                         r" DisableFileOutputParameters or an instance."):
        DisableFileOutputParameters.check_compatibility(disable_file_output=disable_file_output)

def test_disable_file_output_enum_no_error():
    disable_file_output = [DisableFileOutputParameters.pipeline, DisableFileOutputParameters.pipelines]
    DisableFileOutputParameters.check_compatibility(disable_file_output=disable_file_output)