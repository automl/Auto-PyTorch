from autoPyTorch.pipeline.base.pipeline_node import PipelineNode
from autoPyTorch.utils.config.config_option import ConfigOption, to_bool
from autoPyTorch.utils.config.config_file_parser import ConfigFileParser
from autoPyTorch.data_management.data_manager import DataManager
from autoPyTorch.utils.benchmarking.benchmark_pipeline.prepare_result_folder import get_run_result_dir
from autoPyTorch.utils.benchmarking.benchmark_pipeline.read_instance_data import ReadInstanceData
from autoPyTorch.data_management.data_manager import ProblemType
import os
import ast

class ReadInstanceInfo(ReadInstanceData):

    def fit(self, pipeline_config, run_result_dir):

        instance_file_config_parser = ConfigFileParser([
            ConfigOption(name='path', type='directory', required=True),
            ConfigOption(name='is_classification', type=to_bool, required=True),
            ConfigOption(name='is_multilabel', type=to_bool, required=True),
            ConfigOption(name='num_features', type=int, required=True),
            ConfigOption(name='categorical_features', type=bool, required=True, list=True),
            ConfigOption(name='instance_shape', type=[ast.literal_eval, lambda x: isinstance(x, tuple)], required=True)
        ])
        instance_info = instance_file_config_parser.read(os.path.join(run_result_dir, 'instance.info'))
        instance_info = instance_file_config_parser.set_defaults(instance_info)

        dm = DataManager()
        if instance_info["is_multilabel"]:
            dm.problem_type = ProblemType.FeatureMultilabel
        elif instance_info["is_classification"]:
            dm.problem_type = ProblemType.FeatureClassification
        else:
             dm.problem_type = ProblemType.FeatureClassification

        return {'instance_info': instance_info, 'data_manager': dm}