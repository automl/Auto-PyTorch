import os
import logging
from autoPyTorch.utils.config.config_option import ConfigOption, to_bool
from autoPyTorch.utils.config.config_file_parser import ConfigFileParser
from autoPyTorch.pipeline.base.sub_pipeline_node import SubPipelineNode
from autoPyTorch.utils.benchmarking.benchmark_pipeline import ForInstance as BaseForInstance
import traceback

class ForInstance(BaseForInstance):
    def fit(self, pipeline_config, run_id_range):
        instances = self.get_instances(pipeline_config, instance_slice=self.parse_slice(pipeline_config["instance_slice"]))
        for instance in instances:
            try:
                self.sub_pipeline.fit_pipeline(pipeline_config=pipeline_config, instance=instance, run_id_range=run_id_range)
            except Exception as e:
                print(e)
                traceback.print_exc()
        return dict()