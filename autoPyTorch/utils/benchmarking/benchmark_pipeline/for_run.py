import logging
from autoPyTorch.utils.config.config_option import ConfigOption
from autoPyTorch.pipeline.base.sub_pipeline_node import SubPipelineNode
import traceback

class ForRun(SubPipelineNode):
    def fit(self, pipeline_config, autonet, data_manager, instance, run_id, task_id):
        for run_number in self.parse_range(pipeline_config['run_number_range'], pipeline_config['num_runs']):
            try:
                logging.getLogger('benchmark').info("Start run " + str(run_id) + "_" + str(run_number))
                self.sub_pipeline.fit_pipeline(pipeline_config=pipeline_config,
                    data_manager=data_manager, instance=instance, autonet=autonet, 
                    run_number=run_number, run_id=run_id, task_id=task_id)
            except Exception as e:
                print("Exception for (run_id, task_id): ", run_id, task_id)
                print(e)
                traceback.print_exc()
        return dict()

    def get_pipeline_config_options(self):
        options = [
            ConfigOption("num_runs", default=1, type=int),
            ConfigOption("run_number_range", default=None, type=str)
        ]
        return options

    def parse_range(self, range_string, fallback):
        if (range_string is None):
            return range(fallback)

        split = range_string.split(":")
        if len(split) == 1:
            start = int(split[0]) if split[0] != "" else 0
            stop = (int(split[0]) + 1) if split[0] != "" else fallback
            step = 1
        elif len(split) == 2:
            start = int(split[0]) if split[0] != "" else 0
            stop = int(split[1]) if split[1] != "" else fallback
            step = 1
        elif len(split) == 3:
            start = int(split[0]) if split[0] != "" else 0
            stop = int(split[1]) if split[1] != "" else fallback
            step = int(split[2]) if split[2] != "" else 1
        return range(start, stop, step)
