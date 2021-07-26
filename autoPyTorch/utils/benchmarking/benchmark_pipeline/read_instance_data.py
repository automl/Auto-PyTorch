import numpy as np
from autoPyTorch.utils.config.config_option import ConfigOption, to_bool
from autoPyTorch.pipeline.base.pipeline_node import PipelineNode
from autoPyTorch.data_management.data_manager import DataManager, ImageManager

class ReadInstanceData(PipelineNode):

    def fit(self, pipeline_config, instance):
        if pipeline_config['problem_type'] in ['feature_classification', 'feature_multilabel', 'feature_regression']:
            dm = DataManager(verbose=pipeline_config["data_manager_verbose"])
        else:
            dm = ImageManager(verbose=pipeline_config["data_manager_verbose"])

        if pipeline_config['test_instances'] is not None:   # so far only for images
            
            test_instances = [s for s in pipeline_config['test_instances'] if not (s=="[" or s=="]")]
            test_instances = "".join(test_instances)
            test_instances = test_instances.split(",")

            dm.read_data(instance,                                                                            # dm for train/val data
                         is_classification=(pipeline_config["problem_type"] in ['feature_classification', 'feature_multilabel', 'image_classification']),
                         test_split=0.0)
            
            if len(test_instances)==1:
                dm_test = ImageManager(verbose=pipeline_config["data_manager_verbose"])                           # cheat to get test data
                dm_test.read_data(test_instances[0],
                                  is_classification=(pipeline_config["problem_type"] in ['feature_classification', 'feature_multilabel', 'image_classification']),
                                  test_split=0.0)
                dm.X_test, dm.Y_test = dm_test.X_train, dm_test.Y_train.astype(np.int32)
                print("Using train data:", dm.X_train, dm.Y_train)
                print("Found test data:", dm.X_test, dm.Y_test)
            else:
                accum = []
                for test_inst in test_instances:
                    if test_inst.strip()=="None":
                        accum.append(None)
                    else:
                        accum.append(test_inst.strip())
                dm.X_test = accum
                dm.Y_test = [None for i in range(len(test_instances))]
                print("Using train data:", dm.X_train, dm.Y_train)
                print("Found test data:", dm.X_test, dm.Y_test)


        else: # use test split
            dm.read_data(instance,
                is_classification=(pipeline_config["problem_type"] in ['feature_classification', 'feature_multilabel', 'image_classification']),
                test_split=pipeline_config["test_split"])
        return {"data_manager": dm}

    def get_pipeline_config_options(self):
        options = [
            ConfigOption("test_split", default=0.0, type=float),
            ConfigOption("problem_type", default='feature_classification', type=str, choices=['feature_classification', 'feature_multilabel', 'feature_regression', 'image_classification']),
            ConfigOption("data_manager_verbose", default=False, type=to_bool),
            ConfigOption("test_instances", default=None, type=list)
        ]
        return options
