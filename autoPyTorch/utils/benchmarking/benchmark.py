from autoPyTorch.pipeline.base.pipeline import Pipeline
from autoPyTorch.utils.benchmarking.benchmark_pipeline import (BenchmarkSettings,
                                                           CreateAutoNet,
                                                           FitAutoNet,
                                                           ForAutoNetConfig,
                                                           ForInstance, ForRun,
                                                           PrepareResultFolder,
                                                           ReadInstanceData,
                                                           SaveResults,
                                                           SetAutoNetConfig)
from autoPyTorch.utils.benchmarking.visualization_pipeline import (CollectAutoNetConfigTrajectories,
                                                               CollectRunTrajectories,
                                                               GetRunTrajectories,
                                                               PlotTrajectories,
                                                               ReadInstanceInfo,
                                                               VisualizationSettings)
from autoPyTorch.utils.benchmarking.visualization_pipeline import ForInstance as VisualizationForInstance
from autoPyTorch.utils.config.config_file_parser import ConfigFileParser

class Benchmark():
    def __init__(self):
        self.benchmark_pipeline = self.get_benchmark_pipeline()
        self.visualization_pipeline = self.get_visualization_pipeline()

    def get_benchmark_config_file_parser(self):
        return ConfigFileParser(self.benchmark_pipeline.get_pipeline_config_options())

    def run_benchmark(self, **benchmark_config):
        config = self.benchmark_pipeline.get_pipeline_config(**benchmark_config)
        self.benchmark_pipeline.fit_pipeline(pipeline_config=config)
    
    def visualize_benchmark(self, **benchmark_config):
        config = self.visualization_pipeline.get_pipeline_config(throw_error_if_invalid=False, **benchmark_config)
        self.visualization_pipeline.fit_pipeline(pipeline_config=config)

    def get_benchmark_pipeline(self):
        return Pipeline([
            BenchmarkSettings(),
            ForInstance([ #instance_file
                ReadInstanceData(), #test_split, is_classification, instance
                CreateAutoNet(),
                ForAutoNetConfig([ #autonet_config_file
                    SetAutoNetConfig(), #use_dataset_metric, use_dataset_max_runtime
                    ForRun([ #num_runs, run_ids
                        PrepareResultFolder(),
                        FitAutoNet(),
                        SaveResults()
                    ])
                ])
            ])
        ])

    def get_visualization_pipeline(self):
        return Pipeline([
            VisualizationSettings(),
            VisualizationForInstance([
                CollectAutoNetConfigTrajectories([
                    CollectRunTrajectories([
                        ReadInstanceInfo(),
                        CreateAutoNet(),
                        GetRunTrajectories()
                    ])
                ]),
                PlotTrajectories()
            ])
        ])
