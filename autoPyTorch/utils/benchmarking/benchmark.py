from autoPyTorch.pipeline.base.pipeline import Pipeline
from autoPyTorch.utils.benchmarking.benchmark_pipeline import (BenchmarkSettings,
                                                               CreateAutoNet,
                                                               FitAutoNet,
                                                               ForAutoNetConfig,
                                                               ForInstance, ForRun,
                                                               PrepareResultFolder,
                                                               ReadInstanceData,
                                                               SaveResults,
                                                               SetAutoNetConfig,
                                                               ApplyUserUpdates,
                                                               SaveEnsembleLogs,
                                                               SetEnsembleConfig)
from autoPyTorch.utils.benchmarking.visualization_pipeline import (CollectAutoNetConfigTrajectories,
                                                               CollectRunTrajectories,
                                                               CollectInstanceTrajectories,
                                                               GetRunTrajectories,
                                                               PlotTrajectories,
                                                               ReadInstanceInfo,
                                                               VisualizationSettings,
                                                               GetEnsembleTrajectories,
                                                               PlotSummary,
                                                               GetAdditionalTrajectories)
from autoPyTorch.utils.config.config_file_parser import ConfigFileParser

class Benchmark():
    def __init__(self):
        self.benchmark_pipeline = self.get_benchmark_pipeline()
        self.visualization_pipeline = self.get_visualization_pipeline()
        self.compute_ensemble_performance_pipeline = self.get_ensemble_performance_pipeline()

    def get_benchmark_config_file_parser(self):
        return ConfigFileParser(self.benchmark_pipeline.get_pipeline_config_options())

    def run_benchmark(self, **benchmark_config):
        config = self.benchmark_pipeline.get_pipeline_config(**benchmark_config)
        self.benchmark_pipeline.fit_pipeline(pipeline_config=config)
    
    def visualize_benchmark(self, **benchmark_config):
        config = self.visualization_pipeline.get_pipeline_config(throw_error_if_invalid=False, **benchmark_config)
        self.visualization_pipeline.fit_pipeline(pipeline_config=config)
    
    def compute_ensemble_performance(self, **benchmark_config):
        config = self.compute_ensemble_performance_pipeline.get_pipeline_config(throw_error_if_invalid=False, **benchmark_config)
        self.compute_ensemble_performance_pipeline.fit_pipeline(pipeline_config=config)

    def get_benchmark_pipeline(self):
        return Pipeline([
            BenchmarkSettings(),
            ForInstance([                    # loop through instance files
                ReadInstanceData(),          # get test_split, is_classification, instance
                CreateAutoNet(),
                #ApplyUserUpdates(),
                ForAutoNetConfig([           # loop through autonet_config_file
                    SetAutoNetConfig(),      # use_dataset_metric, use_dataset_max_runtime
                    ForRun([                 # loop through num_runs, run_ids
                        PrepareResultFolder(),
                        FitAutoNet(),
                        SaveResults(),
                        SaveEnsembleLogs()
                    ])
                ])
            ])
        ])

    def get_visualization_pipeline(self):
        return Pipeline([
            VisualizationSettings(),
            CollectInstanceTrajectories([
                CollectAutoNetConfigTrajectories([
                    CollectRunTrajectories([
                        ReadInstanceInfo(),
                        CreateAutoNet(),
                        GetRunTrajectories(),
                        GetEnsembleTrajectories()
                    ])
                ]),
                GetAdditionalTrajectories(),
                PlotTrajectories()
            ]),
            PlotSummary()
        ])
    
    def get_ensemble_performance_pipeline(self):
        return Pipeline([
            VisualizationSettings(),
            CollectInstanceTrajectories([
                CollectAutoNetConfigTrajectories([
                    CollectRunTrajectories([
                        ReadInstanceInfo(),
                        CreateAutoNet(),
                        SetEnsembleConfig(),
                        SaveEnsembleLogs(),
                        GetEnsembleTrajectories()
                    ])
                ])
            ])
        ])
