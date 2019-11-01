__author__ = "Max Dippel, Michael Burkart and Matthias Urban"
__version__ = "0.0.1"
__license__ = "BSD"

import unittest
from autoPyTorch.pipeline.base.pipeline import Pipeline
from autoPyTorch.pipeline.nodes.metric_selector import MetricSelector

from autoPyTorch.components.metrics.standard_metrics import accuracy, auc_metric, mean_distance

class TestMetricSelectorMethods(unittest.TestCase):

    def test_selector(self):
        pipeline = Pipeline([
            MetricSelector()
        ])

        selector = pipeline[MetricSelector.get_name()]
        selector.add_metric("auc", auc_metric)
        selector.add_metric("accuracy", accuracy)
        selector.add_metric("mean", mean_distance)

        pipeline_config = pipeline.get_pipeline_config(optimize_metric="accuracy", additional_metrics=['auc', 'mean'])
        pipeline.fit_pipeline(pipeline_config=pipeline_config)

        selected_optimize_metric = selector.fit_output['optimize_metric']
        selected_additional_metrics = selector.fit_output['additional_metrics']

        self.assertEqual(selected_optimize_metric.metric, accuracy)
        self.assertSetEqual(set(x.metric for x in selected_additional_metrics), set([auc_metric, mean_distance]))



