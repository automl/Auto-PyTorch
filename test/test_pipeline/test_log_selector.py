__author__ = "Max Dippel, Michael Burkart and Matthias Urban"
__version__ = "0.0.1"
__license__ = "BSD"

import unittest
from autoPyTorch.pipeline.base.pipeline import Pipeline
from autoPyTorch.pipeline.nodes.log_functions_selector import LogFunctionsSelector

class TestLogFunctionSelectorMethods(unittest.TestCase):

    def test_selector(self):

        def log_fnc1(network, epoch):
            print("a")

        def log_fnc2(network, epoch):
            print("b")

        selector = LogFunctionsSelector()
        pipeline = Pipeline([
            selector
        ])
        selector.add_log_function("log1", log_fnc1)
        selector.add_log_function("log2", log_fnc2)

        pipeline_config = pipeline.get_pipeline_config(additional_logs=["log2"])
        pipeline.fit_pipeline(pipeline_config=pipeline_config)

        log_functions = selector.fit_output['log_functions']

        self.assertListEqual(log_functions, [log_fnc2])



