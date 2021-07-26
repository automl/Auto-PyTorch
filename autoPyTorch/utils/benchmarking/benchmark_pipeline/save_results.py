import os
import json
import time
import logging

from hpbandster.core.result import logged_results_to_HBS_result
from autoPyTorch.pipeline.base.pipeline_node import PipelineNode

class SaveResults(PipelineNode):

    def fit(self, result_dir, fit_duration, final_score, test_score, autonet, task_id):
        if (task_id not in [-1, 1]):
            time.sleep(60)
            return dict()

        logging.getLogger('benchmark').info("Create and save summary")

        res = logged_results_to_HBS_result(result_dir)
        id2config = res.get_id2config_mapping()
        incumbent_trajectory = res.get_incumbent_trajectory(bigger_is_better=False, non_decreasing_budget=False)
        final_config_id = incumbent_trajectory['config_ids'][-1]
        incumbent_config = id2config[final_config_id]['config']
        
        final_info = res.get_runs_by_id(final_config_id)[-1]["info"]

        summary = dict()
        summary["final_loss"] = final_score
        summary["test_score"] = test_score
        summary["incumbent_config"] = incumbent_config
        summary["duration"] = fit_duration
        summary["info"] = final_info
        
        # write as json
        with open(os.path.join(result_dir, "summary.json"), "w") as f:
            json.dump(summary, f)

        return dict()
