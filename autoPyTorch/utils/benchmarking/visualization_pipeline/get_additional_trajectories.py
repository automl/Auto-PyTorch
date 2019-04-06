from autoPyTorch.pipeline.base.pipeline_node import PipelineNode
from autoPyTorch.utils.config.config_option import ConfigOption
import json, os, csv, traceback


class GetAdditionalTrajectories(PipelineNode):
    def fit(self, pipeline_config, trajectories, train_metrics, instance):
        for additional_trajectory_path in pipeline_config["additional_trajectories"]:

            # open trajectory description file
            with open(additional_trajectory_path, "r") as f:
                trajectories_description = json.load(f)
                config_name = trajectories_description["name"]
                file_format = trajectories_description["format"]
                assert file_format in trajectory_loaders.keys(), "Unsupported file type %s" % file_format
                assert not any(config_name in t.keys() for t in trajectories.values()), "Invalid additional trajectory name %s" % config_name

                if instance not in trajectories_description["instances"]:
                    continue
                
                columns_description = trajectories_description["columns"]

                # process all trajectories for current instance
                for path in trajectories_description["instances"][instance]:
                    path = os.path.join(os.path.dirname(additional_trajectory_path), path)
                    try:
                        trajectory_loaders[file_format](path, config_name, columns_description, trajectories)
                    except FileNotFoundError as e:
                        print("Trajectory could not be loaded: %s. Skipping." % e)
                        traceback.print_exc()                        
        return {"trajectories": trajectories,
                "train_metrics": train_metrics}
    
    def get_pipeline_config_options(self):
        options = [
            ConfigOption("additional_trajectories", default=list(), type="directory", list=True)
        ]
        return options

def csv_trajectory_loader(path, config_name, columns_description, trajectories):
    with open(path, "r", newline="") as f:
        reader = csv.reader(f, delimiter=",")
        next(reader)

        # parse the csv
        times_finished = list()
        performances = dict()    
        for row in reader:
            for i, col in enumerate(row):
                if i == columns_description["time_column"]:
                    times_finished.append(max(0, float(col)))
                if str(i) in columns_description["metric_columns"].keys():
                    log_name = columns_description["metric_columns"][str(i)]["name"]
                    transform = columns_description["metric_columns"][str(i)]["transform"] \
                        if "transform" in columns_description["metric_columns"][str(i)] else "x"

                    if log_name not in performances:
                        performances[log_name] = list()
                    
                    performances[log_name].append(eval_expr(transform.replace("x", col)))
        
        # add data to the other trajectories
        for log_name, performance_list in performances.items():
            if log_name not in trajectories:
                trajectories[log_name] = dict()
            if config_name not in trajectories[log_name]:
                trajectories[log_name][config_name] = list()
            trajectories[log_name][config_name].append({
                "times_finished": sorted(times_finished),
                "losses": list(zip(*sorted(zip(times_finished, performance_list))))[1],
                "flipped": False
            })

trajectory_loaders = {"csv": csv_trajectory_loader}

import ast
import operator as op

# supported operators
operators = {ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul,
             ast.Div: op.truediv, ast.Pow: op.pow, ast.BitXor: op.xor,
             ast.USub: op.neg}

def eval_expr(expr):
    return eval_(ast.parse(expr, mode='eval').body)

def eval_(node):
    if isinstance(node, ast.Num): # <number>
        return node.n
    elif isinstance(node, ast.BinOp): # <left> <operator> <right>
        return operators[type(node.op)](eval_(node.left), eval_(node.right))
    elif isinstance(node, ast.UnaryOp): # <operator> <operand> e.g., -1
        return operators[type(node.op)](eval_(node.operand))
    else:
        raise TypeError(node)
