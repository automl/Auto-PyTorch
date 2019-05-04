from autoPyTorch.pipeline.base.pipeline_node import PipelineNode
from autoPyTorch.utils.benchmarking.visualization_pipeline.plot_trajectories import plot, label_rename, process_trajectory
from autoPyTorch.utils.config.config_option import ConfigOption, to_bool
import os
import logging
import numpy as np
import random
import heapq

class PlotSummary(PipelineNode):
    def fit(self, pipeline_config, trajectories, optimize_metrics):
        if not pipeline_config["skip_ranking_plot"]:
            plot(dict(pipeline_config, plot_type="losses", y_scale="linear"), trajectories, optimize_metrics, "ranking", process_summary)
        if not pipeline_config["skip_average_plot"]:
            plot(dict(pipeline_config, scale_uncertainty=0), trajectories, optimize_metrics, "average", process_summary)
            plot(pipeline_config, trajectories, optimize_metrics, "sampled_average", trajectory_sampling)
        return dict()

    def get_pipeline_config_options(self):
        options = [
            ConfigOption('skip_ranking_plot', default=False, type=to_bool),
            ConfigOption('skip_average_plot', default=False, type=to_bool)
        ]
        return options


def get_ranking_plot_values(values, names, agglomeration):
    """ values = instance_name --> [((key=prefix + metric), value), ...] """
    keys = {instance: set([key for key, _ in v]) for instance, v in values.items()}
    values = {instance: [(key, agglomeration([value for k, value in v if k == key])) for key in keys[instance]] for instance, v in values.items()}
    sorted_values = {instance: sorted(map(lambda x: x[1], v)) for instance, v in values.items()}  # configs sorted by value
    ranks = {instance: {n: [sorted_values[instance].index(value) + 1 for config_name, value in v if config_name == n] for n in names}
             for instance, v in values.items()}
    ranks = to_dict([(n, r) for rank_dict in ranks.values() for n, r in rank_dict.items()])
    for name in names:
        ranks[name] = [i for j in ranks[name] for i in j]  # flatten
    return ranks


def get_average_plot_values(values, names, agglomeration):
    """ values = instance_name --> [((key=prefix + metric), value), ...] """
    result = dict()
    for name in names: # prepare lists
        result[name] = list()
    for _, v in values.items():  # aggregate over all instances
        for name, value in v:  # aggregate over all runs
            result[name].append(value)
    return result

get_plot_values_funcs = {
    "ranking": get_ranking_plot_values,
    "average": get_average_plot_values
}

def process_summary(instance_name, metric_name, prefixes, trajectories, plot_type, agglomeration, scale_uncertainty, value_multiplier, cmap):
    assert instance_name in get_plot_values_funcs.keys()
    trajectory_names_to_prefix = {(("%s_%s" % (prefix, metric_name)) if prefix else metric_name): prefix
        for prefix in prefixes}
    trajectory_names = [t for t in trajectory_names_to_prefix.keys() if t in trajectories]

    # save pointers for each trajectory to iterate over them simultaneously
    trajectory_pointers = {(config, name): {instance: ([0] * len(run_trajectories))  # name is trajectory name, which consists of prefix and metric
        for instance, run_trajectories in instance_trajectories.items()}
        for name in trajectory_names
        for config, instance_trajectories in trajectories[name].items()}
    trajectory_values = {(config, name): {instance: ([None] * len(run_trajectories))
        for instance, run_trajectories in instance_trajectories.items()}
        for name in trajectory_names
        for config, instance_trajectories in trajectories[name].items()}
    heap = [(run_trajectories[j]["times_finished"][0], config, name, instance, j)
            for name in trajectory_names
            for config, instance_trajectories in trajectories[name].items()
            for instance, run_trajectories in instance_trajectories.items()
            for j in range(len(run_trajectories))]
    heapq.heapify(heap)

    # data to plot
    center = {(config, name): [] for name in trajectory_names for config in trajectories[name].keys()}
    upper = {(config, name): [] for name in trajectory_names for config in trajectories[name].keys()}
    lower = {(config, name): [] for name in trajectory_names for config in trajectories[name].keys()}
    finishing_times = []
    plot_empty = True

    # iterate simultaneously over all trajectories with increasing finishing times
    while heap:

        # get trajectory with lowest finishing time
        times_finished, current_config, current_name, current_instance, trajectory_id = heapq.heappop(heap)

        # update trajectory values and pointers
        current_trajectory = trajectories[current_name][current_config][current_instance][trajectory_id]
        current_pointer = trajectory_pointers[(current_config, current_name)][current_instance][trajectory_id]
        current_value = current_trajectory[plot_type][current_pointer]

        trajectory_values[(current_config, current_name)][current_instance][trajectory_id] = current_value
        trajectory_pointers[(current_config, current_name)][current_instance][trajectory_id] += 1

        if trajectory_pointers[(current_config, current_name)][current_instance][trajectory_id] < len(current_trajectory[plot_type]):
            heapq.heappush(heap,
                (current_trajectory["times_finished"][trajectory_pointers[(current_config, current_name)][current_instance][trajectory_id]],
                 current_config, current_name, current_instance, trajectory_id))

        if any(value is None for _, instance_values in trajectory_values.items() for _, values in instance_values.items() for value in values):
            continue

        if finishing_times and np.isclose(times_finished, finishing_times[-1]):
            finishing_times.pop()
            [x[k].pop() for x in [center, upper, lower] for k in x.keys()]

        # calculate ranks
        values = to_dict([(instance, (config, name), value * value_multiplier)
            for (config, name), instance_values in trajectory_values.items()
            for instance, values in instance_values.items()
            for value in values if value is not None])
        plot_values = get_plot_values_funcs[instance_name](values, center.keys(), np.median if agglomeration == "median" else np.mean)
        
        # populate plotting data
        for key in center.keys():
            if not plot_values[key]:
                center[key].append(float("nan"))
                lower[key].append(float("nan"))
                upper[key].append(float("nan"))

            center[key].append(np.mean(plot_values[key]))
            lower[key].append(-1 * scale_uncertainty * np.std(plot_values[key]) + center[key][-1])
            upper[key].append(scale_uncertainty * np.std(plot_values[key]) + center[key][-1])
        finishing_times.append(times_finished)
        plot_empty = False
        
    # do the plotting
    plot_data = dict()
    for i, (config, name) in enumerate(center.keys()):
        prefix = trajectory_names_to_prefix[name]
        label = ("%s: %s" % (prefix, config)) if prefix else config
        color = cmap(i / len(center))
        plot_data[label] = {
            "individual_trajectory": None,
            "individual_times_finished": None,
            "color": color,
            "linestyle": "-",
            "center": center[(config, name)],
            "lower": lower[(config, name)],
            "upper": upper[(config, name)],
            "finishing_times": finishing_times
        }
    return plot_empty, plot_data

def to_dict(tuple_list):
    result = dict()
    for v in tuple_list:
        a = v[0]
        b = v[1:]
        if len(b) == 1:
            b = b[0]
        if a not in result:
            result[a] = list()
        result[a].append(b)
    return result


def trajectory_sampling(instance_name, metric_name, prefixes, trajectories, plot_type, agglomeration, scale_uncertainty, value_multiplier, cmap, num_samples=1000):
    averaged_trajectories = dict()

    # sample #num_samples average trajectories
    for i in range(num_samples):
        sampled_trajectories = dict()

        for p, prefix in enumerate(prefixes):
            trajectory_name = ("%s_%s" % (prefix, metric_name)) if prefix else metric_name
            config_trajectories = trajectories[trajectory_name]
            if trajectory_name not in sampled_trajectories:
                        sampled_trajectories[trajectory_name] = dict()

            for config, instance_trajectories in config_trajectories.items():
                if config not in sampled_trajectories[trajectory_name]:
                        sampled_trajectories[trajectory_name][config] = list()

                # for each instance choose a random trajectory over the runs
                for instance, run_trajectories in instance_trajectories.items():
                    run_trajectory = random.choice(run_trajectories)
                    sampled_trajectories[trajectory_name][config].append(run_trajectory)

        # compute the average over the instances
        plot_empty, plot_data = process_trajectory(
            instance_name="prepare_sampled_%s_(%s/%s)" % (instance_name, i, num_samples),
            metric_name=metric_name,
            prefixes=prefixes,
            trajectories=sampled_trajectories,
            plot_type=plot_type,
            agglomeration=agglomeration,
            scale_uncertainty=0,
            value_multiplier=value_multiplier,
            cmap=cmap
        )

        if plot_empty:
            continue

        # save the average trajectories
        for label, d in plot_data.items():
            prefix, config = label.split(": ") if ": " in label else ("", label)
            trajectory_name = ("%s_%s" % (prefix, metric_name)) if prefix else metric_name

            if trajectory_name not in averaged_trajectories:
                averaged_trajectories[trajectory_name] = dict()
            if config not in averaged_trajectories[trajectory_name]:
                averaged_trajectories[trajectory_name][config] = list()

            averaged_trajectories[trajectory_name][config].append({
                "times_finished": d["finishing_times"],
                plot_type: d["center"],
            })
    
    # compute mean and stddev over the averaged trajectories
    return process_trajectory(
        instance_name=instance_name,
        metric_name=metric_name,
        prefixes=prefixes,
        trajectories=averaged_trajectories,
        plot_type=plot_type,
        agglomeration="mean",
        scale_uncertainty=scale_uncertainty,
        value_multiplier=1,
        cmap=cmap
    )
