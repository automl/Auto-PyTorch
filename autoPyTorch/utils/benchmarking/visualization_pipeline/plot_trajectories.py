import os
from autoPyTorch.utils.config.config_option import ConfigOption, to_bool
from autoPyTorch.pipeline.base.pipeline_node import PipelineNode
import numpy as np
import logging
import json
import heapq

class PlotTrajectories(PipelineNode):

    def fit(self, pipeline_config, trajectories, optimize_metrics, instance):
        if not pipeline_config["skip_dataset_plots"]:
            plot(pipeline_config, trajectories, optimize_metrics, instance, process_trajectory)
        return {"trajectories": trajectories, "optimize_metrics": optimize_metrics}
    

    def get_pipeline_config_options(self):
        options = [
            ConfigOption('plot_logs', default=None, type='str', list=True),
            ConfigOption('output_folder', default=None, type='directory'),
            ConfigOption('agglomeration', default='mean', choices=['mean', 'median']),
            ConfigOption('scale_uncertainty', default=1, type=float),
            ConfigOption('font_size', default=12, type=int),
            ConfigOption('prefixes', default=["val"], list=True, choices=["", "train", "val", "test", "ensemble", "ensemble_test"]),
            ConfigOption('label_rename', default=False, type=to_bool),
            ConfigOption('skip_dataset_plots', default=False, type=to_bool),
            ConfigOption('plot_markers', default=False, type=to_bool),
            ConfigOption('plot_individual', default=False, type=to_bool),
            ConfigOption('plot_type', default="values", type=str, choices=["values", "losses"]),
            ConfigOption('xscale', default='log', type=str),
            ConfigOption('yscale', default='linear', type=str),
            ConfigOption('xmin', default=None, type=float),
            ConfigOption('xmax', default=None, type=float),
            ConfigOption('ymin', default=None, type=float),
            ConfigOption('ymax', default=None, type=float),
            ConfigOption('value_multiplier', default=1, type=float)
        ]
        return options


def plot(pipeline_config, trajectories, optimize_metrics, instance, process_fnc):
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    extension = "pdf"

    plot_logs = pipeline_config['plot_logs'] or optimize_metrics
    output_folder = pipeline_config['output_folder']
    instance_name = os.path.basename(instance).split(".")[0]

    if output_folder and not os.path.exists(output_folder):
        os.mkdir(output_folder)

    # iterate over all incumbent trajectories for each metric
    for i, metric_name in enumerate(plot_logs):
        
        # prepare pdf
        if output_folder is not None:
            pdf_destination = os.path.join(output_folder, instance_name + '_' + metric_name + '.' + extension)
            pp = PdfPages(pdf_destination)

        # create figure
        figure = plt.figure(i)
        plot_empty, plot_data = process_fnc(instance_name=instance_name,
                                            metric_name=metric_name,
                                            prefixes=pipeline_config["prefixes"],
                                            trajectories=trajectories,
                                            plot_type=pipeline_config["plot_type"],
                                            agglomeration=pipeline_config["agglomeration"],
                                            scale_uncertainty=pipeline_config['scale_uncertainty'],
                                            value_multiplier=pipeline_config['value_multiplier'],
                                            cmap=plt.get_cmap('jet'))
        if plot_empty:
            logging.getLogger('benchmark').warn('Not showing empty plot for ' + instance)
            plt.close(figure)
            continue

        plot_trajectory(plot_data=plot_data,
                        instance_name=instance_name,
                        metric_name=metric_name,
                        font_size=pipeline_config["font_size"],
                        do_label_rename=pipeline_config['label_rename'],
                        plt=plt,
                        plot_individual=pipeline_config["plot_individual"],
                        plot_markers=pipeline_config["plot_markers"],
                        plot_type=pipeline_config["plot_type"])
        
        plt.xscale(pipeline_config["xscale"])
        plt.yscale(pipeline_config["yscale"])
        plt.xlim((pipeline_config["xmin"], pipeline_config["xmax"]))
        plt.ylim((pipeline_config["ymin"], pipeline_config["ymax"]))

        # show or save
        if output_folder is None:
            logging.getLogger('benchmark').info('Showing plot for ' + instance)
            plt.show()
        else:
            logging.getLogger('benchmark').info('Saving plot for ' + instance + ' at ' + pdf_destination)
            pp.savefig(figure)
            pp.close()
            plt.close(figure)


def process_trajectory(instance_name, metric_name, prefixes, trajectories, plot_type, agglomeration, scale_uncertainty, value_multiplier, cmap):
    # iterate over the incumbent trajectories of the different runs
    linestyles = ['-', '--', '-.', ':']
    plot_empty = True
    plot_data = dict()
    for p, prefix in enumerate(prefixes):
        trajectory_name = ("%s_%s" % (prefix, metric_name)) if prefix else metric_name
        linestyle = linestyles[p % len(linestyles)]
        if trajectory_name not in trajectories:
            continue

        config_trajectories = trajectories[trajectory_name]
        for i, (config_name, trajectory) in enumerate(config_trajectories.items()):
            color = cmap((i *len(prefixes) + p) / (len(config_trajectories) * len(prefixes)))

            trajectory_pointers = [0] * len(trajectory)  # points to current entry of each trajectory
            trajectory_values = [None] * len(trajectory)  # list of current values of each trajectory
            individual_trajectories = [[] for _ in range(len(trajectory))]
            individual_times_finished = [[] for _ in range(len(trajectory))]
            heap = [(trajectory[j]["times_finished"][0], j) for j in range(len(trajectory))]
            heapq.heapify(heap)
            # progress = 0
            # total = sum(len(trajectory[j]["times_finished"]) for j in range(len(trajectory)))

            # data to plot
            center = []
            lower = []
            upper = []
            finishing_times = []
            # print("Calculate plot data for instance %s and trajectory %s and config %s" % (instance_name, trajectory_name, config_name))

            # iterate simultaneously over all trajectories with increasing finishing times
            while heap:

                # get trajectory with lowest finishing times
                times_finished, trajectory_id = heapq.heappop(heap)
                current_trajectory = trajectory[trajectory_id]

                # update trajectory values and pointers
                trajectory_values[trajectory_id] = current_trajectory[plot_type][trajectory_pointers[trajectory_id]]
                individual_trajectories[trajectory_id].append(trajectory_values[trajectory_id])
                individual_times_finished[trajectory_id].append(times_finished)
                trajectory_pointers[trajectory_id] += 1
                if trajectory_pointers[trajectory_id] < len(current_trajectory[plot_type]):
                    heapq.heappush(heap,
                        (trajectory[trajectory_id]["times_finished"][trajectory_pointers[trajectory_id]], trajectory_id)
                    )

                # progress += 1
                # print("Progress:", (progress / total) * 100, " " * 20, end="\r" if progress != total else "\n")

                # populate plotting data
                if any(v is None for v in trajectory_values):
                    continue
                if finishing_times and np.isclose(times_finished, finishing_times[-1]):
                    [x.pop() for x in [center, upper, lower, finishing_times]]
                values = [v * value_multiplier for v in trajectory_values if v is not None]
                if agglomeration == "median":
                    center.append(np.median(values))
                    lower.append(np.percentile(values, int(50 - scale_uncertainty * 25)))
                    upper.append(np.percentile(values, int(50 + scale_uncertainty * 25)))
                elif agglomeration == "mean":
                    center.append(np.mean(values))
                    lower.append(-1 * scale_uncertainty * np.std(values) + center[-1])
                    upper.append(scale_uncertainty * np.std(values) + center[-1])
                finishing_times.append(times_finished)
                plot_empty = False
            label = ("%s: %s" % (prefix, config_name)) if prefix else config_name

            plot_data[label] = {
                "individual_trajectory": individual_trajectories,
                "individual_times_finished": individual_times_finished,
                "color": color,
                "linestyle": linestyle,
                "center": center,
                "lower": lower,
                "upper": upper,
                "finishing_times": finishing_times
            }
    return plot_empty, plot_data
    
def plot_trajectory(plot_data, instance_name, metric_name, font_size, do_label_rename, plt, plot_individual, plot_markers, plot_type):
    for label, d in plot_data.items():

        if do_label_rename:
            label = label_rename(label)
        
        if plot_individual and d["individual_trajectories"] and d["individual_times_finished"]:
            for individual_trajectory, individual_times_finished in zip(d["individual_trajectories"], d["individual_times_finished"]):
                plt.step(individual_times_finished, individual_trajectory, color=d["color"], where='post', linestyle=":", marker="x" if plot_markers else None)
        
        plt.step(d["finishing_times"], d["center"], color=d["color"], label=label, where='post', linestyle=d["linestyle"], marker="o" if plot_markers else None)
        plt.fill_between(d["finishing_times"], d["lower"], d["upper"], step="post", color=[(d["color"][0], d["color"][1], d["color"][2], 0.5)])
    plt.xlabel('wall clock time [s]', fontsize=font_size)
    plt.ylabel('incumbent %s %s' % (metric_name, plot_type), fontsize=font_size)
    plt.legend(loc='best', prop={'size': font_size})
    plt.title(instance_name, fontsize=font_size)

LABEL_RENAME = dict()
def label_rename(label):
    if label not in LABEL_RENAME:
        rename = input("Rename label %s to? (Leave empty for no rename) " % label)
        LABEL_RENAME[label] = rename if rename else label
    return LABEL_RENAME[label]
