import os
from autonet.utils.config.config_option import ConfigOption, to_bool
from autonet.pipeline.base.pipeline_node import PipelineNode
import numpy as np
import logging

class PlotTrajectories(PipelineNode):

    def fit(self, pipeline_config, trajectories, train_metrics, instance):
        # these imports won't work on meta
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_pdf import PdfPages

        plot_logs = pipeline_config['plot_logs'] or train_metrics
        output_folder = pipeline_config['output_folder']
        instance_name = os.path.basename(instance).split(".")[0]

        if output_folder and not os.path.exists(output_folder):
            os.mkdir(output_folder)

        for log in plot_logs:
            if log not in trajectories.keys():
                logging.getLogger('benchmark').warn('No trajectory found for ' + log)

        # iterate over all incumbent trajectories for each metric
        for i, (metric_name, run_trajectories) in enumerate(trajectories.items()):
            if metric_name not in plot_logs:
                continue
            
            # prepare pdf
            if output_folder is not None:
                pdf_destination = os.path.join(output_folder, instance_name + '_' + metric_name + '.pdf')
                pp = PdfPages(pdf_destination)

            # create figure
            figure = plt.figure(i)
            if not plot_trajectory(instance_name,
                                   metric_name,
                                   run_trajectories,
                                   pipeline_config['agglomeration'],
                                   pipeline_config['scale_uncertainty'],
                                   pipeline_config['font_size']):
                logging.getLogger('benchmark').warn('Not showing empty plot for ' + instance)
                continue

            # show or save
            if output_folder is None:
                logging.getLogger('benchmark').info('Showing plot for ' + instance)
                plt.show()
            else:
                logging.getLogger('benchmark').info('Saving plot for ' + instance + ' at ' + pdf_destination)
                pp.savefig(figure)
                pp.close()
                plt.close(figure)
        return dict()
    

    def get_pipeline_config_options(self):
        options = [
            ConfigOption('plot_logs', default=None, type='str', list=True),
            ConfigOption('output_folder', default=None, type='directory'),
            ConfigOption('agglomeration', default='mean', choices=['mean', 'median']),
            ConfigOption('scale_uncertainty', default=1, type=float),
            ConfigOption('font_size', default=12, type=int)
        ]
        return options

def plot_trajectory(instance_name, metric_name, run_trajectories, agglomeration, scale_uncertainty, font_size):
    # iterate over the incumbent trajectories of the different runs
    import matplotlib.pyplot as plt
    cmap = plt.get_cmap('jet')
    for i, (config_name, trajectory) in enumerate(run_trajectories.items()):
        color = cmap(i / (len(run_trajectories)))

        trajectory_pointers = [0] * len(trajectory)  # points to current entry of each trajectory
        trajectory_values = [None] * len(trajectory)  # list of current values of each trajectory

        # data to plot
        center = []
        lower = []
        upper = []
        finishing_times = []

        # iterate simultaneously over all trajectories with increasing finishing times
        while any(trajectory_pointers[j] < len(trajectory[j]["config_ids"]) for j in range(len(trajectory))):

            # get trajectory with lowest finishing times
            times_finished, trajectory_id = min([(trajectory[j]["times_finished"][trajectory_pointers[j]], j)
                for j in range(len(trajectory)) if trajectory_pointers[j] < len(trajectory[j]["config_ids"])])
            current_trajectory = trajectory[trajectory_id]

            # update trajectory values and pointers
            trajectory_values[trajectory_id] = current_trajectory["losses"][trajectory_pointers[trajectory_id]]
            trajectory_pointers[trajectory_id] += 1

            # populate plotting data
            values = [v * (-1 if current_trajectory["flipped"] else 1) for v in trajectory_values if v is not None]
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

        # insert into plot
        plt.step(finishing_times, center, color=color, label=config_name, where='post')
        color = (color[0], color[1], color[2], 0.5)
        plt.fill_between(finishing_times, lower, upper, step="post", color=[color])
    plt.xlabel('wall clock time [s]', fontsize=font_size)
    plt.ylabel('incumbent ' + metric_name, fontsize=font_size)
    plt.legend(loc='best', prop={'size': font_size})
    plt.title(instance_name, fontsize=font_size)
    plt.xscale("log")
    return not plot_empty