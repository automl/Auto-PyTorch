import matplotlib.pyplot as plt
import argparse
import numpy as np
import os, sys
import heapq

hpbandster = os.path.abspath(os.path.join(__file__, '..', '..', 'submodules', 'HpBandSter'))
sys.path.append(hpbandster)

from hpbandster.core.result import logged_results_to_HBS_result

sum_of_weights_history = None


if __name__ == "__main__": 
    parser = argparse.ArgumentParser(description='Run benchmarks for autonet.')
    parser.add_argument("--only_summary", action="store_true", help="The maximum number of configs in the legend")
    parser.add_argument("--max_legend_size", default=5, type=int, help="The maximum number of datasets in the legend")
    parser.add_argument("--num_consider_in_summary", default=15, type=int, help="The number of datasets considered in the summary")
    parser.add_argument("weight_history_files", type=str, nargs="+", help="The files to plot")

    args = parser.parse_args()
    
    weight_deviation_history = list()
    weight_deviation_timestamps = list()
    
    #iterate over all files and lot the weight history
    for i, weight_history_file in enumerate(args.weight_history_files):
        print(i)
        plot_data = dict()
        with open(weight_history_file, "r") as f:
            for line in f:
                
                #read the data
                line = line.split("\t")
                if len(line) == 1 or not line[-1].strip():
                    continue
                data = line[-1]
                data = list(map(float, map(str.strip, data.split(","))))
                title = "\t".join(line[:-1]).strip()
                
                # and save it later for plotting
                plot_data[title] = data
                
        # only show labels for top datasets
        sorted_keys = sorted(plot_data.keys(), key=lambda x, d=plot_data: -d[x][-1] if x != "current" else -float("inf"))
        show_labels = set(sorted_keys[:args.max_legend_size])
        consider_in_summary = set(sorted_keys[:args.num_consider_in_summary])

        # parse results to get the timestamps for the weights
        x_axis = []
        try:
            r = logged_results_to_HBS_result(os.path.dirname(weight_history_file))
            sampled_configs = set()
            for run in sorted(r.get_all_runs(), key=(lambda run: run.time_stamps["submitted"])):
                if run.config_id not in sampled_configs:
                    x_axis.append(run.time_stamps["submitted"])
                sampled_configs |= set([run.config_id])
        except Exception as e:
            continue

        # do the plotting
        if not args.only_summary:
            for title, data in sorted(plot_data.items()):
                    plt.plot(x_axis, data[:len(x_axis)],
                        label=title if title in show_labels else None,
                        linestyle="-." if title == "current" else ("-" if title in show_labels else ":"),
                        marker="x")

            plt.legend(loc='best')
            plt.title(weight_history_file)
            plt.xscale("log")
            plt.show()

        # save data for summary
        for title, data in plot_data.items():
            if title in consider_in_summary:
                weight_deviation_history.append([abs(d - data[-1]) for d in data])
                weight_deviation_timestamps.append(x_axis)

    # plot summary
    weight_deviation_history = np.array(weight_deviation_history)
    weight_deviation_timestamps = np.array(weight_deviation_timestamps)
    
    # iterate over all weight deviation histories simultaneously, ordered by increasing timestamps
    history_pointers = [0] * weight_deviation_timestamps.shape[0]
    current_values = [None] * weight_deviation_timestamps.shape[0]
    heap = [(weight_deviation_timestamps[i, 0], weight_deviation_history[i, 0], i, 0)  # use heap to sort by timestamps
            for i in range(weight_deviation_timestamps.shape[0])]
    heapq.heapify(heap)
    # progress = 0
    # total = weight_deviation_timestamps.shape[0] * weight_deviation_timestamps.shape[1]

    times = []
    values = []
    while heap:
        time, v, i, p = heapq.heappop(heap)
        current_values[i] = v
        values.append(np.mean([v for v in current_values if v is not None]))
        times.append(time)
        history_pointers[i] += 1
        if p + 1 < weight_deviation_timestamps.shape[1]:
            heapq.heappush(heap, (weight_deviation_timestamps[i, p + 1], weight_deviation_history[i, p + 1], i, p + 1))
            
        # progress += 1
        # print("Progress Summary:", (progress / total) * 100, " " * 20, end="\r" if progress != total else "\n")

    plt.plot(times, values, marker="x")
    plt.title("weight deviation over time")
    plt.xscale("log")
    plt.show()
