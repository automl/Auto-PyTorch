import argparse
import os as os
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from IPython import embed


def load_bench(path):
    with open(path, "r") as f:
        data = json.load(f)
    return data

def get_config_hash(config):
    config_without_tmax = {k:v for k,v in config.items() if not "T_max" in k}
    return hash(config_without_tmax.__repr__())

def create_config_hashdict(data):
    config_hdict = {}

    ds_name = "APSFailure"

    for config_id in data[ds_name]:
        config = data[ds_name][config_id]["12"]["config"]
        config_hash = get_config_hash(config)
        config_hdict[config_hash] = config_id

    return config_hdict

def combine_data(main_data, add_data, budget, seed):
    
    print("==> Creating config hashdict...")
    config_hdict = create_config_hashdict(main_data)
    
    print("==> Removing empty datasets")
    ds_names = list(main_data.keys())
    for ds in ds_names:
        if len(main_data[ds])==0:
            del main_data[ds]

    print("==> Adding data...")
    for ds_name in tqdm(add_data.keys()):
        for cid in add_data[ds_name].keys():
            log = add_data[ds_name][cid]["log"]
            results = add_data[ds_name][cid]["results"]
            config = add_data[ds_name][cid]["config"]
            config_hash = get_config_hash(config)

            main_id = config_hdict[config_hash]
            try:
                main_data[ds_name][main_id][str(budget)]["results"][str(seed)] = results
                main_data[ds_name][main_id][str(budget)]["log"][str(seed)] = log
            except Exception as e:
                if isinstance(e, KeyError):
                    pass
                else:
                    raise e

    return main_data


if __name__=="__main__":

    main_path = "bench.json"
    add_path = "/home/zimmerl/LCBench_analysis/LCBench/cached/data_2k_lw.json"

    print("==> Loading main data...")
    main_data = load_bench(main_path)
    print("==> Loading add data...")
    add_data = load_bench(add_path)

    main_data = combine_data(main_data, add_data, budget=50, seed=1)

    save_dir = "bench_full.json"

    print("==> Saving...")
    # Save
    with open(save_dir, "w") as f:
        json.dump(main_data, f)

    print("==> Done.")
