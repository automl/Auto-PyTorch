
import os, json, re
from autoPyTorch.utils.config.config_file_parser import ConfigFileParser
from autoPyTorch.utils.config.config_option import ConfigOption
from autoPyTorch.utils.loggers import get_incumbents

def get_child_directories(root):
    if not os.path.exists(root):
        return []
    return [os.path.join(root, path) for path in os.listdir(root) if os.path.isdir(os.path.join(root, path))]

def get_all_run_directories():
    root = ConfigFileParser.get_autonet_home()
    root = os.path.join(root, 'benchmark_results_cluster')
    run_dirs = []
    for base in get_child_directories(root):
        for subdir in get_child_directories(base):
            for run_dir in get_child_directories(subdir):
                if os.path.basename(run_dir).startswith('run_'):
                    run_dirs.append(run_dir)
    return run_dirs
    
def get_filtered_run_dirs(regex):
    regex = re.compile(regex)
    return [run_dir for run_dir in get_all_run_directories() if regex.match(run_dir) and os.path.exists(os.path.join(run_dir, 'results.json'))]

class RunInfo():
    def __init__(self, run_dir):
        self.dir = run_dir
        self.auto_config_dir = os.path.join(run_dir, 'autonet.config')
        self.configs_dir = os.path.join(run_dir, 'configs.json')
        self.results_dir = os.path.join(run_dir, 'results.json')
        self.status_dir = os.path.join(run_dir, 'bohb_status.json')
        self.instance_dir = os.path.join(run_dir, 'instance.info')
        self.incumbent_files = get_incumbents(run_dir)
        self.refit_dir = os.path.join(run_dir, 'refit')
        self.root = os.path.abspath(os.path.join(self.dir, os.pardir))

    def get_refit_info(self):
        return RunInfo(self.refit_dir)

    def is_valid(self):
        return  os.path.exists(self.auto_config_dir) and \
                os.path.exists(self.configs_dir) and \
                os.path.exists(self.results_dir) and \
                os.path.exists(self.status_dir) and \
                os.path.exists(self.instance_dir)

    def get_highest_budget_incumbent_dirs(self):
        return [max(configs, key=lambda x: x[0])[1] for configs in self.incumbent_files if len(configs) > 0]

    def get_incumbent_config(self):
        with open(self.get_highest_budget_incumbent_dirs()[0], 'r') as f:
            return json.load(f)

    def get_incumbent_refit_config(self):
        with open(self.get_highest_budget_incumbent_dirs()[1], 'r') as f:
            return json.load(f)

    def get_incumbent_result(self):
        with open(self.get_highest_budget_incumbent_dirs()[2], 'r') as f:
            return json.load(f)
    
    def get_incumbent_result_info(self):
        return self.get_incumbent_result()[3]['info']

    def get_incumbent_checkpoint(self):
        with open(self.get_highest_budget_incumbent_dirs()[3], 'r') as f:
            return json.load(f)

    def get_dataset_paths(self):
        parser = ConfigFileParser([
            ConfigOption('path', type=str, list=True)
        ])
        return parser.read(self.instance_dir, silent=True)['path']

    def get_validation_results(self):
        result_info = self.get_incumbent_result_info()
        simple_results = []
        for result in result_info:
            id = result['dataset_id'] if 'dataset_id' in result else 0
            simple_results.append([id, result['dataset_path'], result['val_top1']])
        return [[x[1], x[2]] for x in sorted(simple_results, key=lambda res: res[0])]

    def get_seed(self):
        parser = ConfigFileParser([
            ConfigOption('random_seed', type=int)
        ])
        return parser.read(self.auto_config_dir, silent=True)['random_seed']



def get_run_infos(run_dir_regex):
    run_dirs = get_filtered_run_dirs(run_dir_regex)
    return [RunInfo(run_dir) for run_dir in run_dirs]

def get_global_results():
    root = ConfigFileParser.get_autonet_home()
    results = os.path.join(root, 'results.json')
    with open(results, 'r') as f:
        return json.load(f)

def get_dataset_name_dict():
    root = ConfigFileParser.get_autonet_home()
    name_dict = os.path.join(root, 'dataset_names.json')
    with open(name_dict, 'r') as f:
        return json.load(f)

def get_dataset_name_dict_inv():
    return {value: key for key, value in get_dataset_name_dict().items()}

def get_dataset_name(path):
    return os.path.splitext(os.path.split(path)[1])[0]

def get_dataset_id(path):
    name_dict = get_dataset_name_dict()
    name = get_dataset_name(path)
    return name_dict[name] if name in name_dict else name

def dataset_id_to_name(id):
    id_dict = get_dataset_name_dict_inv()
    return id_dict[id] if id in id_dict else id

def incumbent_config_path_to_run_dir(path):
    return os.path.abspath(os.path.join(path, os.pardir, os.pardir))