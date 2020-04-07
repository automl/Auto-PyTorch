import time, os, shutil
from hpbandster.core.result import json_result_logger

from autoPyTorch.utils.tensorboard_logging import get_tb_logger


class bohb_logger(json_result_logger):
    def __init__(self, constant_hyperparameter, directory, overwrite=False):
        super(bohb_logger, self).__init__(directory, overwrite)
        self.constants = constant_hyperparameter

    
    def new_config(self, config_id, config, config_info):
        import json
        if not config_id in self.config_ids:
            self.config_ids.add(config_id)

        full_config = dict()
        full_config.update(self.constants)
        full_config.update(config)

        with open(self.config_fn, 'a') as fh:
            fh.write(json.dumps([config_id, full_config, config_info]))
            fh.write('\n')


class tensorboard_logger(object):
    def __init__(self, pipeline_config, constant_hyperparameter, global_results_dir):
        self.start_time = time.time()

        b = pipeline_config['max_budget']
        budgets = []
        while b >= pipeline_config['min_budget']:
            budgets.append(int(b))
            b /= pipeline_config['eta']

        self.incumbent_results = {b: 0 for b in budgets}
        self.mean_results = {b: [0, 0] for b in budgets}

        self.constants = constant_hyperparameter
        self.results_logged = 0
        self.seed = pipeline_config['random_seed']
        self.max_budget = pipeline_config['max_budget']
        self.global_results_dir = global_results_dir

        self.keep_only_incumbent_checkpoints = pipeline_config['keep_only_incumbent_checkpoints']

        self.incumbent_configs_dir = os.path.join(pipeline_config['result_logger_dir'], 'incumbents')
        self.status_dir = pipeline_config['result_logger_dir']
        self.run_name = '-'.join(pipeline_config['run_id'].split('-')[1:])
        os.makedirs(self.incumbent_configs_dir, exist_ok=True)


    def new_config(self, config_id, config, config_info):
        pass

    def __call__(self, job):
        import json

        id = job.id
        budget = int(job.kwargs['budget'])
        config = job.kwargs['config']
        # timestamps = job.timestamps
        result = job.result
        # exception = job.exception

        if result is None:
            return

        self.results_logged += 1

        writer = get_tb_logger()
        writer.add_scalar('BOHB/all_results', result['loss'] * -1, self.results_logged)

        if budget not in self.incumbent_results or result['loss'] < self.incumbent_results[budget]:
            self.incumbent_results[budget] = result['loss']
            
            full_config = dict()
            full_config.update(self.constants)
            full_config.update(config)

            refit_config = dict()
            refit_config['budget'] = budget
            refit_config['seed'] = self.seed
            
            refit_config['incumbent_config_path'] = os.path.join(self.incumbent_configs_dir, 'config_' + str(budget) + '.json')
            with open(refit_config['incumbent_config_path'], 'w+') as f:
                f.write(json.dumps(full_config, indent=4, sort_keys=True))
            
            with open(os.path.join(self.incumbent_configs_dir, 'result_' + str(budget) + '.json'), 'w+') as f:
                f.write(json.dumps([job.id, job.kwargs['budget'], job.timestamps, job.result, job.exception], indent=4, sort_keys=True))

            checkpoints, refit_config['dataset_order'] = get_checkpoints(result['info']) or ([],None)
            refit_config['incumbent_checkpoint_paths'] = []
            for i, checkpoint in enumerate(checkpoints):
                dest = os.path.join(self.incumbent_configs_dir, 'checkpoint_' + str(budget) + '_' + str(i) + '.pt' if len(checkpoints) > 1 else 'checkpoint_' + str(budget) + '.pt')
                if os.path.exists(dest):
                    os.remove(dest)
                if self.keep_only_incumbent_checkpoints:
                    shutil.move(checkpoint, dest)
                else:
                    shutil.copy(checkpoint, dest)
                refit_config['incumbent_checkpoint_paths'].append(dest)

            refit_path = os.path.join(self.incumbent_configs_dir, 'refit_config_' + str(budget) + '.json')
            with open(refit_path, 'w+') as f:
                f.write(json.dumps(refit_config, indent=4, sort_keys=True))

            if budget >= self.max_budget and self.global_results_dir is not None:
                import autoPyTorch.utils.thread_read_write as thread_read_write
                import datetime

                dataset_names = sorted([os.path.splitext(os.path.split(info['dataset_path'])[1])[0] for info in result['info']])
                suffix = ''
                if len(result['info']) > 1:
                    suffix += '+[' + ', '.join(dataset_names) + ']'
                if budget > self.max_budget:
                    suffix += '+Refit'

                for info in result['info']:
                    thread_read_write.update_results(self.global_results_dir, {
                        'name': os.path.splitext(os.path.split(info['dataset_path'])[1])[0] + suffix, 
                        'result': round(info['val_top1'], 2), 
                        'seed': self.seed,
                        'refit_config': refit_path, 
                        'text': "{0}/{1} -- {2}".format(
                            round(info['val_datapoints'] * (info['val_top1'] / 100)),
                            info['val_datapoints'],
                            round(budget / len(result['info'])))
                        })

        if self.keep_only_incumbent_checkpoints and get_checkpoints(result['info']):
            for checkpoint in get_checkpoints(result['info'])[0]:
                if os.path.exists(checkpoint):
                    os.remove(checkpoint)

        if budget not in self.mean_results:
            self.mean_results[budget] = [result['loss'], 1]
        else:
            self.mean_results[budget][0] += result['loss']
            self.mean_results[budget][1] += 1

        for b, loss in self.incumbent_results.items():
            writer.add_scalar('BOHB/incumbent_results_' + str(b), loss * -1, self.mean_results[b][1])

        for b, (loss, n) in self.mean_results.items():
            writer.add_scalar('BOHB/mean_results_' + str(b), loss * -1 / n if n > 0 else 0, n)

        status = dict()
        for b, loss in self.incumbent_results.items():
            budget_status = dict()
            budget_status['incumbent'] = loss * -1
            mean_res = self.mean_results[b]
            budget_status['mean'] = mean_res[0] / mean_res[1] * -1 if mean_res[1] > 0 else 0
            budget_status['configs'] = mean_res[1]
            status['budget: ' + str(b)] = budget_status

        import datetime
        status["runtime"] = str(datetime.timedelta(seconds=time.time() - self.start_time))

        with open(os.path.join(self.status_dir, 'bohb_status.json'), 'w+') as f:
            f.write(json.dumps(status, indent=4, sort_keys=True))


def get_checkpoints(info):
    if not isinstance(info, list):
        if 'checkpoint' in info:
            return [info['checkpoint']]
        return []

    checkpoints = []
    dataset_order = []
    for subinfo in info:
        if 'checkpoint' in subinfo:
            checkpoints.append(subinfo['checkpoint'])
            dataset_order.append(subinfo['dataset_id'])
    return checkpoints, dataset_order

class combined_logger(object):
    def __init__(self, *loggers):
        self.loggers = loggers

    def new_config(self, config_id, config, config_info):
        for logger in self.loggers:
            logger.new_config(config_id, config, config_info)

    def __call__(self, job):
        for logger in self.loggers:
            logger(job)
        
def get_incumbents(directory):
    
    incumbents = os.path.join(directory, 'incumbents')

    if not os.path.exists(incumbents):
        return None

    import re
    file_re = [
        re.compile('config_([0-9]+).json'),
        re.compile('refit_config_([0-9]+).json'),
        re.compile('result_([0-9]+).json'),
        re.compile('checkpoint_([0-9]+).*.pt'),
    ]

    incumbent_files = [[] for _ in range(len(file_re))]
    for filename in sorted(os.listdir(incumbents)):
        for i, reg in enumerate(file_re):
            match = reg.match(filename)
            
            if match:
                budget = int(match.group(1))
                inc_file = os.path.join(incumbents, filename)
                incumbent_files[i].append([budget, inc_file])

    return incumbent_files


def get_refit_config(directory):
    _, refit_configs, _, _ = get_incumbents(directory)
    refit_config = max(refit_configs, key=lambda x: x[0]) #get config of max budget
    return refit_config[1]
