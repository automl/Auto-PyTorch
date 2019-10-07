
from autoPyTorch.utils.config.config_option import ConfigOption, to_bool
from autoPyTorch.pipeline.base.pipeline_node import PipelineNode
from autoPyTorch.utils.configspace_wrapper import ConfigWrapper

import re
import os
import pandas as pd
import math
import numpy as np


class ApplyUserUpdates(PipelineNode):

    def fit(self, pipeline_config, autonet):

        path = pipeline_config['user_updates_config']
        if path is None:
            return dict()

        if not os.path.exists(path):
            raise ValueError('Invalid path: ' + path)

        data = np.array(pd.read_csv(path, header=None, sep=';'))

        for row in data:
            name, value_range, is_log = row[0].strip(), self.string_to_list(str(row[1])), to_bool(row[2].strip())
            name_split = name.split(ConfigWrapper.delimiter)
            autonet.pipeline[name_split[0]]._update_hyperparameter_range(ConfigWrapper.delimiter.join(name_split[1:]), value_range, is_log, check_validity=False)

        # print(autonet.get_hyperparameter_search_space())

        return { 'autonet': autonet }

    
    def get_pipeline_config_options(self):
        options = [
            ConfigOption("user_updates_config", default=None, type='directory'),
        ]
        return options

    def string_to_list(self, string):
        pattern = "\[(.*)\]"
        match = re.search(pattern, string)

        if match is None:
            # no list > make constant range
            match = re.search(pattern, '[' + string + ',' + string + ']')

        if match is None:
            raise ValueError('No valid range specified got: ' + string)

        lst = map(self.try_convert, match.group(1).split(','))
        return list(lst)

    def try_convert(self, string):
        string = string.strip()
        try:
            return int(string)
        except:
            try:
                return float(string)
            except:
                if string == 'True':
                    return True
                if string == 'False':
                    return False
                return string
        


