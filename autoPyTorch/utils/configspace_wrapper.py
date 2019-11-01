__author__ = "Max Dippel, Michael Burkart and Matthias Urban"
__version__ = "0.0.1"
__license__ = "BSD"

import pprint
    
class ConfigWrapper(object):
    delimiter = ':'

    def __init__(self, config_prefix, config):
        """A wrapper for hyperparameter configs that are specified with a prefix (add_configspace(prefix=...)).
        The wrapper will provide key access without having to know/specify the prefix of the respective hyperparameter.
        
        Arguments:
            config_prefix {string} -- prefix of keys
            config {dict} -- hyperparameter config
        """

        self.config_prefix = config_prefix + ConfigWrapper.delimiter
        self.config = config

    def __getitem__(self, key):
        if ((self.config_prefix + key) not in self.config):
            pprint.pprint(self.config)
        return self.config[self.config_prefix + key]

    def __iter__(self):
        for k in self.config.__iter__():
            if not k.startswith(self.config_prefix):
                continue
            yield k[len(self.config_prefix):]

    def __str__(self):
        return str(self.config)
    
    def __contains__(self, key):
        return (self.config_prefix + key) in self.config

    def update(self, update_dict):
        self.config.update({"%s%s" % (self.config_prefix, key) : value for key, value in update_dict.items()})
    
    def get_dictionary(self):
        result = dict()
        config = self.config if isinstance(self.config, dict) else self.config.get_dictionary()
        for key, value in config.items():
            if key.startswith(self.config_prefix):
                result[key[len(self.config_prefix):]] = value
        return result
