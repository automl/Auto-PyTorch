__author__ = "Max Dippel, Michael Burkart and Matthias Urban"
__version__ = "0.0.1"
__license__ = "BSD"

import os

class ConfigFileParser():
    """
    Class to parse config files
    """

    def __init__(self, config_options=[], verbose=False):
        """
        Initialize to ConfigFileParser.
        
        Parameters:
            config_options: A list of ConfigOptions, which the ConfigFileParser should be able to parse
        """
        self.config_options = {option.name: option for option in config_options}
        self.verbose = verbose
        
    def add_option(self, option):
        """
        Add a ConfigOption to the ConfigParser.
        """
        self.config_options[option.name] = option

    @staticmethod
    def read_key_values_from_file(filename, delimiter='='):
        key_values = dict()

        if (filename is None):
            return key_values
            
        # open the config file
        with open(filename, "r") as configfile:
            for line in configfile:
                key, value = map(lambda x: x.strip(), line.split(delimiter))
                key_values[key] = value
        return key_values
        
    def read(self, filename, key_values_dict=None):
        """
        Read a config file.
        
        Parameters:
            filename: The file name of the config file.
            
        Result:
            A dictionary containing the read values for the ConfigOptions.
        """
        # parse benchmark.txt
        autonet_home = self.get_autonet_home()

        key_values_dict = key_values_dict or ConfigFileParser.read_key_values_from_file(filename)
        config = dict()
            
        # open the config file
        for key, value in key_values_dict.items():
            if (key not in self.config_options):
                raise ValueError("Config key '" + key + "' is not a valid autonet config option")

            option = self.config_options[key]

            # parse list configs
            values = [value]
            if option.list:
                value = value.strip("[]")
                if not value.strip():
                    values = []
                else:
                    values = list(map(lambda x: x.strip("'\" "), value.split(",")))
                
            # convert the values
            converted_values = []
            for value in values:
                type_list = option.type if isinstance(option.type, list) else [option.type]
                for type_conversion in type_list:
                    # convert relative directories to absolute ones
                    if type_conversion == "directory" and not os.path.isabs(value):
                        value = os.path.abspath(os.path.join(autonet_home, value))
                    elif isinstance(type_conversion, dict):
                        value = type_conversion[value]
                    elif type_conversion != "directory":
                        value = type_conversion(value)
                converted_values.append(value)
            config[key] = converted_values if option.list else converted_values[0]
        return config
    
    def check_required(self, config):
        """
        Check if the given config is required.
        """
        for key, option in self.config_options.items():
            if option.required:
                assert key in config, key + " must be specified"
    
    def set_defaults(self, config, throw_error_if_invalid=True):
        """
        Set the default values for the ConfigOptions which are not specified in the given config.
        """
        default_depends_configs = []
        for key, option in self.config_options.items():
            if key not in config:
                if option.depends:
                    default_depends_configs.append((key, option.default))
                else:
                    config[key] = option.default
        
        # set the value for those configs, that have not been specified and whose default value depends on other values
        for key, default in default_depends_configs:
            config[key] = default(config)

        try:
            self.check_validity(config)
        except Exception as e:
            print(e)
            if throw_error_if_invalid:
                raise e
        return config

    def check_validity(self, config):
        if (len(config) != len(self.config_options)):
            additional_keys = set(config.keys()).difference(self.config_options.keys())
            if (len(additional_keys) > 0):
                raise ValueError("The following unknown config options have been defined: " + str(additional_keys))
            missing_keys = set(self.config_options.keys()).difference(config.keys())
            if (len(missing_keys) > 0):
                raise ValueError("The following config options have not been assigned: " + str(missing_keys))
            raise NotImplementedError()

        for option_name, option in self.config_options.items():
            if (option_name not in config):
                raise ValueError("Config option '" + option_name + "' has not been assigned.")

            choices = option.choices
            if (choices is None):
                continue
                
            value = config[option_name]
            if (option.list):
                if (not isinstance(value, list)):
                    raise ValueError("Config option " + option_name + " has been assigned with value '" + str(value) + "', list required")
                diff = set(value).difference(choices)
                if (len(diff) > 0):
                    raise ValueError("Config option " + option_name + " contains following invalid values " + str(diff) + ", chose a subset of " + str(choices))
            else:
                if (option.type is int or option.type is float):
                    if (value < choices[0] or value > choices[1]):
                        raise ValueError("Config option " + option_name + " has been assigned with value '" + str(value) + "' which is not in required interval [" + choices[0] + ", " + choices[1] + "]")
                else:
                    if (value not in choices):
                        raise ValueError("Config option " + option_name + " has been assigned with value '" + str(value) + "', only values in " + str(choices) + " are allowed")

    def print_help(self, max_column_width=40):
        columns = ["name", "default", "choices", "type"]
        default = self.set_defaults({})
        column_width = {c: len(c) for c in columns}
        format_string = dict()
        num_lines = dict()
        
        for option in self.config_options.values():
            num_lines[option] = 1
            for column in columns:
                value = getattr(option, column) if column != "default" else default[option.name]
                if isinstance(value, list) and len(value) > 0:
                    column_width[column] = max(column_width[column],
                                               max(map(lambda x: len(str(x)) + 2, value)))
                    num_lines[option] = max(num_lines[option], len(value))
                elif isinstance(value, list):
                    column_width[column] = max(column_width[column], 2)
                else:
                    column_width[column] = max(column_width[column], len(str(value)))
                format_string[column] = "{0: <" + str(min(max_column_width, column_width[column]) + 1) + "}"

        for column in columns:
            print(format_string[column].format(column), end="")
        print()
        print("=" * sum(map(lambda x: min(x, max_column_width) + 1, column_width.values())))

        for option in sorted(self.config_options.values(), key=lambda x:x.name):
            for i in range(num_lines[option]):
                for column in columns:
                    value = getattr(option, column) if column != "default" else default[option.name]
                    if isinstance(value, list) and i < len(value):
                        prefix = "[" if i == 0 else " "
                        suffix = "]" if i == (len(value) - 1) else ","
                        print(format_string[column].format(prefix + str(value[i])[:max_column_width-2] + suffix), end="")
                    elif isinstance(value, list) and i == 0:
                        print(format_string[column].format("[]"), end="")
                    elif i == 0:
                        print(format_string[column].format(str(value)[:max_column_width]), end="")
                    else:
                        print(format_string[column].format(""), end="")
                print()
            if option.info is not None:
                print("\tinfo:", option.info)
            print("-" * sum(map(lambda x: min(x, max_column_width) + 1, column_width.values())))
    
    @staticmethod
    def get_autonet_home():
        """ Get the home directory of autonet """
        if "AUTONET_HOME" in os.environ:
            return os.environ["AUTONET_HOME"]
        return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))