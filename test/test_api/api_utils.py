import os
import glob


def print_debug_information(automl):

    # Log file path
    log_file = glob.glob(os.path.join(
        automl._backend.temporary_directory, 'AutPyTorch*.log'))[0]

    include_messages = ['INFO', 'DEBUG', 'WARN',
                        'CRITICAL', 'ERROR', 'FATAL']

    # There is a lot of content in the log files. Only
    # parsing the main message and ignore the metalearning
    # messages
    try:
        with open(log_file) as logfile:
            content = logfile.readlines()

        # Get the messages to debug easier!
        content = [line for line in content if any(
            msg in line for msg in include_messages
        ) and 'metalearning' not in line]

    except Exception as e:
        return str(e)

    # Also add the run history if any
    if hasattr(automl, 'runhistory') and hasattr(automl.runhistory, 'data'):
        for k, v in automl.runhistory_.data.items():
            content += ["{}->{}".format(k, v)]
    else:
        content += ['No RunHistory']

    # Also add the ensemble history if any
    if len(automl.ensemble_performance_history) > 0:
        content += [str(h) for h in automl.ensemble_performance_history]
    else:
        content += ['No Ensemble History']

    return os.linesep.join(content)
