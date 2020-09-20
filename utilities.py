import json
import os


def return_best_config(
        result_path: str,
        number_of_configs: int,
        seed: int = 11,
):

    result_folder = os.path.join(
        result_path,
        'hpo_run',
        f'{seed}',
    )

    best_test_accuracy = 0
    best_config_id = None
    index = 0
    with open(os.path.join(result_folder, 'results.json')) as result_file:
        for line in result_file:
            config_info = json.loads(line)
            config_id = config_info[0]
            result_info = config_info[3]['info']
            test_result = result_info[0]['test_result']
            final_test_accuracy = test_result[-1]

            if final_test_accuracy > best_test_accuracy:
                best_test_accuracy = final_test_accuracy
                best_config_id = config_id
            index += 1

            if index == number_of_configs:
                print("Max number of configs reached")
                break

    with open(os.path.join(result_folder, 'configs.json')) as config_file:
        for line in config_file:
            config = json.loads(line)
            config_id = config[0]
            if config_id == best_config_id:
                return config[1]
