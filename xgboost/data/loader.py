from utilities import get_dataset_openml, get_dataset_split

class Loader():

    def __init__(self, task_id, val_fraction=0.2, test_fraction=0.2, seed=11):

        dataset = get_dataset_openml(task_id)
        self.categorical_information, self.splits = get_dataset_split(
            dataset,
            val_fraction=val_fraction,
            test_fraction=test_fraction,
            seed=seed,
        )
        self.dataset_id = dataset.dataset_id


    def get_splits(self):

        return self.splits

    def get_dataset_id(self):

        return self.dataset_id
