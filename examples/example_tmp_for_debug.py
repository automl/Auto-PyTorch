import os

import sklearn.datasets
import time
import shutil

from autoPyTorch.utils.backend import create

from autoPyTorch.api.tabular_classification import TabularClassificationTask
from autoPyTorch.datasets.resampling_strategy import (
    CrossValTypes,
)
import re


from pathlib import Path


class DisplayablePath(object):
    display_filename_prefix_middle = '├──'
    display_filename_prefix_last = '└──'
    display_parent_prefix_middle = '    '
    display_parent_prefix_last = '│   '

    def __init__(self, path, parent_path, is_last):
        self.path = Path(str(path))
        self.parent = parent_path
        self.is_last = is_last
        if self.parent:
            self.depth = self.parent.depth + 1
        else:
            self.depth = 0

    @property
    def displayname(self):
        if self.path.is_dir():
            return self.path.name + '/'
        return self.path.name

    @classmethod
    def make_tree(cls, root, parent=None, is_last=False, criteria=None):
        root = Path(str(root))
        criteria = criteria or cls._default_criteria

        displayable_root = cls(root, parent, is_last)
        yield displayable_root

        children = sorted(list(path
                               for path in root.iterdir()
                               if criteria(path)),
                          key=lambda s: str(s).lower())
        count = 1
        for path in children:
            is_last = count == len(children)
            if path.is_dir():
                yield from cls.make_tree(path,
                                         parent=displayable_root,
                                         is_last=is_last,
                                         criteria=criteria)
            else:
                yield cls(path, displayable_root, is_last)
            count += 1

    @classmethod
    def _default_criteria(cls, path):
        return True

    @property
    def displayname(self):
        if self.path.is_dir():
            return self.path.name + '/'
        return self.path.name

    def displayable(self):
        if self.parent is None:
            return self.displayname

        _filename_prefix = (self.display_filename_prefix_last
                            if self.is_last
                            else self.display_filename_prefix_middle)

        parts = ['{!s} {!s}'.format(_filename_prefix,
                                    self.displayname)]

        parent = self.parent
        while parent and parent.parent is not None:
            parts.append(self.display_parent_prefix_middle
                         if parent.is_last
                         else self.display_parent_prefix_last)
            parent = parent.parent

        return ''.join(reversed(parts))


def slugify(text):
    return re.sub(r'[\[\]]+', '-', text.lower())


test_dir = os.path.dirname(__file__)
tmp = slugify(os.path.join(
    test_dir, '.tmp__%s' % __file__))
output = slugify(os.path.join(
    test_dir, '.output__%s' % __file__))

for dir in (tmp, output):
    for i in range(10):
        if os.path.exists(dir):
            try:
                shutil.rmtree(dir)
                break
            except OSError:
                time.sleep(1)

# Make sure the folders we wanna create do not already exist.
backend = create(
    tmp,
    output,
    delete_tmp_folder_after_terminate=True,
    delete_output_folder_after_terminate=True,
)

openml_id = 40981
resampling_strategy = CrossValTypes.k_fold_cross_validation
X, y = sklearn.datasets.fetch_openml(
    data_id=int(openml_id),
    return_X_y=True, as_frame=True
)
X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
    X, y, random_state=1)

# Search for a good configuration
estimator = TabularClassificationTask(
    backend=backend,
    resampling_strategy=resampling_strategy,
)

estimator.search(
    X_train=X_train, y_train=y_train,
    X_test=X_test, y_test=y_test,
    optimize_metric='accuracy',
    total_walltime_limit=150,
    func_eval_time_limit=50,
    traditional_per_total_budget=0
)

# Search for an existing run key in disc. A individual model might have
# a timeout and hence was not written to disc
for i, (run_key, value) in enumerate(estimator.run_history.data.items()):
    if i == 0:
        # Ignore dummy run
        continue
    if 'SUCCESS' not in str(value.status):
        continue

    run_key_model_run_dir = estimator._backend.get_numrun_directory(
        estimator.seed, run_key.config_id, run_key.budget)
    if os.path.exists(run_key_model_run_dir):
        break


model_file = os.path.join(
    run_key_model_run_dir,
    f"{estimator.seed}.{run_key.config_id}.{run_key.budget}.cv_model"
)
if not os.path.exists(model_file):
    paths = DisplayablePath.make_tree(run_key_model_run_dir)
    for path in paths:
        print(path.displayable())
