import openml
from pprint import pprint
from autoPyTorch import AutoNetClassification
from sklearn.metrics import accuracy_score


# get OpenML task by its ID
task = openml.tasks.get_task(task_id=32)
X, y = task.get_X_and_y()
ind_train, ind_test = task.get_train_test_split_indices()


# run Auto-PyTorch
autoPyTorch = AutoNetClassification("tiny_cs",  # config preset
                                    log_level='info',
                                    max_runtime=300,
                                    min_budget=30,
                                    max_budget=90)

autoPyTorch.fit(X[ind_train], y[ind_train], validation_split=0.3)


# predict
y_pred = autoPyTorch.predict(X[ind_test])

print("Accuracy score", accuracy_score(y[ind_test], y_pred))


# print network configuration
pprint(autoPyTorch.fit_result["optimized_hyperparameter_config"])
