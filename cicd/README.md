###########################################################
# Continuous integration and continuous delivery/deployment
###########################################################

This part of the code is tasked to make sure that we can perform reliable NAS.
To this end, we rely on pytest to run some long running configurations from both
the greedy portafolio and the default configuration.

```
python -m pytest cicd/test_preselected_configs.py -vs
```
