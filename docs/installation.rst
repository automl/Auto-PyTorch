:orphan:

.. _installation:

============
Installation
============

System requirements
===================

Auto-PyTorch has the following system requirements:

* Linux operating system (for example Ubuntu) `(get Linux here) <https://www.wikihow.com/Install-Linux>`_,
* Python (>=3.6) `(get Python here) <https://www.python.org/downloads/>`_.
* C++ compiler (with C++11 supports) `(get GCC here) <https://www.tutorialspoint.com/How-to-Install-Cplusplus-Compiler-on-Linux>`_ and
* SWIG (version 3.0.* is required; >=4.0.0 is not supported) `(get SWIG here) <http://www.swig.org/survey.html>`_.

Installing Auto-Pytorch
=======================

PyPI Installation
-----------------

.. code:: bash
    pip install autoPyTorch

Auto-PyTorch for Time Series Forecasting requires additional dependencies

.. code:: bash
    pip install autoPyTorch[forecasting]


Manual Installation
-------------------

.. code:: bash

    # Following commands assume the user is in a cloned directory of Auto-Pytorch

    # We also need to initialize the automl_common repository as follows
    # You can find more information about this here:
    # https://github.com/automl/automl_common/
    git submodule update --init --recursive

    # Create the environment
    conda create -n autopytorch python=3.8
    conda activate autopytorch
    conda install swig
    cat requirements.txt | xargs -n 1 -L 1 pip install
    python setup.py install

Similarly, Auto-PyTorch for time series forecasting requires additional dependencies

.. code:: bash
    git submodule update --init --recursive

    conda create -n auto-pytorch python=3.8
    conda activate auto-pytorch
    conda install swig
    pip install -e[forecasting]


Docker Image
============
A Docker image is also provided on dockerhub. To download from dockerhub,
use:

.. code:: bash

    docker pull automlorg/autopytorch:master

You can also verify that the image was downloaded via:

.. code:: bash

    docker images  # Verify that the image was downloaded

This image can be used to start an interactive session as follows:

.. code:: bash

    docker run -it automlorg/autopytorch:master

To start a Jupyter notebook, you could instead run e.g.:

.. code:: bash

    docker run -it -v ${PWD}:/opt/nb -p 8888:8888 automlorg/autopytorch:master /bin/bash -c "mkdir -p /opt/nb && jupyter notebook --notebook-dir=/opt/nb --ip='0.0.0.0' --port=8888 --no-browser --allow-root"

Alternatively, it is possible to use the development version of autoPyTorch by replacing all
occurences of ``master`` by ``development``.
