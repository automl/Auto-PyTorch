:orphan:

.. _installation:

============
Installation
============

System requirements
===================

Auto-PyTorch has the following system requirements:

* Linux operating system (for example Ubuntu), Mac OS X `(get Linux here) <https://www.wikihow.com/Install-Linux>`_,
* Python (>=3.6) `(get Python here) <https://www.python.org/downloads/>`_.
* C++ compiler (with C++11 supports) `(get GCC here) <https://www.tutorialspoint.com/How-to-Install-Cplusplus-Compiler-on-Linux>`_ and
* SWIG (version 3.0.* is required; >=4.0.0 is not supported) `(get SWIG here) <http://www.swig.org/survey.html>`_.

Installing Auto-Pytorch
=======================

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


Docker Image
=========================
 TODO
