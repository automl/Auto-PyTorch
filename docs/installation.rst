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

.. code:: bash

    conda create -n autopytorch python=3.8
    conda activate autopytorch
    conda install gxx_linux-64 gcc_linux-64 swig
    cat requirements.txt | xargs -n 1 -L 1 pip install
    python setup.py install

Docker Image
=========================
 TODO
