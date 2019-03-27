Installation & Setup
====================

This section will explain how to "install" the scripts and set up and prepare 
your Python environment in order to get started with the sample generation.

.. warning::
   Since the ``PyCBC`` package is currently only available for **Python 2.7**
   (apparently due to some hard-to-migrate dependencies), all code in this 
   repository is written only for Python 2.7 --- sorry!

.. warning::
   This repository makes use of Python's built-in ``multiprocessing`` module,
   whose implementation depends on the operating system.
   Under some circumstances, this may lead to issues such as, for example, 
   a ``NotImplementedError`` when running on macOS (`see here for more
   information <https://docs.python.org/2/library/multiprocessing.html
   #multiprocessing.Queue.qsize>`_).





Installing this repository
--------------------------

Since for now, this repository is only a collection of scripts (and not a 
proper Python package), no installation is required! :)





Setting up your Python environment
----------------------------------

In order to be able to run the scripts in this repository, you need to make 
sure you have the necessary packages (dependencies) installed. 
All necessary packages are specified in the ``requirements.txt`` file at the
root of the repository.
If you are using ``pip`` to manage your Python installation, you can simply 
run the following command to install them all:

.. code-block:: bash

   pip install -r requirements.txt

We highly recommend you to install these packages in a new, separate 
*virtual environment*. 
If you don't know how to do that, maybe check out the `tutorial on virtualenv 
on the Hitchhiker's Guide to Python 
<https://docs.python-guide.org/dev/virtualenvs/#lower-level-virtualenv>`_.
Again, note that for now ``PyCBC`` only works under Python 2.7, so be sure
to choose the right Python executable when creating your virtual environment.

Also, be sure to *activate* your virtual environment before trying to run 
anything ;-)
