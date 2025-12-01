Installation
============

This guide will help you install IceModels and its dependencies.

Requirements
------------

IceModels requires the following Python packages:

* numpy
* astropy
* scipy
* matplotlib
* requests (for downloading data)

Installing IceModels
--------------------

You can install IceModels using pip:

.. code-block:: bash

    pip install icemodels

Or install from source:

.. code-block:: bash

    git clone https://github.com/keflavich/icemodels.git
    cd icemodels
    pip install -e .

Configuration
-------------

After installation, you may want to configure the data directory where IceModels will store downloaded data:

.. code-block:: python

    import icemodels
    icemodels.set_data_dir('path/to/data')
