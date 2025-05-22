Installation
============

Requirements
-----------

IceModels requires the following Python packages:

* numpy
* astropy
* beautifulsoup4
* requests
* tqdm
* pylatexenc
* molmass

Installing IceModels
------------------

You can install IceModels directly from GitHub using pip:

.. code-block:: bash

    git clone https://github.com/keflavich/icemodels.git
    cd icemodels
    pip install -e .

This will install the package in development mode, allowing you to modify the code and immediately see the effects without reinstalling.

Configuration
------------

By default, IceModels will cache downloaded data in the package's data directory. You can modify this behavior by setting the environment variable ``ICEMODELS_CACHE_DIR``.