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

You can install IceModels using pip:

.. code-block:: bash

    pip install icemodels

For development installation, clone the repository and install in development mode:

.. code-block:: bash

    git clone https://github.com/your-username/icemodels.git
    cd icemodels
    pip install -e .

Configuration
------------

By default, IceModels will cache downloaded data in the package's data directory. You can modify this behavior by setting the environment variable ``ICEMODELS_CACHE_DIR``.