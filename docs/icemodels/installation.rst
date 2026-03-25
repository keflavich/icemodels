Installation
============

This guide will help you install icemodels and its dependencies.

Requirements
------------

icemodels requires the following Python packages:

* numpy
* astropy
* scipy
* matplotlib
* requests (for downloading data)
* synphot
* stsynphot

Installing icemodels
--------------------

You can install icemodels using pip:

.. code-block:: bash

    pip install icemodels

Or install from source:

.. code-block:: bash

    git clone https://github.com/keflavich/icemodels.git
    cd icemodels
    pip install -e .

Configuration
-------------

After installation, you may want to configure the data directory where ``icemodels`` will store downloaded data:

.. code-block:: python

    import icemodels

Synphot data on documentation builders
--------------------------------------

The stellar SED routines use ``synphot``/``stsynphot`` when a valid
``PYSYN_CDBS`` grid is available. For lightweight environments (e.g.,
ReadTheDocs), a small subset of data are downloaded.

This keeps documentation plotting functional without downloading the full
multi-GB CDBS archive.
