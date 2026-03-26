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

Testing docs like ReadTheDocs
-----------------------------

You can run a local ReadTheDocs-style documentation build with tox:

.. code-block:: bash

    tox -e build_docs_rtd

This runs the same strict Sphinx flags used in RTD-style builds
(``-T -W --keep-going``) and writes output under ``docs/_build/html``.
