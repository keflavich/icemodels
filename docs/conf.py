# -*- coding: utf-8 -*-
# Licensed under a 3-clause BSD style license - see LICENSE.rst
#
# Astropy documentation build configuration file.
#
# This file is execfile()d with the current directory set to its containing dir.
#
# Note that not all possible configuration values are present in this file.
#
# All configuration values have a default. Some values are defined in
# the global Astropy configuration which is loaded here before anything else.
# See astropy.sphinx.conf for which values are set there.

import os
import sys
import datetime
from importlib import import_module

# Add the project root directory to the Python path
sys.path.insert(0, os.path.abspath('..'))

# -- Project information -----------------------------------------------------
project = 'icemodels'
copyright = '2024, Adam Ginsburg'
author = 'Adam Ginsburg'

# The full version, including alpha/beta/rc tags
import icemodels
release = icemodels.__version__
version = release.split('-', 1)[0]

# -- General configuration ---------------------------------------------------
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.mathjax',
    'sphinx.ext.intersphinx',
    'sphinx_gallery.gen_gallery',
    'matplotlib.sphinxext.plot_directive',
    'myst_parser',
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# Ignore specific warnings
nitpicky = True

# Treat warnings as errors
nitpick_ignore = []

# -- Options for HTML output -------------------------------------------------
html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

# Sphinx-Gallery configuration
sphinx_gallery_conf = {
    'examples_dirs': 'examples',   # path to your example scripts
    'gallery_dirs': 'auto_examples',  # path to where to save gallery generated output
    'filename_pattern': '/plot_',  # pattern to match example files
    'plot_gallery': True,  # generate the gallery
    'download_all_examples': True,  # download all examples to build
    'line_numbers': True,  # add line numbers to examples
    'remove_config_comments': True,  # remove config comments from code
}

# Intersphinx configuration
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'astropy': ('https://docs.astropy.org/en/stable/', None),
}

# Napoleon settings
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_type_aliases = None

# -- General configuration ----------------------------------------------------

# By default, highlight as Python 3.
highlight_language = 'python3'

# If your documentation needs a minimal Sphinx version, state it here.
#needs_sphinx = '1.2'

# To perform a Sphinx version check that needs to be more specific than
# major.minor, call `check_sphinx_version("X.Y.Z")` here.
# check_sphinx_version("1.2.1")

# This is added to the end of RST files - a good place to put substitutions to
# be used globally.
rst_epilog = """
"""

# -- Project information ------------------------------------------------------

# The version info for the project you're documenting, acts as replacement for
# |version| and |release|, also used in various other places throughout the
# built documents.

# -- Options for HTML output --------------------------------------------------

# A NOTE ON HTML THEMES
# The global astropy configuration uses a custom theme, 'bootstrap-astropy',
# which is installed along with astropy. A different theme can be used or
# the options for this theme can be modified by overriding some of the
# variables set in the global configuration. The variables set in the
# global configuration are listed below, commented out.


# Add any paths that contain custom themes here, relative to this directory.
# To use a different custom theme, add the directory containing the theme.
#html_theme_path = []

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes. To override the custom theme, set this to the
# name of a builtin theme or the name of a custom theme in html_theme_path.
#html_theme = None

# Custom sidebar templates, maps document names to template names.
#html_sidebars = {}

# The name of an image file (relative to this directory) to place at the top
# of the sidebar.
#html_logo = ''

# The name of an image file (within the static path) to use as favicon of the
# docs.  This file should be a Windows icon file (.ico) being 16x16 or 32x32
# pixels large.
#html_favicon = ''

# If not '', a 'Last updated on:' timestamp is inserted at every page bottom,
# using the given strftime format.
#html_last_updated_fmt = ''

# The name for this set of Sphinx documents.  If None, it defaults to
# "<project> v<release> documentation".
html_title = '{0} v{1}'.format(project, release)

# Output file base name for HTML help builder.
htmlhelp_basename = project + 'doc'

# Prefixes that are ignored for sorting the Python module index
modindex_common_prefix = ["icemodels."]


# -- Options for LaTeX output -------------------------------------------------

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title, author, documentclass [howto/manual]).
latex_documents = [('index', project + '.tex', project + u' Documentation',
                    author, 'manual')]


# -- Options for manual page output -------------------------------------------

# One entry per manual page. List of tuples
# (source start file, name, description, authors, manual section).
man_pages = [('index', project.lower(), project + u' Documentation',
              [author], 1)]


# -- Options for linkcheck output -------------------------------------------
linkcheck_retry = 5
linkcheck_ignore = [
    r'https://github\.com/keflavich/icemodels/(?:issues|pull)/\d+',
]
linkcheck_timeout = 180
linkcheck_anchors = False

# -- Turn on nitpicky mode for sphinx (to warn about references not found) ----
#
# nitpicky = True
# nitpick_ignore = []
#
# Some warnings are impossible to suppress, and you can list specific references
# that should be ignored in a nitpick-exceptions file which should be inside
# the docs/ directory. The format of the file should be:
#
# <type> <class>
#
# for example:
#
# py:class astropy.io.votable.tree.Element
# py:class astropy.io.votable.tree.SimpleElement
# py:class astropy.io.votable.tree.SimpleElementWithContent
#
# Uncomment the following lines to enable the exceptions:
