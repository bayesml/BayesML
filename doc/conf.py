# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.

import os
import sys
sys.path.insert(0, os.path.abspath('..'))


# -- Project information -----------------------------------------------------

project = 'BayesML'
copyright = '2022-2026, BayesML Developers'
author = 'BayesML Developers'

# The full version, including alpha/beta/rc tags
release = '0.5.0'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
	'sphinx.ext.mathjax',
	'myst_parser',
	'sphinx.ext.autodoc',
	'numpydoc',
	'sphinx.ext.autosummary',
	'sphinx.ext.intersphinx',
	#'sphinx.ext.napoleon',
	'nbsphinx',
    'sphinx_favicon',
    'sphinxext.opengraph',
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store','devdoc']


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_book_theme'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

html_css_files = ["custom.css"]

html_title = 'BayesML'

myst_enable_extensions = ["dollarmath", "amsmath","html_image"]

# get the language to build (Makefile is modified to create SPHINX_LANGUAGE variable)
language = os.environ.get('SPHINX_LANGUAGE', 'en')
if language == 'ja':
  ogp_site_url = "https://bayesml.github.io/BayesML/ja/"
  announcement_text = "メタツリーモデルに対するアルゴリズムがAISTATS 2025に採択！詳細は<a href='https://bayesml.github.io/BayesML/ja/examples/metatree_prediction_interval.html'>こちら</a>！"
else:
  ogp_site_url = "https://bayesml.github.io/BayesML/en/"
  announcement_text = "Our algorithm for the meta-tree model is accepted at AISTATS 2025! Click <a href='https://bayesml.github.io/BayesML/en/examples/metatree_prediction_interval.html'>here</a>!"

ogp_image = "./_static/BayesML_logo_ogp.png"
ogp_use_first_image = True

html_theme_options = {
  "repository_url": "https://github.com/bayesml/BayesML/",
  "use_repository_button": True,
  "announcement": announcement_text,
  "analytics": {
      "google_analytics_id": "G-59F6KL8C5D",
  },
  "logo": {
      "image_light": "logos/BayesML_logo.svg",
      "image_dark": "logos/BayesML_logo_reverse.svg",
  },
}

napoleon_use_rtype = False

autodoc_default_options = {
    'member-order': 'bysource',
}

#numpydoc_show_class_members = False

autosummary_generate = True

#numpydoc_xref_param_type = True
intersphinx_mapping = {'python': ('https://docs.python.org/3', None),
                       'numpy': ('https://numpy.org/doc/stable/', None),
                       'scipy': ('https://docs.scipy.org/doc/scipy/reference/', None),
                       'graphviz': ('https://graphviz.readthedocs.io/en/stable/', None),
                       'sklarn': ('https://scikit-learn.org/stable/', None),
}

favicons = [
    {
        "rel": "icon",
        "href": "favicon.ico",
        # NOTE: favicon.ico actually contains 16x16/32x32/48x48 images,
        # but we deliberately declare only "32x32" here.
        # If we list all sizes accurately (e.g. "16x16 32x32 48x48"),
        # some browsers prioritize this exact-size match over the
        # SVG's "sizes=any", causing the dark-mode SVG favicon below
        # to be ignored. Declaring a single specific size keeps this
        # icon's priority lower than the SVG, so browsers that support
        # SVG favicons pick the dark-mode-aware SVG, while older
        # browsers fall back to this ICO.
        "sizes": "32x32",
    },
    {
        "rel": "icon",
        "href": "favicon.svg",
        "sizes": "any",
    },
]

locale_dirs = ['locale/']   # path is example but recommended.
gettext_compact = False     # optional.

html_sidebars = {
    '**': ['navbar-logo.html',
           'icon-links.html',
           'language-switcher.html',
           'search-button-field.html',
           'sbt-sidebar-nav.html']
}

autodoc_preserve_defaults = True
