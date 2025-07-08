# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html
import os
import sys
#sys.path.insert(0, os.path.abspath('/Users/walder2/_fake_pyproject'))
sys.path.insert(0, os.path.abspath('/Users/walder2/pysp_1.0.0/pysparkplug'))
# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'pysparkplug'
copyright = '2024, adam'
author = 'adam'
release = '0.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',       # auto-generate documentation
    'sphinx.ext.napoleon',      # support for Google and NumPy docstrings
    'sphinx_autodoc_typehints', # auto-document type hints
    'sphinx.ext.mathjax',
]

templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'alabaster'
html_static_path = ['_static']
