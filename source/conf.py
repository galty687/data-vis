# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

from enum import auto


project = 'data visulization'
copyright = '2025, Zhijun Gao'
author = 'Zhijun Gao'
release = '0.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = []

templates_path = ['_templates']
extensions = [
    'sphinx.ext.githubpages',
    'sphinxcontrib.googleanalytics',
    'sphinx.ext.todo',
    'sphinx_comments',
    'sphinx_copybutton',
    'myst_parser',
]

language = 'zh'

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'shibuya'
html_static_path = ['_static']
html_css_files = [
    'css/custom.css'
]

# html_theme_options = {
#   "accent_color": "grass",
#   "dark_code": auto,
# }



#Google Analytics

googleanalytics_id = 'G-GESLLLJC6M'