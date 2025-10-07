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
release = '0.3'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = []

templates_path = ['_templates']
extensions = [
    'sphinx.ext.githubpages',
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

html_extra_path = ['CNAME']

#Google Analytics


def setup(app):
    app.add_js_file(
        'https://www.googletagmanager.com/gtag/js?id=G-YSF8MXKHBM',
        **{'async': 'async'}
    )
    app.add_js_file(None, body=r"""
window.dataLayer = window.dataLayer || [];
function gtag(){dataLayer.push(arguments);}
gtag('js', new Date());
gtag('config', 'G-YSF8MXKHBM');
""")
