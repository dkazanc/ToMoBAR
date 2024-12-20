#!/usr/bin/env python
import os
import sys
from datetime import date
from unittest import mock


# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.

sys.path.insert(0, os.path.abspath("../.."))

# -- Mock imports -------------------------------------------------------------


# Mock imports instead of full environment in readthedocs
MOCK_MODULES = [
    "cupy",
    "astra",
    "scipy",
    "scipy.fftpack",
]

# MOCK_MODULES = [
#     "cupy",
#     "skimage",
#     "scipy",
#     "scipy.fft",
#     "scipy.ndimage",
# ]

for mod_name in MOCK_MODULES:
    sys.modules[mod_name] = mock.Mock()

autodoc_mock_imports = [
    "astra",
    "astra.experimental",
    "cupy",
    "scipy",
    "scipy.fftpack",
    "bm3d",
    "skimage",
    "tomobar.astra_wrappers.astra_base",
    "tomobar.astra_wrappers.astra_tools2d",
    "tomobar.astra_wrappers.astra_tools3d",
]


class CustomMock(mock.Mock):
    def __repr__(self):
        return "<cp.ndarray>"


sys.modules["cupy"] = CustomMock()
sys.modules["numpy"] = CustomMock()


# ------------------------------------------------------------------------------

project = "ToMoBAR"
author = "Daniil Kazantsev"
copyright = f"{date.today().year}, Diamond Light Source and Manchester University, UK"

# Specify a base language to help assistive technology
language = "en"

# Save the commit hash, this is displayed in the page title
release = os.popen('git log -1 --format="%H"').read().strip()

# Set version as the latest tag in the current branch
version = os.popen("git describe --tags --abbrev=0").read().strip()

# Add any Sphinx extension module names here, as strings. They can be extensions
# coming with Sphinx (named 'sphinx.ext.*') or your custom ones.
extensions = [
    # Generates api files
    "sphinx.ext.autodoc",
    # Generates a short summary from docstring
    "sphinx.ext.autosummary",
    # Allows parsing of google style docstrings
    "sphinx.ext.napoleon",
    # Add links to highlighted source code
    "sphinx.ext.viewcode",
    # Allows a grid layout and dropdown boxes
    "sphinx_design",
    # copy to clipboard button
    "sphinx_copybutton",
    # use jupyter notebooks
    "nbsphinx",
    #'IPython.sphinxext.ipython_console_highlighting',
    "sphinx.ext.githubpages",
    # Generate .nojekyll file for git pages build
]
autosummary_generate = True
numfig = True
template_patterns = ["_templates"]

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = "sphinx"

# -- Options for HTML output --------------------------------------------------

html_theme = "sphinx_book_theme"
html_logo = "_static/tomobar_logo_light.png"
html_title = "Tomobar Documentation page"
html_copy_source = True
html_favicon = "_static/tomobar_logo_light.png"
html_last_updated_fmt = ""
html_static_path = ["_static"]
html_use_smartypants = True

html_theme_options = {
    "logo": {
        "image_dark": "_static/tomobar_logo_dark.png",
    },
    "show_toc_level": 1,
    "navbar_align": "left",  # [left, content, right] For testing that the navbar items align properly
}

html_context = {
    "github_user": "tomobar",
    "github_repo": "https://github.com/dkazanc/ToMoBAR",
    "github_version": "main",
    "doc_path": "docs",
}


def setup(app):
    app.add_css_file("css/general.css")
