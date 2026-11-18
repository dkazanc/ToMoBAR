#!/usr/bin/env python
import sys
from datetime import date
from pathlib import Path
import tomllib

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

# -- Optional GPU dependencies ------------------------------------------------
# Keep NumPy and ToMoBAR real so geometry, annotations and API links resolve.
# Sphinx provides temporary mocks only while importing optional dependencies.
autodoc_mock_imports = ["astra", "cupy", "cupyx", "bm3d"]

project = "ToMoBAR"
author = "Daniil Kazantsev"
copyright = f"{date.today().year}, Diamond Light Source and Manchester University, UK"

# Specify a base language to help assistive technology
language = "en"

# Match the package version, including source archives without Git metadata.
release = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["version"]
version = release

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
sd_card_legacy_separators = False
template_patterns = ["_templates"]
# Publish the outputs saved by notebook authors without executing any cells.
nbsphinx_execute = "never"
# Ignore generated files left over from the previous gallery workflow.
exclude_patterns = ["auto_examples/**"]

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = "sphinx"

# -- Options for HTML output --------------------------------------------------

html_theme = "sphinx_book_theme"
html_logo = "_static/tomobar_logo_v3_light.png"
html_title = "ToMoBAR documentation"
html_copy_source = True
html_favicon = "_static/tomobar_icon_v3.png"
html_last_updated_fmt = ""
html_static_path = ["_static"]
html_use_smartypants = True

html_theme_options = {
    "logo": {
        "image_dark": "_static/tomobar_logo_v3_dark.png",
    },
    "show_toc_level": 1,
    "navbar_persistent": [],  # Keep the search field in the sidebar only.
    "navbar_align": "left",  # [left, content, right] For testing that the navbar items align properly
}

html_context = {
    "github_user": "dkazanc",
    "github_repo": "ToMoBAR",
    "github_version": "master",
    "doc_path": "docs",
}


def setup(app):
    app.add_css_file("css/general.css")
