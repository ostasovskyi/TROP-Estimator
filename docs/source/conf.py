import os
import sys


sys.path.insert(0, os.path.abspath("../.."))

project = "trop"
copyright = "2026, Susan Athey, Guido Imbens, Zhaonan Qu, Davide Viviano"
author = "Susan Athey, Guido Imbens, Zhaonan Qu, Davide Viviano"
release = "0.1.5"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx_autodoc_typehints",
]

templates_path = ["_templates"]
exclude_patterns = []

html_theme = "furo"

autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
}
napoleon_numpy_docstring = True
napoleon_google_docstring = False
