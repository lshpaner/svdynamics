
from pathlib import Path

from docutils import nodes
from docutils.parsers.rst import roles

# -- Project information -----------------------------------------------------

project = "svdynamics"
copyright = "2026, Leon Shpaner"
author = "Leon Shpaner"
release = "0.0.1a1"
version = release

# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx_copybutton",
    "sphinx.ext.todo",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx_multiversion",
]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "sklearn": ("https://scikit-learn.org/stable/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
}

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- Sphinx multiversion -----------------------------------------------------

smv_branch_whitelist = r"^(main|docs)$"
smv_tag_whitelist = r"^v0\.0\.1a[0-9]+$"
smv_remote_whitelist = r"^origin$"

# -- HTML output -------------------------------------------------------------

html_theme = "pydata_sphinx_theme"

html_static_path = ["_static"]

html_show_sourcelink = False

html_context = {
    "display_github": False,
    "github_user": "lshpaner",
    "github_repo": "svdynamics",
    "github_version": "docs",
}

html_css_files = ["custom.css"]

# Register JavaScript only if the file exists.
if (Path(__file__).parent / "_static" / "custom.js").is_file():
    html_js_files = ["custom.js"]

# -- Custom reStructuredText roles -------------------------------------------

def bold_literal_role(
    name,
    rawtext,
    text,
    lineno,
    inliner,
    options=None,
    content=None,
):
    """Render inline literals using bold formatting."""
    if options is None:
        options = {}
    if content is None:
        content = []

    node = nodes.strong()
    literal_node = nodes.literal(
        text,
        text,
        classes=["bold-literal"],
    )
    node += literal_node

    return [node], []


roles.register_canonical_role(
    "bold-literal",
    bold_literal_role,
)
