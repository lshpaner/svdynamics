"""Compatibility shim for old pip (< 21.3) editable installs.

Modern pip builds svdynamics from pyproject.toml (hatchling) and ignores this
file. Old pip, such as the pip 19 bundled with some Python 3.8 venvs, needs a
setup.py to run `pip install -e .`. Package metadata lives in pyproject.toml;
keep the version and dependencies here in sync with it.
"""
import re
from pathlib import Path

from setuptools import find_packages, setup

version = re.search(
    r'^__version__ = "([^"]+)"',
    Path("src/svdynamics/__init__.py").read_text(),
    re.M,
).group(1)

setup(
    name="svdynamics",
    version=version,
    package_dir={"": "src"},
    packages=find_packages("src"),
    python_requires=">=3.8",
    install_requires=["numpy>=1.21", "scikit-learn>=1.0"],
)