################################################################################
# GLOBALS                                                                      #
################################################################################

PROJECT_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
PROJECT_NAME = Support Vector Dynamics (svdynamics)
PYTHON_INTERPRETER = python3
VENV_NAME = venv_svd

# bash with pipefail so a failing script stops `make` even when piped to tee
SHELL := /bin/bash
.SHELLFLAGS := -o pipefail -c

ifeq (,$(shell which conda))
	HAS_CONDA=False
else
	HAS_CONDA=True
endif

### general usage notes
### 2>&1 | tee ==>pipe operation to save script output from terminal to .txt file
### every target with a "## description" line above it is listed by `make help`

.DEFAULT_GOAL := help

################################################################################
# COMMANDS                                                                     #
################################################################################

################################################################################
############## Setting up a Virtual Environment and Dependencies ###############
################################################################################

## Create the virtual environment (then run: source venv_svd/bin/activate)
.PHONY: venv
venv:
	$(PYTHON_INTERPRETER) -m venv $(VENV_NAME)
	@echo "Activate with: source $(VENV_NAME)/bin/activate"

## Create a conda environment instead of a venv
.PHONY: conda_env
conda_env:
ifeq (True,$(HAS_CONDA))
	conda create -y -n $(VENV_NAME) python=3.11
	@echo "Activate with: conda activate $(VENV_NAME)"
else
	@echo "conda not found; use 'make venv' instead"
endif

## Install dependencies (requirements.txt installs svdynamics in editable mode)
.PHONY: requirements_local
requirements_local:
	$(PYTHON_INTERPRETER) -m pip install -U pip setuptools wheel
	$(PYTHON_INTERPRETER) -m pip install -r requirements.txt

## Create the venv and print the install step
.PHONY: venv_dep_setup_local
venv_dep_setup_local: venv
	@echo "Next: source $(VENV_NAME)/bin/activate && make requirements_local"

################################################################################
################################## Unit Tests ##################################
################################################################################

## Run the unit tests
.PHONY: test
test:
	$(PYTHON_INTERPRETER) -m pytest unittests

## Run the unit tests with a coverage report
.PHONY: test_cov
test_cov:
	$(PYTHON_INTERPRETER) -m pytest unittests --cov=src/svdynamics --cov-report=term

################################################################################
############################## Python Example Scripts ##########################
################################################################################

# ─────────────────────────────────────────────────────────────────────────────
# Output directory (gitignored)
# ─────────────────────────────────────────────────────────────────────────────
DATA_OUTPUT := data_output

$(DATA_OUTPUT):
	mkdir -p $(DATA_OUTPUT)

# ─────────────────────────────────────────────────────────────────────────────
# Individual script targets
# ─────────────────────────────────────────────────────────────────────────────

## Composite-kernel classifier on make_classification
.PHONY: run_make_classification
run_make_classification: $(DATA_OUTPUT)
	$(PYTHON_INTERPRETER) py_example_scripts/make_classification.py \
	2>&1 | tee $(DATA_OUTPUT)/make_classification.txt

## Composite-kernel regressor on the diabetes dataset (model_tuner)
.PHONY: run_svd_regression
run_svd_regression: $(DATA_OUTPUT)
	$(PYTHON_INTERPRETER) py_example_scripts/svd_regression.py \
	2>&1 | tee $(DATA_OUTPUT)/svd_regression.txt

## Composite-kernel classifier with SMOTE on breast cancer (model_tuner)
.PHONY: run_svm_smote
run_svm_smote: $(DATA_OUTPUT)
	$(PYTHON_INTERPRETER) py_example_scripts/svm_smote.py \
	2>&1 | tee $(DATA_OUTPUT)/svm_smote.txt

# ─────────────────────────────────────────────────────────────────────────────
# Run all scripts sequentially
# ─────────────────────────────────────────────────────────────────────────────

## Run every example script; logs go to data_output/
.PHONY: run_all_py_example_scripts
run_all_py_example_scripts: \
	run_make_classification \
	run_svd_regression \
	run_svm_smote
	@echo "All scripts complete. Logs in $(DATA_OUTPUT)/"

################################################################################
############################## Build and Release ###############################
################################################################################

## Remove build artifacts
.PHONY: clean_build
clean_build:
	rm -rf dist/ build/ src/*.egg-info

## Build the sdist and wheel, then validate them with twine
.PHONY: build
build: clean_build
	$(PYTHON_INTERPRETER) -m build
	$(PYTHON_INTERPRETER) -m twine check dist/*

## Upload dist/ to PyPI (run `make build` first)
.PHONY: publish
publish:
	$(PYTHON_INTERPRETER) -m twine upload dist/*

## Remove example-script logs and Python caches
.PHONY: clean_dir
clean_dir:
	@echo "Cleaning directory..."
	rm -rf $(DATA_OUTPUT)/
	find . -type d -name "__pycache__" -prune -exec rm -rf {} +

################################################################################
#################################### Help ######################################
################################################################################

## Show this help
.PHONY: help
help:
	@echo "$(PROJECT_NAME)"
	@echo ""
	@awk '/^## /{desc=substr($$0,4); next} \
		/^[a-zA-Z0-9_]+:/{ if (desc) { split($$0,t,":"); \
		printf "  \033[36m%-28s\033[0m %s\n", t[1], desc }; desc="" } \
		!/^## /{ if ($$0 !~ /^\.PHONY/) desc="" }' $(MAKEFILE_LIST)