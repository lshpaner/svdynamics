################################################################################
# GLOBALS
################################################################################
PROJECT_DIR := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
PROJECT_NAME := Support Vector Dynamics (svdynamics)
PYTHON_INTERPRETER ?= python3
VENV_NAME := venv_svd
SCRIPT_DIR := py_example_scripts
DATA_OUTPUT := data_output

SHELL := /bin/bash
.SHELLFLAGS := -o pipefail -c
.DEFAULT_GOAL := help

HAS_CONDA := $(if $(shell command -v conda 2>/dev/null),True,False)

################################################################################
# ENVIRONMENT
################################################################################
## Create the virtual environment
.PHONY: venv
venv:
	$(PYTHON_INTERPRETER) -m venv $(VENV_NAME)
	@echo "Activate with: source $(VENV_NAME)/bin/activate"

## Create a conda environment
.PHONY: conda_env
conda_env:
ifeq ($(HAS_CONDA),True)
	conda create -y -n $(VENV_NAME) python=3.11
	@echo "Activate with: conda activate $(VENV_NAME)"
else
	@echo "conda not found; use 'make venv' instead"
endif

## Install development requirements
.PHONY: requirements_local
requirements_local:
	$(PYTHON_INTERPRETER) -m pip install -U pip setuptools wheel
	$(PYTHON_INTERPRETER) -m pip install -r requirements.txt

## Create virtual environment and show next step
.PHONY: venv_dep_setup_local
venv_dep_setup_local: venv
	@echo "Next: source $(VENV_NAME)/bin/activate && make requirements_local"

################################################################################
# TESTS
################################################################################
## Run unit tests
.PHONY: test
test:
	$(PYTHON_INTERPRETER) -m pytest unittests

## Run unit tests with coverage
.PHONY: test_cov
test_cov:
	$(PYTHON_INTERPRETER) -m pytest unittests --cov=src/svdynamics --cov-report=term

################################################################################
# EXAMPLE SCRIPTS
################################################################################
$(DATA_OUTPUT):
	mkdir -p $(DATA_OUTPUT)

# Macro ensures errors from Python propagate through tee.
define run_example
	$(PYTHON_INTERPRETER) $(SCRIPT_DIR)/$(1).py 2>&1 | tee $(DATA_OUTPUT)/$(1).txt
endef

## Run held-out classification example
.PHONY: run_make_classification
run_make_classification: | $(DATA_OUTPUT)
	$(call run_example,make_classification)

## Run held-out diabetes regression example
.PHONY: run_svd_regression
run_svd_regression: | $(DATA_OUTPUT)
	$(call run_example,svd_regression)

## Run SMOTE classification example
.PHONY: run_svm_smote
run_svm_smote: | $(DATA_OUTPUT)
	$(call run_example,svm_smote)

## Run repeated-CV adaptive kernel benchmark
.PHONY: run_adaptive_kernel_comparison
run_adaptive_kernel_comparison: | $(DATA_OUTPUT)
	$(call run_example,adaptive_kernel_comparison)

## Plot learned adaptive kernel weights
.PHONY: run_kernel_weight_visualization
run_kernel_weight_visualization: | $(DATA_OUTPUT)
	$(call run_example,kernel_weight_visualization)

## Plot SVM decision boundaries
.PHONY: run_decision_boundary_visualization
run_decision_boundary_visualization: | $(DATA_OUTPUT)
	$(call run_example,decision_boundary_visualization)

## Run sklearn Pipeline and GridSearchCV example
.PHONY: run_sklearn_pipeline_gridsearch
run_sklearn_pipeline_gridsearch: | $(DATA_OUTPUT)
	$(call run_example,sklearn_pipeline_gridsearch)

## Run custom kernel example
.PHONY: run_custom_kernel
run_custom_kernel: | $(DATA_OUTPUT)
	$(call run_example,custom_kernel)

## Run kernel PSD diagnostic example
.PHONY: run_kernel_psd_diagnostics
run_kernel_psd_diagnostics: | $(DATA_OUTPUT)
	$(call run_example,kernel_psd_diagnostics)

## Run the original three examples
.PHONY: run_core_py_example_scripts
run_core_py_example_scripts: run_make_classification run_svd_regression run_svm_smote
	@echo "Core scripts complete. Logs in $(DATA_OUTPUT)/"

## Run all nine examples
.PHONY: run_all_py_example_scripts
run_all_py_example_scripts: run_core_py_example_scripts run_adaptive_kernel_comparison run_kernel_weight_visualization run_decision_boundary_visualization run_sklearn_pipeline_gridsearch run_custom_kernel run_kernel_psd_diagnostics
	@echo "All scripts complete. Logs in $(DATA_OUTPUT)/"

################################################################################
# BUILD AND RELEASE
################################################################################
## Remove build artifacts
.PHONY: clean_build
clean_build:
	rm -rf dist/ build/ src/*.egg-info

## Build sdist and wheel, then validate
.PHONY: build
build: clean_build
	$(PYTHON_INTERPRETER) -m build
	$(PYTHON_INTERPRETER) -m twine check dist/*

## Upload dist/ to PyPI (run make build first)
.PHONY: publish
publish:
	$(PYTHON_INTERPRETER) -m twine upload dist/*

## Remove example logs and Python caches
.PHONY: clean_dir
clean_dir:
	rm -rf $(DATA_OUTPUT)/
	find . -type d -name '__pycache__' -prune -exec rm -rf {} +

################################################################################
# HELP
################################################################################
## Show this help
.PHONY: help
help:
	@echo "$(PROJECT_NAME)"
	@echo ""
	@awk '/^## /{desc=substr($$0,4); next} /^[a-zA-Z0-9_]+:/{ if (desc) {split($$0,t,":"); printf "  \033[36m%-34s\033[0m %s\n",t[1],desc}; desc=""} !/^## /{if ($$0 !~ /^\.PHONY/) desc=""}' $(MAKEFILE_LIST)
