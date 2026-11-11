################################################################################
# GLOBALS
################################################################################
PROJECT_DIR := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
PROJECT_NAME := Support Vector Dynamics (svdynamics)
PYTHON_INTERPRETER ?= python3
VENV_NAME := venv_svd
SCRIPT_DIR := py_example_scripts
DATA_OUTPUT := data_output
RESEARCH_DIR := lucy_svm_research

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

## Plot repeated-CV accuracy box plots
.PHONY: run_cv_comparison_visualization
run_cv_comparison_visualization: | $(DATA_OUTPUT)
	$(call run_example,cv_comparison_visualization)

## Run the original three examples
.PHONY: run_core_py_example_scripts
run_core_py_example_scripts: run_make_classification run_svd_regression run_svm_smote
	@echo "Core scripts complete. Logs in $(DATA_OUTPUT)/"

## Run all ten examples
.PHONY: run_all_py_example_scripts
run_all_py_example_scripts: run_core_py_example_scripts run_adaptive_kernel_comparison run_kernel_weight_visualization run_decision_boundary_visualization run_sklearn_pipeline_gridsearch run_custom_kernel run_kernel_psd_diagnostics run_cv_comparison_visualization
	@echo "All scripts complete. Logs in $(DATA_OUTPUT)/"

################################################################################
# RESEARCH: LUCY-SVM
################################################################################

## Run LUSI baseline experiment
.PHONY: research_lusi_baseline
research_lusi_baseline:
	$(PYTHON_INTERPRETER) $(RESEARCH_DIR)/lusi_baseline.py

## Run repeated stratified CV for LUSI vs KRR
.PHONY: research_lusi_cv
research_lusi_cv:
	cd $(RESEARCH_DIR) && $(PYTHON_INTERPRETER) lusi_cross_validation.py

## Download, validate, and sample the MAGIC dataset
.PHONY: research_magic_data
research_magic_data:
	$(PYTHON_INTERPRETER) $(RESEARCH_DIR)/load_magic.py

## Validate MAGIC kernel module import
.PHONY: research_magic_kernels
research_magic_kernels:
	$(PYTHON_INTERPRETER) -c "import sys; sys.path.insert(0, '$(RESEARCH_DIR)'); import magic_kernels; print('MAGIC kernel module imported successfully')"

## Run conventional two-feature MAGIC KRR baseline
.PHONY: research_magic_krr
research_magic_krr:
	$(PYTHON_INTERPRETER) $(RESEARCH_DIR)/lusi_magic_benchmark.py

## Compare MAGIC KRR against invariant-weighted LUSI
.PHONY: research_magic_lusi
research_magic_lusi:
	$(PYTHON_INTERPRETER) $(RESEARCH_DIR)/lusi_magic_benchmark.py

## Evaluate MAGIC LUSI sensitivity to invariant strength tau
.PHONY: research_magic_tau
research_magic_tau:
	$(PYTHON_INTERPRETER) $(RESEARCH_DIR)/lusi_magic_tau_sensitivity.py

## Run all LUCY-SVM research experiments
.PHONY: research_all
research_all: research_lusi_baseline research_lusi_cv research_magic_data research_magic_lusi research_magic_tau
	@echo "LUCY-SVM research experiments complete."

################################################################################
# NOTEBOOK CLEANUP AND GIT INTEGRATION
################################################################################

## Install nbstripout in the active Python environment
.PHONY: install_nbstripout
install_nbstripout:
	$(PYTHON_INTERPRETER) -m pip install nbstripout

## Configure nbstripout Git filter for all notebooks
.PHONY: setup_nbstripout
setup_nbstripout:
	nbstripout --install --attributes .gitattributes
	@echo "nbstripout Git filter installed."

## Strip outputs and execution counts from all notebooks
.PHONY: strip_notebooks
strip_notebooks:
	@find . -type f -name '*.ipynb' \
		-not -path './.git/*' \
		-not -path '*/.ipynb_checkpoints/*' \
		-not -path '*/site-packages/*' \
		-exec nbstripout {} +
	@echo "Notebook outputs stripped."

## Install and configure nbstripout
.PHONY: notebook_setup
notebook_setup: install_nbstripout setup_nbstripout
	@echo "Notebook Git integration configured."

################################################################################
# DOCUMENTATION (run on the docs branch, which holds source/ and docs/)
################################################################################

DOCS_SOURCE := source
DOCS_STATIC := $(DOCS_SOURCE)/_static
DOCS_OUTPUT := docs
DOCS_FIG_DPI := 150
DOCS_FIGURES := decision_boundary_visualization kernel_weight_visualization cv_comparison_visualization

## Regenerate documentation figures into source/_static
.PHONY: docs_figures
docs_figures:
	@test -d $(DOCS_SOURCE) || { echo "No $(DOCS_SOURCE)/ folder: run this on the docs branch"; exit 1; }
	@for script in $(DOCS_FIGURES); do \
		$(PYTHON_INTERPRETER) $(SCRIPT_DIR)/$$script.py --out-dir $(DOCS_STATIC) --dpi $(DOCS_FIG_DPI) || exit 1; \
	done

## Rebuild figures and the full HTML site into docs/ from scratch
.PHONY: docs_html
docs_html: docs_figures
	rm -rf $(DOCS_SOURCE)/_build
	$(PYTHON_INTERPRETER) -m sphinx -q -b html $(DOCS_SOURCE) $(DOCS_SOURCE)/_build/html
	mkdir -p $(DOCS_OUTPUT)
	find $(DOCS_OUTPUT) -mindepth 1 ! -name .nojekyll -delete
	cp -r $(DOCS_SOURCE)/_build/html/. $(DOCS_OUTPUT)/
	rm -rf $(DOCS_SOURCE)/_build
	@echo "Figures in the built site:"
	@ls $(DOCS_OUTPUT)/_static/*.png

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
