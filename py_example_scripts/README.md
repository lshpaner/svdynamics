# svdynamics examples

Run from a Python environment with `svdynamics`, `scikit-learn`, and `numpy` installed.

```bash
make run_all_py_example_scripts   # all nine; logs and figures in data_output/
python py_example_scripts/make_classification.py
```

Visualization examples also require `matplotlib` and save their figures to
`data_output/`. The SMOTE example requires
`imbalanced-learn`. All performance examples use held-out splits or
cross-validation. Preprocessing and adaptive weight fitting occur within
training partitions. The examples intentionally do not guarantee that adaptive
kernels outperform single kernels.

The three original example filenames are retained, but their implementations
have been modernized. The SMOTE and regression scripts are standalone sklearn
examples.