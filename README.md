# Important 
- Install the environment.yml to visualize de same results
- pip install -r environment.yml or conda install -r environment.yml

# Project 17 Final project TRIPLETEN

- Objective: Predict customer cancellation rate
- Model type: Classification
- Target feature: `'EndDate'` column is equal to `'No'`.
- Main metric: AUC-ROC.
- Additional metric: accuracy.
- Evaluation criteria:
- AUC-ROC < 0.75 — 0 SP
- 0.75 ≤ AUC-ROC < 0.81 — 4 SP
- 0.81 ≤ AUC-ROC < 0.85 — 4.5 SP
- 0.85 ≤ AUC-ROC < 0.87 — 5 SP
- 0.87 ≤ AUC-ROC < 0.88 — 5.5 SP
- AUC-ROC ≥ 0.88 — 6 SP

# Repository structure

- datasets
- results
  - eda_results
  - correlation_results
  - model_results
- src (All the code is here)
  - main.py (This script will run in order the following scripts)
  - data_loader
    - load_data.py
    - preprocess.py
  - eda
    - eda_report.py
  - feature_engineering
    - feature_engineering.py
    - ml_imputation.py
  - models
    - build_model
    - hyper_parameters
- sandbox (Practice Area)

# Contact
Email me at jm_hernandezr2@gmail.com

<!-- TODO Agregar .vscode -->
