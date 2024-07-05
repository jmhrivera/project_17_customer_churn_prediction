# Important 
- Install the environment.yml to create an env with the specific libraries
- pip install -r requirements.txt or conda install -r requirements.txt

<!-- TODO Ni  pip ni conda funcionan con environment.yml. Lo cambié por un requirements.txt -->

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


<!-- TODO La carpeta results no está -->
<!-- TODO Hay varios pycache. Te recomiendo ignorar todos esos. Los comiteaste por algún motivo especial? Si no, es mejor que los borres -->
<!-- TODO Los archivos que hay que leer es mejor dejarlos como markdown, no txt. -->
<!-- TODO En general tu README es un buen comienzo ;) Además de lo que ya escribí, falta que expliques cómo ejecutar tu proyecto. Cómo alguien podría usarlo? --> -->