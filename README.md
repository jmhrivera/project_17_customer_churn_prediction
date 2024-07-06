# Important 
- Make sure to install the environment with the specific libraries:

- For conda
  - Execute the line:
  - conda env create -f environment.yml

- For pip
  - Execute the following lines:
  - pip install -r requirements.txt

# Repository structure

- src
  - main.py (This script will run in order the following scripts)
    1. data_loader
        - load_data.py
        - preprocess.py
    2. eda
        - eda_report.py
    3. feature_engineering
        - feature_engineering.py
        - ml_imputation.py
    4. models
        - build_model
        - hyper_parameters
- datasets
- results
  - eda_results
  - correlation_results
  - model_results
- sandbox (Practice Area)

Folders:

- Datasets: Will locate the 4 files to execute the project, when executing the code a new output folder will be generated with the merge of thes files and an preprocessed version.

- Results: This folder will be generated while executing the code: will host the results of the correlation, eda analysis and model results.

- src (source): This folder host the code of the project divided by stages (data_loade, eda, feature_engineering and models) and each one executed by a main file (main.py).

- Sandbox() : An area to practice your code

Files:
- project_instrucctions.md : The objective followed to execute the code.

# Contact
Email me at jm_hernandezr2@gmail.com

