## Main Assignment Conditions

Telecommunications operator Interconnect would like to forecast its customer churn rate. If a user is found to be planning to leave, they will be offered promotional codes and special plan options. Interconnect's marketing team has collected some personal data from its customers, including information about their plans and contracts.

Interconnect Services
Interconnect mainly provides two types of services:

- Fixed-line telephone communication. The phone can be connected to multiple lines simultaneously.
- Internet. The network can be set up via a telephone line (DSL, digital subscriber line) or via fiber optic cable.

Some other services offered by the company include:

- Internet security: antivirus software (DeviceProtection) and a malicious website blocker (OnlineSecurity).
- Technical support line (TechSupport).
- Cloud file storage and data backup (OnlineBackup).
- TV streaming (StreamingTV) and movie directory (StreamingMovies).
- Customers can choose between monthly payments or signing a 1- or 2-year contract. They can use various payment methods and receive an electronic bill after a transaction.

Data Description
The data consists of files obtained from different sources:

- contract.csv : contract information;
- personal.csv : customer personal data;
- internet.csv : information about Internet services;
- phone.csv : information about phone services.

In each file, the customerID column contains a unique code assigned to each customer. The contract information is valid starting from February 1, 2020.


# Work Plan

## Clarifying Questions
These are the clarifying questions I would ask the company:

- Can you provide a data dictionary for each of the shared files?
- Is this the complete dataset or is there a separate test set available?
- How do you plan to utilize the model results in Interconnect's daily operations?
- Are there known usage patterns of services (phone, internet, etc.) that correlate with customer retention or churn?

# Workflow Proposal

## Stage 1: Define Project Objective
- Analyze client needs and objectives, assess feasibility, and generate a plan with defined timelines and responsibilities.
- Translate the objective into an analytical problem.

## Stage 2: Data Collection
- Acquire and extract project-related data from the company and load it into our analysis.

## Stage 3: Preprocessing
- Data cleaning (remove duplicates, handle null values, ensure data consistency).
- Data transformation (Normalization, scaling, and/or creation of new variables).
- Data splitting (Split data into training, validation, and test sets).

## Stage 4: Exploration and Analysis
- Exploratory Data Analysis (EDA) to understand data distribution and relationships.
- Feature engineering.

## Stage 5: Modeling
- Apply various ML models and optimize for the best AUC metric (Classification) using cross-validation and hyperparameter tuning.

## Stage 6: Model Deployment
- Deploy the winning model to obtain prediction results.

## Stage 7: Results Presentation
- Summarize project outcomes and present the learning curve derived from the project.
