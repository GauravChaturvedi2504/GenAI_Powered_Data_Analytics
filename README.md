TATA IQ – Delinquency Prediction Using Machine Learning
#Project Overview
This project focuses on predicting customer loan delinquency using Machine Learning techniques.
The objective is to analyze customer financial and credit-related information, identify patterns associated with delinquency, and build predictive models that can help identify customers who may be at risk of missing loan payments.
This project was developed as part of a TATA GenAI simulation project.
#Problem Statement
Financial institutions need to identify customers who are more likely to miss loan payments.
By analyzing customer information such as income, credit score, credit utilization, missed payments, and loan balance, Machine Learning models can help identify potential delinquency risks.
#Project Objectives
Analyze customer financial data.
Perform data cleaning and preprocessing.
Handle missing values.
Conduct Exploratory Data Analysis (EDA).
Build and compare Machine Learning models.
Evaluate model performance using classification metrics.
Understand the factors associated with customer delinquency.
#Dataset Information
Total Records: 500
Total Features: 19
Target Variable: Delinquent_Account
Target Classes:
Non-Delinquent (0): 420
Delinquent (1): 80
The dataset includes customer-related attributes such as:
Age
Income
Credit Score
Credit Utilization
Missed Payments
Loan Balance
Debt-to-Income Ratio
Employment Status
Account Tenure
Credit Card Type
Location
Monthly payment-related information
#Data Preprocessing
The following preprocessing steps were performed:
Checked for missing values.
Handled missing numerical values using median imputation.
Checked and handled duplicate records.
Prepared the target variable for classification.
Analyzed class distribution.
#Exploratory Data Analysis (EDA)
EDA was performed to understand customer financial patterns and relationships between different features and delinquency.
The EDA report is available in this repository.
#Machine Learning Models
The following classification algorithms were explored:
Logistic Regression
Random Forest Classifier
XGBoost Classifier
The models were evaluated and compared to understand their performance in identifying delinquent customers.
#Model Evaluation
The following evaluation metrics were considered:
Accuracy
Precision
Recall
Confusion Matrix
Classification Report
Special attention was given to Recall for the delinquent class because identifying customers at risk of missing payments is an important objective.
Refer to model result.txt for the recorded model evaluation results.
#Technologies Used
Python
Pandas
NumPy
Matplotlib
Scikit-learn
XGBoost
Jupyter Notebook / VS Code
Git & GitHub
#Project Structure
TATA_IQ_PROJECT/
│
├── main.py
├── Delinquency_prediction_dataset.xlsx
├── EDA_SummaryReport.pdf
├── lrm.pkl
├── model result.txt
└── README.md
#How to Run the Project
Clone the repository:
git clone https://github.com/GauravChaturvedi2504/GenAI_Powered_Data_Analytics.git
Navigate to the project directory:
cd GenAI_Powered_Data_Analytics
Install the required libraries:
pip install pandas numpy matplotlib scikit-learn xgboost openpyxl
Run the Python file:
python main.py
#Key Learnings
Data cleaning and preprocessing.
Handling missing values using median imputation.
Exploratory Data Analysis.
Working with imbalanced classification datasets.
Training and evaluating classification models.
Comparing Logistic Regression, Random Forest, and XGBoost.
Understanding the importance of Recall in risk prediction problems.
#Disclaimer
This repository represents an educational project developed as part of a simulation. It is not an official production system or a financial decision-making tool.
