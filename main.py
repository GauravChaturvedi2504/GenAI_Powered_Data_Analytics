import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier

from sklearn.inspection import permutation_importance

from sklearn.model_selection import cross_val_score

from sklearn.impute import SimpleImputer

from sklearn.pipeline import Pipeline

import joblib

#Load Dataset 
df = pd.read_excel("Delinquency_prediction_dataset.xlsx")

# Display first 5 rows
print(df.head())

# Dataset information
print(df.info())

# Dataset shape 
print("Dataset Shape:", df.shape)

#Check missing values 
print("\nMissing Values:")
print(df.isnull().sum())

#Check duplicate records
print("\nDuplicate Rows:")
print(df.duplicated().sum())

# Display all column names 
print("\nColumn Names:")
print(df.columns.tolist())


#Create a copy of original dataset 
df_clean = df.copy()



#Check missing values again 
print("\nMissing Values After Cleaning:")
print(df_clean.isnull().sum())


import matplotlib.pyplot as plt
import seaborn as sns

# Count delinquent and non-delinquent customers
print(df_clean['Delinquent_Account'].value_counts())

# Visualize customer delinquency
sns.countplot(data=df_clean, x='Delinquent_Account')

plt.title('Delinquency Distribution')
plt.xlabel('Delinquent Account')
plt.ylabel('Number of Customers')

plt.show()

# Income vs Delinquency Analysis 

plt.figure(figsize=(8, 5))

sns.boxplot(
    data=df_clean,
    x='Delinquent_Account',
    y='Income'
)

plt.title('Income vs Delinquency')
plt.xlabel('Delinquency Account (0 = No, 1 = Yes)')
plt.ylabel('Customer Income')

plt.show()


print("\nMedian Income Comparison:")
print(df_clean.groupby('Delinquent_Account')
      ['Income'].median())
print("\nAverage Income Comparison:")
print(df_clean.groupby('Delinquent_Account')
      ['Income'].mean())

# Missed Payments Analysis

print("\nMissed Payments Comparison:")

print(
    df_clean.groupby('Delinquent_Account')
    ['Missed_Payments'].mean()
)

plt.figure(figsize=(8, 5))

sns.boxplot(
    data=df_clean,
    x='Delinquent_Account',
    y='Missed_Payments'
)

plt.title('Missed Payments vs Delinquency')
plt.xlabel('Delinquency Account (0=No, 1=Yes)')
plt.ylabel('Missed Payments')

plt.show()


#Credit Score Analysis

print("\nCredit Score Comparison:")

print(
    df_clean.groupby('Delinquent_Account')
    ['Credit_Score'].mean()
)

plt.figure(figsize=(8, 5))

sns.boxplot(
    data=df_clean,
    x='Delinquent_Account',
    y='Credit_Score'
)

plt.title('Credit Score vs Delinquency')
plt.xlabel('Delinquent Account (0=No, 1=Yes)')
plt.ylabel('Credit Score')

plt.show()

# Loan Balance Analysis

print("\nLoan Balance Comparison:")

print(
    df_clean.groupby('Delinquent_Account')['Loan_Balance'].mean()
)

plt.figure(figsize=(8, 5))

sns.boxplot(
    data=df_clean,
    x='Delinquent_Account',
    y='Loan_Balance'
)

plt.title('Loan Balance vs Delinquency')
plt.xlabel('Delinquent Account (0=No, 1=Yes)')
plt.ylabel('Loan Balance')

plt.show()


# Check categorial columns
print("\nCategorical Columns:")

print(df_clean.select_dtypes(include='object').
      columns.tolist())

 # Check data types 
print("\nData Types:")

print(df_clean.dtypes)

# Inspect categorical values

categorical_columns = [
    'Employment_Status',
    'Credit_Card_Type',
    'Location',
    'Month_1',
    'Month_2',
    'Month_3',
    'Month_4',
    'Month_5',
    'Month_6'
]

for col in categorical_columns:

    print("\n----------------------")
    print("Column:", col)
    print("----------------------")

    print(df_clean[col].value_counts(dropna=False))

    # Standardize Employment Status

df_clean['Employment_Status'] = (
    df_clean['Employment_Status']
    .replace({
        'EMP': 'Employed',
        'employed': 'Employed',
        'retired': 'Retired'
    })
)

# Check cleaned values

print("\nCleaned Employment Status:")

print(df_clean['Employment_Status'].value_counts())

# Monthly Payment Status vs Delinquency

for col in ['Month_1', 'Month_2', 'Month_3',
            'Month_4', 'Month_5', 'Month_6']:

    print("\n", col)

    print(pd.crosstab(
        df_clean[col],
        df_clean['Delinquent_Account'],
        normalize='index'
    ).round(2))

    # Select features for baseline model

features = [
    'Age',
    'Income',
    'Credit_Score',
    'Credit_Utilization',
    'Missed_Payments',
    'Loan_Balance',
    'Debt_to_Income_Ratio',
    'Account_Tenure',
    'Employment_Status',
    'Credit_Card_Type',
    'Location'
]

# Monthly Payment History Analysis
print("\nMonthly Payment History:")
for col in ['Month_1', 'Month_2', 'Month_3', 'Month_4', 'Month_5', 'Month_6']:
    print(f"\n{col}:")
    print(df_clean[col].value_counts())

print("\nMonthly Payment Status vs Delinquency:")

for col in ['Month_1', 'Month_2', 'Month_3', 'Month_4', 'Month_5', 'Month_6']:
    print(f"\n{col}:")
    print((pd.crosstab(df_clean[col], df_clean['Delinquent_Account'], normalize='index') * 100).round(2))

print("\n")

X = df_clean[features]

y = df_clean['Delinquent_Account']

print("\nFeatures Shape:", X.shape)
print("Target Shape:", y.shape)

print("\nFeature Information:")
print(X.dtypes)

print("\nNumerical Feature Correlation with Target:")
print(df_clean[features + ['Delinquent_Account']].corr(numeric_only=True)['Delinquent_Account'].sort_values(ascending=False))

# Categorical Feature analysis
print("\nCategorical Feature Analysis:")

for col in ['Employment_Status', 'Credit_Card_Type', 'Location']:
    print(f"\n{col} vs Delinquency:")
    print(pd.crosstab(df_clean[col], df_clean['Delinquent_Account'], normalize='index').round(3) * 100)
print("\nTarget Distribution:")
print(y.value_counts())    

print("\nTarget Distribution:")
print(y.value_counts())


from sklearn.model_selection import train_test_split
# First Train-test split 

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
# Second Train-Validation Split
X_train_new, X_val, y_train_new, y_val = train_test_split(
    X_train,
    y_train,
    test_size=0.2,
    random_state=42,
    stratify=y_train
)

print("Training Features:", X_train.shape)
print("Testing Features:", X_test.shape)

print("\nTraining Target:")
print(y_train.value_counts())

print("\nTesting Target:")
print(y_test.value_counts())

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

numeric_features = [
    'Age',
    'Income',
    'Credit_Score',
    'Credit_Utilization',
    'Missed_Payments',
    'Loan_Balance',
    'Debt_to_Income_Ratio',
    'Account_Tenure'
]

categorical_features = [
    'Employment_Status',
    'Credit_Card_Type',
    'Location'
]

# Numerical Feature Preprocessing
numeric_transformer = Pipeline(steps=[('imputer',SimpleImputer(strategy='median')), ('scaler',StandardScaler())])

# Categorical Features Preprocessing
categorical_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='most_frequent')), ('encoder', OneHotEncoder(handle_unknown='ignore'))])

# Combined Preprocessor 
preprocessor = ColumnTransformer(transformers=[('num', numeric_transformer, numeric_features), ('cat', categorical_transformer, categorical_features)])
print("Preprocessing setup completed!")

# Verify Missing Value Imputation

imputed_data = preprocessor.fit_transform(X_train_new)

print("\nMissing Values Before Imputation:")
print(X_train_new.isnull().sum())

print("\nMissing Values After Imputation:")
print(pd.DataFrame(imputed_data).isnull().sum().sum())


from sklearn.linear_model import LogisticRegression


model= Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(
        max_iter=1000,
        random_state=42,
        class_weight="balanced"
    ))
])

# version 3: Random Forest Model
rf_model = Pipeline(steps=[('preprocessor', preprocessor),('classifier', RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight='balanced'
))])


# Version 4 : XGBoost Model

xgb_model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', XGBClassifier(
        n_estimators=200,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=4.0,
        eval_metric='logloss',
        random_state=42 
    ))
])

# Train XGBoost Model 
xgb_model.fit(X_train_new, y_train_new)

xgb_probability = xgb_model.predict_proba(X_val)
xgb_pred = (xgb_probability[:, 1] >= 0.10).astype(int)

# XGBoost Evaluation 
print("\nXGBoost Accuracy:")
print(accuracy_score(y_val, xgb_pred))

print("\nXGBoost Confusion Matrix:")
print(confusion_matrix(y_val, xgb_pred))

print("\nXGBoost Classification Report:")
print(classification_report(y_val, xgb_pred, zero_division=0))


# Train Random Forest Model 

rf_model.fit(X_train_new, y_train_new)

# Predict Probabilities on validation data
rf_probability = rf_model.predict_proba(X_val)

# Apply threshold 0.32 
rf_pred = (rf_probability[:, 1]>= 0.31).astype(int)

print("\nRandom Forest Training Completed!")


# Random Forest Evaluation 

print("\nRandom Forest Accuracy:")
print(accuracy_score(y_val, rf_pred))

print("\nRandom Forest Confusion Matrix:")
print(confusion_matrix(y_val, rf_pred))

print("\nRandom Forest Classification Report:")
print(classification_report(y_val, rf_pred, zero_division=0))

# Train the LogisticRegression model 
model.fit(X_train, y_train)

# Save trained model
joblib.dump(model,"logistic_regression_model.pkl")

# Load Saved Model
loaded_model = joblib.load("logistic_regression_model.pkl")
print("Saved Model Loaded Successfully!")

print("Model save successfully!")

# Create New Customer Data
new_customer = pd.DataFrame([{'Age': 30, 
                              'Income': 60000, 
                              'Credit_Score': 650,
                              'Credit_Utilization': 0.35,
                              'Missed_Payments': 2,
                              'Loan_Balance': 25000,
                              'Debt_to_Income_Ratio': 0.30,
                              'Account_Tenure': 4,
                              'Employment_Status': 'Employed',
                              'Credit_Card_Type': 'Standard',
                              'Location': 'Bhopal'}])

print("\nNew Customer Data:")
print(new_customer)

print("Expected Features:", features)
print("New Customer Columns:", new_customer.columns.tolist())
print("Missing Columns:", set(features) - set(new_customer.columns))

print("\nAll New Customer Columns:")
print(new_customer.columns.tolist())
print("\nDataFrame Shape:")
print(new_customer.shape)
print("\nCheck Debt Column:")
print('Debt_to_Income_Ratio' in new_customer.columns)
print(set(features) - set(new_customer.columns))

# Predict New Customer

prediction = loaded_model.predict(new_customer)

print("\nNew Customer Prediction:")

if prediction[0] == 1:
    print("Customer is likely to be Delinquent.")
else:
    print("Customer is likely to be Non-Delinquent.")    

# Validation Predictions
val_probability = model.predict_proba(X_val)

val_threshold = 0.45

val_pred = (val_probability[:, 1] >= val_threshold).astype(int)

print("\nValidation Results")
print("Accuracy:", accuracy_score(y_val, val_pred))
print("Confusion Matrix:\n", confusion_matrix(y_val, val_pred))
print("Classification Report:\n",
      classification_report(y_val, val_pred, zero_division=0))


# Retrain on Complete Training Data
model.fit(X_train, y_train)


# Final Test Predictions
y_probability = model.predict_proba(X_test)

y_pred = (y_probability[:, 1] >= val_threshold).astype(int)

print("\nFinal Test Results")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n",
      classification_report(y_test, y_pred, zero_division=0))

# Fairness Evaluation
fairness_data = X_test.copy()

fairness_data["Actual"] = y_test.to_numpy()
fairness_data["Predicted"] = y_pred

fairness_data["True_Positive"] = (
    (fairness_data["Actual"] == 1) &
    (fairness_data["Predicted"] == 1)
).astype(int)

fairness_data["Actual_Positive"] = (
    fairness_data["Actual"] == 1
).astype(int)

fairness_data["Predicted_Positive"] = (
    fairness_data["Predicted"] == 1
).astype(int)


# Check Fairness Across Customer Groups
fairness_columns = [
    "Employment_Status",
    "Location",
    "Credit_Card_Type"
]

for column in fairness_columns:

    print(f"\nFairness Analysis by {column}")

    fairness_summary = fairness_data.groupby(
        column, dropna=False
    ).agg(
        Total_Customers=("Actual", "size"),
        Actual_Delinquent=("Actual_Positive", "sum"),
        Predicted_Delinquent=("Predicted_Positive", "sum"),
        True_Positive=("True_Positive", "sum")
    )

    fairness_summary["Actual_Delinquency_Rate"] = (
        fairness_summary["Actual_Delinquent"] /
        fairness_summary["Total_Customers"] * 100
    )

    fairness_summary["Predicted_Delinquency_Rate"] = (
        fairness_summary["Predicted_Delinquent"] /
        fairness_summary["Total_Customers"] * 100
    )

    fairness_summary["Recall"] = (
        fairness_summary["True_Positive"] /
        fairness_summary["Actual_Delinquent"].replace(0, np.nan)
        * 100
    )

    print(fairness_summary.round(2))


from sklearn.metrics import(
    accuracy_score,
    confusion_matrix,
    classification_report
)

# Model Evalution 
accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:", accuracy_score(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))


#XGBoost Final Test Evaluation

xgb_model.fit(X_train, y_train)

xgb_test_probability = xgb_model.predict_proba(X_test)

xgb_test_pred = (xgb_test_probability[:, 1] >= 0.20).astype(int)

print("\nXGBoost Final Test Accuracy:")
print(accuracy_score(y_test, xgb_test_pred))

print("\nXGBoost Final Test Confusion Matrix:")
print(confusion_matrix(y_test, xgb_test_pred))

print("\nXGBoost Final Test Classification Report:")
print(classification_report(y_test, xgb_test_pred, zero_division=0))

#XGBoost Threshold Tunning 

thresholds=[0.10, 0.20, 0.30, 0.40, 0.50]

threshold_results=[]

for threshold in thresholds:
    predictions = (xgb_probability[:, 1] >= threshold).astype(int)

    report = classification_report(
        y_val, predictions, output_dict=True, zero_division=0
    )

    threshold_results.append({
        "Threshold": threshold, "Accuracy": accuracy_score(y_val, predictions), 
        "Precision": report["1"]["precision"], "Recall": report["1"]["recall"],
        "F1-Score": report["1"]["f1-score"]
    })

threshold_df = pd.DataFrame(threshold_results)

print("\nXGBoost Threshold Comparison:")
print(threshold_df.round(3).to_string(index=False))


# Random Forest Final Test Evaluation 

rf_model.fit(X_train, y_train)

rf_test_probability = rf_model.predict_proba(X_test)

rf_test_pred = (rf_test_probability[:, 1] >=0.32).astype(int)

print("\nRandom Forest Final Test Accuracy")
print(accuracy_score(y_test, rf_test_pred))

print("\nRandom Forest Final Test Confusion Matrix:")
print(confusion_matrix(y_test, rf_test_pred))

print("\nRandom Forest Final Test Classification Report:")
print(classification_report(y_test, rf_test_pred, zero_division=0))


# Model Comparison - Final Test Results

models = [("Logistic Regression", y_pred), ("Random Forest", rf_test_pred),
          ("XGBoost", xgb_test_pred)]

comparison_results = []

for model_name, predictions in models:
    report=classification_report(
        y_test, predictions, output_dict=True, zero_division=0
    )

    comparison_results.append({
        "Model": model_name, "Accuracy": accuracy_score(y_test, predictions),
        "precision":report["1"]["precision"],
        "Recall":report["1"]["recall"],
        "F1-Score":report["1"]["f1-score"]
    })
model_comparison=pd.DataFrame(comparison_results)

print("\nFinal Model Comparison:")
print(model_comparison.round(3))


# XGBoost Feature Importance

feature_names = xgb_model.named_steps['preprocessor'].get_feature_names_out()
importance_values = xgb_model.named_steps['classifier'].feature_importances_
importance_df = pd.DataFrame({
    "Feature": feature_names, "Importance": importance_values
})
importance_df = importance_df.sort_values(by="Importance", ascending=False)

print("\nXGBoost Feature Importance:")
print(importance_df.head(10).round(4).to_string(index=False))


# Feature Importance Graph 

plt.figure(figsize=(10, 6))
plt.barh(importance_df.head(10)["Feature"][::-1], importance_df.head(10)["Importance"][::-1])

plt.xlabel("Importance Score")
plt.ylabel("Features")
plt.title("Top 10 XGBoost Feature Importance")
plt.tight_layout()
plt.show()

# Permutation Feature Importance

perm_result = permutation_importance(xgb_model, X_test, y_test, n_repeats=10, random_state=42,
                                     scoring='roc_auc')

permutation_df = pd.DataFrame({"Feature": X_test.columns, "Importance": perm_result.importances_mean})

permutation_df = permutation_df.sort_values(by="Importance", ascending=False)

print("\nPermutation Feature Importance:")
print(permutation_df.round(4).to_string(index=False))


# 5-Fold Cross Validation - All 3 Models

models = [("Logistic Regression", model), ("Random Forest", rf_model), ("XGBoost", xgb_model)]

cv_results = []

for model_name, current_model in models:
    scores = cross_val_score(current_model, X_train, y_train, cv=5, scoring='roc_auc')

    cv_results.append({"Model": model_name, "Average ROC-AUC": scores.mean(), "Standard Deviation": scores.std()})
    print(f"\n{model_name} CV Scores:")
    print(scores.round(3))
cv_df = pd.DataFrame(cv_results)

print("\nFinal Cross Validation Comparison:")
print(cv_df.round(3).to_string(index=False))