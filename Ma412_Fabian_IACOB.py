#--*-- coding:utf8 --*--
#-------------------------------------------------------------------------------
# Fabian IACOB 4SM2
# Mi412
# Second session
# Mathematical tools for data science
#-------------------------------------------------------------------------------

import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import seaborn as sns


# Load the datasets
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

# Display information about the datasets
print("\nTrain Data Info:")
print(train_data.info())
print("\nTest Data Info:")
print(test_data.info())

# Fix missing values for 'Arrival Delay in Minutes'
imputer = SimpleImputer(strategy='median')
train_data['Arrival Delay in Minutes'] = imputer.fit_transform(train_data[['Arrival Delay in Minutes']])
test_data['Arrival Delay in Minutes'] = imputer.transform(test_data[['Arrival Delay in Minutes']])

# Encode categorical variables
categorical_features = ['Gender', 'Customer Type', 'Type of Travel', 'Class']

# Using label encoding for simplicity
label_encoders = {}
for feature in categorical_features:
    le = LabelEncoder()
    train_data[feature] = le.fit_transform(train_data[feature])
    test_data[feature] = le.transform(test_data[feature])
    label_encoders[feature] = le

# Encode the target variable
satisfaction_le = LabelEncoder()
train_data['satisfaction'] = satisfaction_le.fit_transform(train_data['satisfaction'])

# Identify numerical features
numerical_features = train_data.select_dtypes(include=['int64', 'float64']).columns.tolist()

# Remove non-feature columns (assuming 'Unnamed: 0' and 'id' exist in your dataset)
non_feature_columns = ['Unnamed: 0', 'id', 'satisfaction']
numerical_features = [col for col in numerical_features if col not in non_feature_columns]

# Normalize numerical features
scaler = StandardScaler()
train_data[numerical_features] = scaler.fit_transform(train_data[numerical_features])
test_data[numerical_features] = scaler.transform(test_data[numerical_features])

# Drop unnecessary columns
train_data.drop(columns=['Unnamed: 0', 'id'], inplace=True)
test_data.drop(columns=['Unnamed: 0', 'id'], inplace=True)

# Split the data into training and validation sets
X = train_data.drop('satisfaction', axis=1)
y = train_data['satisfaction']

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Define and train the models
models = {
    'Logistic regression': LogisticRegression(max_iter=1000),
    'K-Nearest Neighbors': KNeighborsClassifier(),
    'SVM': SVC(),
    'Random Forest': RandomForestClassifier(random_state=42)
}

# Train and evaluate each model
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)
    report = classification_report(y_val, y_pred, target_names=satisfaction_le.classes_)
    print("Model:", name)
    print("Accuracy:", accuracy)
    print("Classification Report:")
    print(report)
    print("-------------------------\n")

# Train the Random Forest model
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)

# Extract feature importance
feature_importances = rf_model.feature_importances_

# Create a DataFrame for feature importance
feature_importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': feature_importances
})

# Sort the DataFrame by importance
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

# Display the feature importance DataFrame
print(feature_importance_df)

plt.figure()
sns.barplot(x='Importance', y='Feature', data=feature_importance_df)
plt.title('Factor importance in satisfaction (Random Forest Model)')
plt.xlabel('Importance')
plt.ylabel('Factor')
plt.show()
