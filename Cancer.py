import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load the dataset
data = pd.read_csv('cancer_dataset.csv')

# Display the first few rows of the dataset
print("First few rows of the dataset:")
print(data.head())

# Display the shape of the dataset
print("\nShape of the dataset:")
print(data.shape)

# Display the column names
print("\nColumn names:")
print(data.columns)

# Display the data types of the columns
print("\nData types:")
print(data.dtypes)

# Display summary statistics of the dataset
print("\nSummary statistics:")
print(data.describe())

# Display the distribution of the target variable
print("\nDistribution of the target variable:")
print(data['diagnosis'].value_counts())

print("---------------------------------------------------------------------------------")

# Separate the features and target variable
X = data.drop('diagnosis', axis=1)
y = data['diagnosis']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred)

# Generate a random input array with associated feature names
input_data = pd.DataFrame(np.random.rand(1, X.shape[1]), columns=X.columns)

# Make a prediction using the trained model
prediction = model.predict(input_data)

print("Accuracy:", accuracy)
print("Confusion Matrix:\n", cm)
print("Classification Report:\n", report)

# Print the input data
print("New patient data:")
print(input_data)

# Interpret the prediction
if prediction[0] == 0:
    print("The patient does not have cancer.")
else:
    print("The patient has cancer.")
