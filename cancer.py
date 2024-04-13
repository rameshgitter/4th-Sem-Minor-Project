import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load the dataset
data = pd.read_csv('cancer_dataset.csv')

# Separate the features and target variable
X = data.drop('diagnosis', axis=1)
y = data['diagnosis'].replace({'M': 1, 'B': 0})  # Convert target variable to numeric

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Squared error cost function
def cost_function(theta, X, y):
    m = len(y)
    h = sigmoid(np.dot(X, theta))
    return (-1/m) * (np.dot(y.T, np.log(h)) + np.dot((1 - y).T, np.log(1 - h)))

# Gradient descent
def gradient_descent(theta, X, y, alpha, num_iters):
    m = len(y)
    theta = theta.copy()
    
    for i in range(num_iters):
        h = sigmoid(np.dot(X, theta))
        theta = theta - (alpha/m) * np.dot(X.T, (h - y))
    
    return theta

# Train the logistic regression model
initial_theta = np.zeros(X.shape[1])
alpha = 0.01
num_iters = 1000

theta = gradient_descent(initial_theta, X_train, y_train, alpha, num_iters)

# Make predictions on the test set
y_pred = sigmoid(np.dot(X_test, theta)) >= 0.5

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred)

print("Accuracy:", accuracy)
print("Confusion Matrix:\n", cm)
print("Classification Report:\n", report)
