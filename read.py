import pandas as pd

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
