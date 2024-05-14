# 4th-Sem-Minor-Project
# Breast Cancer Detection using Logistic Regression

This project aims to develop a machine learning model for detecting breast cancer using logistic regression. The model is trained on a dataset containing features computed from fine needle aspirate (FNA) images of breast masses. The goal is to classify the masses as either benign or malignant based on the input features.

## Dataset

The dataset used in this project is the Breast Cancer Wisconsin (Diagnostic) Dataset, obtained from the University of Wisconsin Hospitals, Madison. It consists of 569 instances, each with 30 features describing the characteristics of cell nuclei present in the FNA images. The dataset is available on the Kaggle platform and can be accessed through the following link:

[https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data)

## Prerequisites

To run this project, you need to have the following dependencies installed:

- Python 3.x
- NumPy
- Pandas
- Scikit-learn

You can install these dependencies using pip:
```python
pip install numpy pandas scikit-learn
```
## Usage

1. Clone the repository or download the source code.
2. Place the `cancer_dataset.csv` file in the project directory.
3. Run the `cancer.py` script: ```python cancer.py```
## The script will perform the following steps:

1. Load the dataset from the CSV file.
2. Preprocess the data by separating the features and target variable, and splitting the dataset into training and testing sets.
3. Implement the logistic regression model using the scikit-learn library.
4. Train the model on the training data.
5. Make predictions on the test data.
6. Evaluate the model's performance using various metrics, such as accuracy, confusion matrix, and classification report.

The script will print the evaluation metrics to the console, allowing you to analyze the model's performance.

## Customization

You can customize the code to experiment with different techniques or improve the model's performance. For example, you can:

- Explore feature engineering techniques to create more informative features.
- Tune the hyperparameters of the logistic regression model, such as the regularization parameter or the regularization type.
- Implement different machine learning algorithms and compare their performance.
- Integrate cross-validation techniques for better model evaluation and selection.

## Contributing

Contributions to this project are welcome. If you find any issues or have suggestions for improvements, please open an issue or submit a pull request.

## License

This project is licensed under the [MIT License](LICENSE).

## Acknowledgments

- The Breast Cancer Wisconsin (Diagnostic) Dataset was obtained from the University of Wisconsin Hospitals, Madison, and made available through the UCI Machine Learning Repository.
- This project was inspired by the need for accurate and reliable cancer detection systems to support healthcare professionals and improve patient outcomes.
