# Breast Cancer Prediction App

This project involves building a machine learning model to predict whether a breast tumor is malignant or benign based on various characteristics of the tumor. We use the Breast Cancer dataset provided by `sklearn` for training the model, and the model is deployed via a Streamlit web application.

## Project Components

1. **Model Training (app.ipynb)**: 
    - Loads the Breast Cancer dataset.
    - Preprocesses the data, including scaling and feature selection.
    - Trains a neural network (MLPClassifier) and tunes hyperparameters using GridSearchCV.
    - Saves the best model, feature selector, and scaler using `joblib` for later use.

2. **Web Application (stream.py)**: 
    - Provides an interactive interface using Streamlit for users to input features and get predictions.
    - Loads the saved model, scaler, and selected features.
    - Scales input data, performs predictions, and displays results (malignant/benign with probabilities).

## Project Setup

### Prerequisites

Ensure you have Python 3.7 or later installed. You can create a virtual environment to manage dependencies.

### Install Dependencies

Run the following command to install the required dependencies listed in `requirements.txt`:

```bash
pip install -r requirements.txt

This will install all the necessary packages for model training, hyperparameter tuning, and running the Streamlit app.
Running the Model Training

You can train the model by running the Jupyter notebook (app.ipynb). This will perform the following:

    Load and preprocess the dataset.
    Perform feature selection using SelectKBest.
    Train an MLPClassifier with hyperparameter tuning using GridSearchCV.
    Save the model, feature selector, and scaler for later use.

jupyter notebook app.ipynb

Running the Streamlit Web Application

Once the model is trained and the necessary files (model.joblib, scaler.joblib, and selected_features.joblib) are saved, you can run the Streamlit web app by executing:

streamlit run stream.py

This will open the Streamlit app in your browser, where you can input the values for selected features and get a prediction (malignant or benign) along with the probability of malignancy.
File Description

    app.ipynb: Jupyter notebook used for data preprocessing, model training, and saving the trained model.
    stream.py: Streamlit app for providing a user-friendly interface to interact with the trained model.
    model.joblib: The saved MLPClassifier model.
    scaler.joblib: The saved StandardScaler used to scale the input data.
    selected_features.joblib: The list of selected features used for prediction.

Model Details

The model uses a Multi-layer Perceptron (MLP) classifier for binary classification (malignant vs benign). Hyperparameter tuning is done using GridSearchCV, exploring different combinations of parameters like hidden layer sizes, activation functions, solvers, and regularization parameters.