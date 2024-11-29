Breast Cancer Prediction Web App

This project aims to predict whether a breast tumor is malignant or benign based on various characteristics of the tumor, using machine learning techniques. The model is built using the Breast Cancer dataset provided by the sklearn library, and the model is deployed via an interactive Streamlit web application, enabling users to input tumor features and receive predictions.
Table of Contents

    Project Overview
    Key Features
    Technology Stack
    Getting Started
    Model Training
    Streamlit Web App
    File Descriptions
    How to Use
    Evaluation and Results
    License

Project Overview

The goal of this project is to develop a machine learning model that predicts whether a breast tumor is malignant or benign based on features such as radius, texture, perimeter, area, smoothness, concavity, etc. The model is trained using the Multi-layer Perceptron (MLP) classifier, which is tuned for optimal performance using GridSearchCV to select the best hyperparameters.

The project includes two major parts:

    Model Training: The model is trained in a Jupyter notebook (app.ipynb), which also includes data preprocessing, feature selection, and hyperparameter tuning.
    Deployment with Streamlit: A web-based interface (stream.py) is built using Streamlit, allowing users to interact with the trained model and receive predictions based on their input features.

Key Features

    Feature Selection: The top 10 most relevant features are selected using SelectKBest to improve model performance and reduce overfitting.
    Hyperparameter Tuning: Grid search is used to tune the model's hyperparameters (e.g., hidden layer sizes, activation functions, solver types) to achieve the best performance.
    Web Interface: An easy-to-use Streamlit app allows users to input feature values and see predictions in real-time, including probabilities for the tumor being malignant or benign.
    Model Persistence: The trained model, scaler, and selected features are saved using joblib, ensuring the model can be reused without retraining.

Technology Stack

    Python: The primary programming language used.
    scikit-learn: Machine learning library for building, training, and evaluating models.
    Streamlit: A framework for building interactive web apps for machine learning models.
    Joblib: For saving and loading trained models and pre-processing tools.
    Jupyter Notebook: For exploratory data analysis, model training, and evaluation.

Getting Started

To run this project locally, you'll need Python 3.7 or later installed. The following steps outline how to set up the project.
Prerequisites

Make sure you have Python 3.7+ installed. You can also set up a virtual environment to keep the dependencies isolated:

python3 -m venv venv
source venv/bin/activate   # On Windows, use `venv\Scripts\activate`

Install Dependencies

You can install the required dependencies using pip. First, clone the repository (or download the project files), then navigate to the project folder and install the dependencies:

pip install -r requirements.txt

This will install all necessary libraries such as pandas, numpy, scikit-learn, streamlit, and others.
Model Training

The model training process is done in the app.ipynb Jupyter notebook. This notebook includes the following steps:

    Load Dataset: The Breast Cancer dataset is loaded from sklearn.
    Data Preprocessing: Missing values are checked, and the features are scaled using StandardScaler.
    Feature Selection: The top 10 features are selected using SelectKBest with the f_classif scoring function.
    Model Training: A Multi-layer Perceptron (MLP) classifier is trained using GridSearchCV to tune hyperparameters like the number of hidden layers, activation function, and solver.
    Saving the Model: The trained model, scaler, and selected features are saved using joblib for later use in the Streamlit app.

To run the notebook, execute the following in your terminal:

jupyter notebook app.ipynb

This will start the Jupyter Notebook server and open the notebook in your browser. You can train the model and save the necessary files (model.joblib, scaler.joblib, and selected_features.joblib).
Streamlit Web App

Once the model is trained and the necessary files are saved, you can launch the Streamlit web app by running:

streamlit run stream.py

This command will start a local server and open the app in your web browser. You will be able to input the tumor characteristics and get predictions in real-time.
Input Features

The app will prompt you to enter values for the following features:

    Mean Radius
    Mean Perimeter
    Mean Area
    Mean Smoothness
    Mean Concavity
    Worst Radius
    Worst Texture
    Worst Smoothness
    Worst Concavity
    Worst Area

Predictions

Once you enter the feature values and click on "Predict", the app will:

    Scale the input data using the saved scaler.joblib.
    Predict whether the tumor is malignant or benign using the saved model.joblib.
    Show the prediction result along with the probability of malignancy.

File Descriptions

    app.ipynb: Jupyter notebook for data preprocessing, model training, hyperparameter tuning, and saving the model.
    stream.py: Streamlit app that allows users to input features and get predictions.
    model.joblib: Saved machine learning model after training (best model from GridSearchCV).
    scaler.joblib: Saved StandardScaler used to scale the input features before feeding them into the model.
    selected_features.joblib: List of features selected during the feature selection process.

How to Use

    Model Training: First, run the app.ipynb notebook to train the model, scale the data, and save the necessary files.

    Launch Streamlit: Once the model is saved, run the Streamlit app using the command:

    streamlit run stream.py

    Input Features: In the web app, enter values for the tumor features.

    Get Prediction: Click the "Predict" button to receive the prediction (malignant or benign) along with the probability.

Evaluation and Results

After training the model using GridSearchCV and evaluating it on the test set, the following evaluation metrics were achieved:

    Accuracy: 97% (on the test set).
    Classification Report: Displays precision, recall, and F1-score for both the malignant and benign classes.

These results indicate that the model performs well on this dataset, providing accurate predictions for breast cancer diagnosis.
License

This project is licensed under the MIT License - see the LICENSE file for details.