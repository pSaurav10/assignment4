import streamlit as st
import pandas as pd
import joblib
import warnings

# Suppress all warnings
warnings.filterwarnings("ignore")

# Load the model, scaler, and selected feature names
model = joblib.load('model.joblib')
scaler = joblib.load('scaler.joblib')
selected_features = joblib.load('selected_features.joblib')

# Set up the Streamlit app
st.title("Breast Cancer Prediction App")
st.write("""
    This application predicts whether a tumor is malignant or benign based on the input features from the Breast Cancer dataset.
    Enter the values for each feature, and the app will predict the outcome.
""")

# Create input fields dynamically based on the selected features
user_input = {}

# Create an input box for each feature
for feature in selected_features:
    feature_display = feature.replace('_', ' ').capitalize()  # Make the feature name more user-friendly
    user_input[feature] = st.number_input(f"Enter value for {feature_display}:",
                                         min_value=0.0, step=0.1)

# Check if the user filled all the fields before proceeding
missing_fields = [feature for feature, value in user_input.items() if value is None]

if missing_fields:
    st.warning(f"Please fill all the input fields. Missing: {', '.join(missing_fields)}")
else:
    # Convert the user input into a DataFrame
    input_data = pd.DataFrame([user_input])

    # Predict button
    if st.button("Predict"):
        try:
            # Ensure that the input data has the correct columns in the correct order
            input_data_selected = input_data[selected_features]

            # Scale the input data
            input_scaled = scaler.transform(input_data_selected)

            # Make predictions
            prediction = model.predict(input_scaled)
            prediction_proba = model.predict_proba(input_scaled)

            # Show the results
            result = "Malignant" if prediction[0] == 1 else "Benign"
            st.write(f"Prediction: **{result}**")
            st.write(f"Probability of Malignancy: {prediction_proba[0][1] * 100:.2f}%")
        except KeyError as e:
            st.error(f"Error: Missing feature {e} in the input data.")
        except ValueError as e:
            st.error(f"Error: Invalid input data format. Please ensure all features are correctly filled.")
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")
