import streamlit as st
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import pandas as pd
from streamlit_option_menu import option_menu
import pickle
from PIL import Image
import numpy as np
import plotly.figure_factory as ff
from code.DiseaseModel import DiseaseModel
from code.helper import prepare_symptoms_array
import seaborn as sns
import joblib
import http.client
import requests
import base64
import io

# Try loading joblib model safely
try:
    lung_cancer_model = joblib.load('models/lung_cancer_model.sav')
except Exception as e:
    lung_cancer_model = None
    st.warning(f"Could not load lung cancer model: {e}")

# Try to import TensorFlow and Keras helpers — set have_tf flag
try:
    import tensorflow as tf
    from tensorflow.keras.applications import ResNet50
    from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
    have_tf = True
except Exception:
    have_tf = False

# Option menu
selected = option_menu('Early Disease Prediction', [
    'Disease Prediction',
    'Lung Cancer Prediction',
],
    icons=['','activity', 'heart'],
    default_index=0,
    orientation="horizontal",
    )

# sidebar
with st.sidebar:
    st.sidebar.title("Welcome to the Disease Prediction Project")
    st.sidebar.write(
        "-The project predicts what disease the patient might be suffering and how likely they are to have that disease.\n\n"
        "-The user does not need to traverse different places in order to predict whether he/she has a particular disease or not."
    )
    # spacing
    for _ in range(7):
        st.sidebar.title("")

# multiple disease prediction
if selected == 'Disease Prediction':
    # Create disease class and load ML model
    disease_model = DiseaseModel()
    disease_model.load_xgboost('model/xgboost_model.json')

    # Title
    st.write('# Disease Prediction using Machine Learning')

    symptoms = st.multiselect('What are your symptoms?', options=disease_model.all_symptoms)

    X = prepare_symptoms_array(symptoms)

    # Trigger XGBoost model
    if st.button('Predict'):
        prediction, prob = disease_model.predict(X)
        st.write(f'## Disease: {prediction} with {prob*100:.2f}% probability')

        tab1, tab2 = st.tabs(["Description", "Precautions"])
        with tab1:
            st.write(disease_model.describe_predicted_disease())
        with tab2:
            precautions = disease_model.predicted_disease_precautions()
            for i in range(4):
                st.write(f'{i+1}. {precautions[i]}')

# Load the dataset
lung_cancer_data = pd.read_csv('data/lung_cancer.csv')

# Convert 'M' to 'Male' and 'F' to 'Female' in the 'GENDER' column
lung_cancer_data['GENDER'] = lung_cancer_data['GENDER'].map({'M': 'Male', 'F': 'Female'})

# Lung Cancer prediction page
if selected == 'Lung Cancer Prediction':
    st.title("Lung Cancer Prediction")
    try:
        image = Image.open('h.png')
        st.image(image, caption='Lung Cancer Prediction')
    except Exception:
        pass

    # Columns
    name = st.text_input("Name:")
    col1, col2, col3 = st.columns(3)

    with col1:
        gender = st.selectbox("Gender:", lung_cancer_data['GENDER'].unique())

    with col2:
        url = "https://age-detection2.p.rapidapi.com/age"
        headers = {
            "x-rapidapi-key": "bbc5145240msh0e91c37e7185464p106ea8jsn408663b4fe84",  # Replace with your own key if needed
            "x-rapidapi-host": "age-detection2.p.rapidapi.com",
            "Content-Type": "application/json"
        }

        age = "Upload image for age detection"
        uploaded_file2 = st.file_uploader("Age Detection", type=["jpg", "jpeg", "png"])

        if uploaded_file2 is not None:
            try:
                image_for_age = Image.open(uploaded_file2)
                buffered = io.BytesIO()
                image_for_age.save(buffered, format="JPEG")
                img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
                payload = {"image": f"data:image/jpeg;base64,{img_str}", "return_face": True}
                response = requests.post(url, json=payload, headers=headers)
                if response.status_code == 200:
                    result = response.json()
                    age = result.get("age", age)
                else:
                    st.write("Error: Upload another image")
            except Exception as e:
                st.write("Error processing age image:", e)

        st.write(str(age))

    with col3:
        smoking = st.selectbox("Smoking:", ['NO', 'YES'])
    with col1:
        yellow_fingers = st.selectbox("Yellow Fingers:", ['NO', 'YES'])

    # ---- Anxiety detection block (TensorFlow guarded) ----
    # Provide default for anxiety so later code won't fail if TF disabled
    anxiety = 'NO'

    with col2:
        # Define preprocess function that works when TF is absent
        def preprocess_image(img):
            img = img.resize((224, 224))
            img_array = np.array(img)
            img_array = np.expand_dims(img_array, axis=0)
            if have_tf:
                # uses preprocess_input if TF available
                img_array = preprocess_input(img_array)
            return img_array

        if have_tf:
            # Load model lazily (only if TF available)
            try:
                model = ResNet50(weights='imagenet')
            except Exception as e:
                st.warning(f"Could not load ResNet50 model: {e}")
                model = None

            uploaded_file = st.file_uploader("Anxiety Detection", type=["png", "jpg", "jpeg"])

            if uploaded_file is not None and model is not None:
                try:
                    uploaded_image = Image.open(uploaded_file)
                    img_array = preprocess_image(uploaded_image)
                    predictions = model.predict(img_array)
                    decoded_predictions = tf.keras.applications.resnet50.decode_predictions(predictions, top=3)[0]

                    anxiety_detected = False
                    for _, label, score in decoded_predictions:
                        if 'fear' in label.lower() or 'stress' in label.lower():
                            anxiety_detected = True
                            break

                    if anxiety_detected:
                        anxiety = 'YES'
                        st.write("Anxiety detected based on the image.")
                    else:
                        anxiety = 'NO'
                        st.write("No anxiety detected based on the image.")
                except Exception as e:
                    st.write("Error during anxiety detection:", e)
        else:
            st.info("TensorFlow is not available on the server — image-based anxiety detection disabled.")
            _ = st.file_uploader("Anxiety Detection (disabled)", type=["png", "jpg", "jpeg"], disabled=True)
            anxiety = 'NO'
    # ------------------------------------------------------

    with col3:
        peer_pressure = st.selectbox("Peer Pressure:", ['NO', 'YES'])
    with col1:
        chronic_disease = st.selectbox("Chronic Disease:", ['NO', 'YES'])
    with col2:
        fatigue = st.selectbox("Fatigue:", ['NO', 'YES'])
    with col3:
        allergy = st.selectbox("Allergy:", ['NO', 'YES'])
    with col1:
        wheezing = st.selectbox("Wheezing:", ['NO', 'YES'])
    with col2:
        alcohol_consuming = st.selectbox("Alcohol Consuming:", ['NO', 'YES'])
    with col3:
        coughing = st.selectbox("Coughing:", ['NO', 'YES'])
    with col1:
        shortness_of_breath = st.selectbox("Shortness of Breath:", ['NO', 'YES'])
    with col2:
        swallowing_difficulty = st.selectbox("Swallowing Difficulty:", ['NO', 'YES'])
    with col3:
        chest_pain = st.selectbox("Chest Pain:", ['NO', 'YES'])

    # Code for prediction
    cancer_result = ''

    # Button
    if st.button("Predict Lung Cancer"):
        # Ensure model exists
        if lung_cancer_model is None:
            st.error("Lung cancer model not loaded. Prediction not available.")
        else:
            # Create a DataFrame with user inputs
            user_data = pd.DataFrame({
                'GENDER': [gender],
                'AGE': [age],
                'SMOKING': [smoking],
                'YELLOW_FINGERS': [yellow_fingers],
                'ANXIETY': [anxiety],
                'PEER_PRESSURE': [peer_pressure],
                'CHRONICDISEASE': [chronic_disease],
                'FATIGUE': [fatigue],
                'ALLERGY': [allergy],
                'WHEEZING': [wheezing],
                'ALCOHOLCONSUMING': [alcohol_consuming],
                'COUGHING': [coughing],
                'SHORTNESSOFBREATH': [shortness_of_breath],
                'SWALLOWINGDIFFICULTY': [swallowing_difficulty],
                'CHESTPAIN': [chest_pain]
            })

            # Map string values to numeric
            user_data.replace({'NO': 1, 'YES': 2}, inplace=True)

            # Strip leading and trailing whitespaces from column names
            user_data.columns = user_data.columns.str.strip()

            # Convert columns to numeric where necessary
            numeric_columns = ['AGE', 'FATIGUE', 'ALLERGY', 'ALCOHOLCONSUMING', 'COUGHING', 'SHORTNESSOFBREATH']
            user_data[numeric_columns] = user_data[numeric_columns].apply(pd.to_numeric, errors='coerce')

            # Perform prediction
            try:
                cancer_prediction = lung_cancer_model.predict(user_data)
            except Exception as e:
                st.error(f"Prediction failed: {e}")
                cancer_prediction = ['NO']

            # Display result
            if cancer_prediction[0] == 'YES':
                cancer_result = "The model predicts that there is a risk of Lung Cancer."
                try:
                    image = Image.open('positive.jpg')
                    st.image(image, caption='')
                except Exception:
                    st.write(cancer_result)
            else:
                cancer_result = "The model predicts no significant risk of Lung Cancer."
                try:
                    image = Image.open('negative.jpg')
                    st.image(image, caption='')
                except Exception:
                    st.write(cancer_result)

            st.success(f"{name}, {cancer_result}")
