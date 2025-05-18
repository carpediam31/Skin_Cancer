import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import joblib
from PIL import Image

# Title of the app
st.title("Skin Cancer Classifier")

# Upload the image
uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    # Open and display the image
    img = Image.open(uploaded_image)
    st.image(img, caption="Uploaded Image", use_container_width=True)

    
    # Preprocess the image (resize to the same size as the model input)
    img = img.resize((256, 256))  # Matching the input size for the model
    img_array = np.array(img) / 255.0  # Normalize the image

    # Reshape it to fit the model input (add a batch dimension)
    img_array = np.expand_dims(img_array, axis=0)

    # Load model and labels (downloaded earlier from Colab)
    model = load_model('model.h5')  # Replace with your model path
    class_names = joblib.load('class_names.pkl')  # Replace with your labels path

    # Make prediction
    prediction = model.predict(img_array)
    
    # Get the predicted class
    predicted_class = class_names[np.argmax(prediction)]

    # Show prediction result
    st.write(f"Prediction: **{predicted_class}**")
