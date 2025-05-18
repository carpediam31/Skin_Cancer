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
    img = img.resize((224, 224))  # Match model input size
    img_array = np.array(img) / 255.0  # Normalize the image

    # Reshape it to fit the model input (add a batch dimension)
    img_array = np.expand_dims(img_array, axis=0)

    # Load model and class names
    model = load_model('model (1).h5')  # Use .keras if needed
    class_names = joblib.load('class_names (1).pkl')

    # Make prediction
    prediction = model.predict(img_array)

    # Get the predicted class
    predicted_class = class_names[np.argmax(prediction)]

    # Show prediction result
    st.write(f"### 🔍 Prediction: **{predicted_class}**")

    # Show probabilities for each class
    st.subheader("📊 Class Probabilities:")
    for i, score in enumerate(prediction[0]):
        st.write(f"{class_names[i]}: {score:.2%}")
