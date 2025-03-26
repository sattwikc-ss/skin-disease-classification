import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from pyngrok import ngrok
import os

# Load trained model
model = tf.keras.models.load_model('/content/skin_disease_model.h5')
class_names = ['Actinic keratosis', 'Atopic Dermatitis', 'Benign keratosis', 'Dermatofibroma', 'Melanocytic nevus',
               'Melanoma', 'Squamous cell carcinoma', 'Tinea Ringworm Candidiasis', 'Vascular lesion']

# Streamlit UI
st.title("Skin Disease Classifier")
st.write("Upload an image to classify.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Save file temporarily
    img_path = "uploaded_image.jpg"
    with open(img_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Display uploaded image
    st.image(img_path, caption="Uploaded Image", use_column_width=True)

    # Preprocess image
    img = load_img(img_path, target_size=(224, 224))
    img_array = img_to_array(img) / 255.0
    img_array = img_array.reshape(1, 224, 224, 3)

    # Make prediction
    predictions = model.predict(img_array)
    predicted_class = class_names[np.argmax(predictions)]
    probabilities = {class_names[i]: float(predictions[0][i]) for i in range(len(class_names))}

    # Show results
    st.write(f"**Predicted Class:** {predicted_class}")
    st.write("**Confidence Scores:**")
    for cls, prob in probabilities.items():
        st.write(f"{cls}: {prob:.4f}")
