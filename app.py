import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import io

# Load model and class names
model = load_model("model/flower_model.h5")
class_names = ['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips']

def preprocess_image(uploaded_file):
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Original Image", use_column_width=True)
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    return np.expand_dims(img_array, axis=0), img

def predict(image_array):
    pred_probs = model.predict(image_array)[0]
    top_indices = np.argsort(pred_probs)[-3:][::-1]
    return [(class_names[i], pred_probs[i]) for i in top_indices]

# UI
st.title("🌸 Flower Classifier")
st.write("Upload an image of a flower to classify it.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image_array, img_display = preprocess_image(uploaded_file)
    results = predict(image_array)

    st.image(img_display, caption="Preprocessed Image", width=224)

    st.subheader("Prediction Results:")
    for flower, score in results:
        st.write(f"**{flower}**: {score*100:.2f}%")
