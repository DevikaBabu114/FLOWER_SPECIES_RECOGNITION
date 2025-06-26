import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

# Streamlit page configuration
st.set_page_config(page_title="Flower Classifier", page_icon="🌸", layout="centered")

# Load pre-trained model and class names
model = load_model("flower_classifier_mobilenetv2.h5")
class_names = ['dandelion', 'daisy', 'tulips', 'sunflowers', 'roses']

# Preprocess uploaded image
def preprocess_image(uploaded_file):
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Original Image", use_container_width=True)
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    return np.expand_dims(img_array, axis=0)

# Predict flower class
def predict(image_array):
    pred_probs = model.predict(image_array)[0]
    top_indices = np.argsort(pred_probs)[-3:][::-1]
    return [(class_names[i], pred_probs[i]) for i in top_indices]

# UI Title
st.markdown("<h1 style='text-align: center;'> Flower Classifier</h1>", unsafe_allow_html=True)
st.write("Upload an image of a flower and the model will try to identify it!")

# Upload UI
uploaded_file = st.file_uploader("Choose an image file...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image_array = preprocess_image(uploaded_file)
    results = predict(image_array)

    # Show top prediction
    top_prediction = results[0]
    st.markdown(f"###  Predicted: **{top_prediction[0].capitalize()}** ({top_prediction[1]*100:.2f}%)")

    # Show prediction results
    st.markdown("<h3 style='font-size: 24px;'> Prediction Results</h3>", unsafe_allow_html=True)

    for flower, score in results:
        st.markdown(f"<p style='font-size:18px;'><strong>{flower.capitalize()}:</strong> {score*100:.2f}%</p>", unsafe_allow_html=True)
        st.progress(int(score * 100))

else:
    st.info("Please upload a JPG, JPEG, or PNG image file (Max 200MB).")
