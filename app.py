import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

# Load model
model = load_model("bird_model.keras")

# Load labels
with open("labels.txt") as f:
    class_names = [line.strip() for line in f.readlines()]

# Title
st.title("🐦Nepal Endangered Bird Identifier")
st.write("Upload an image and the AI will identify the bird")

# Upload image
uploaded_file = st.file_uploader("Choose an image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    st.write("🔍 Predicting...")

    # Preprocess image
    img = image.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    predictions = model.predict(img_array)[0]
    top3 = predictions.argsort()[-3:][::-1]

    st.subheader("Top Predictions")
    for i, idx in enumerate(top3):
        st.write(f"{i+1}. {class_names[idx]} ({predictions[idx]*100:.2f}%)")
