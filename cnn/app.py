import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Page Config
st.set_page_config(
    page_title="Cat vs Dog Classifier",
    page_icon="🐶",
    layout="centered"
)

# Load Model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("cnn_model.h5")

model = load_model()

# Title
st.title("Cat vs Dog Classifier")
st.write("Upload an image and the CNN model will predict whether it is a Cat or Dog.")

# Upload Image
uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    # Display Uploaded Image
    image = Image.open(uploaded_file)

    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )

    # Preprocessing
    img = image.resize((128, 128))

    img_array = np.array(img)

    # Convert RGBA to RGB if needed
    if img_array.shape[-1] == 4:
        img_array = img_array[:, :, :3]

    img_array = img_array / 255.0

    img_array = np.expand_dims(
        img_array,
        axis=0
    )

    # Prediction Button
    if st.button("Predict"):

        prediction = model.predict(img_array)

        confidence = float(prediction[0][0])

        if confidence > 0.5:

            st.success("🐶 Dog")

            st.write(
                f"Confidence: {confidence*100:.2f}%"
            )

        else:

            st.success("🐱 Cat")

            st.write(
                f"Confidence: {(1-confidence)*100:.2f}%"
            )