import streamlit as st
import tensorflow as tf
import numpy as np
import json

from PIL import Image
from tensorflow.keras.preprocessing.image import img_to_array

# Page title
st.set_page_config(
    page_title="Road Damage Detection",
    layout="centered"
)

st.title("Road Damage Detection System")

st.write(
    "Upload a road image to detect damage category."
)

# Load model
model = tf.keras.models.load_model(
    "road_damage_model.h5"
)

# Load label mapping
with open(
    "label_mapping.json",
    "r"
) as f:
    label_mapping = json.load(f)

# Reverse label mapping
class_names = {
    value:key
    for key, value
    in label_mapping.items()
}

# Upload image
uploaded_file = st.file_uploader(
    "Upload a Road Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(
        uploaded_file
    )

    # Show image preview
    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )

    # Resize image
    image = image.resize(
        (128,128)
    )

    # Convert to array
    image_array = img_to_array(
        image
    )

    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    # Normalize
    image_array = image_array / 255.0

    # Prediction
    prediction = model.predict(
        image_array
    )

    predicted_index = np.argmax(
        prediction
    )

    confidence = np.max(
        prediction
    ) * 100

    predicted_class = class_names[
        predicted_index
    ]

    # Show result
    st.success(
        f"Prediction: {predicted_class}"
    )

    st.info(
        f"Confidence Score: {confidence:.2f}%"
    )
