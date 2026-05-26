import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="Road Damage Detection",
    page_icon="🛣️",
    layout="wide"
)

# -----------------------------
# LOAD MODEL
# -----------------------------
@st.cache_resource
def load_cnn_model():
    model = load_model("road_damage_model.h5")
    return model

model = load_cnn_model()

# Class labels
class_names = ["Pothole", "Crack", "Normal Road"]

# -----------------------------
# SECTION 1 — HEADER
# -----------------------------
st.title("AI-Based Road Damage Detection System")
st.subheader("Smart City Infrastructure Monitoring using CNN")

st.markdown("---")

# -----------------------------
# SECTION 2 — ABOUT PROJECT
# -----------------------------
st.header("About the Project")

st.write("""
Road monitoring is important for ensuring transportation safety,
reducing accidents, and maintaining infrastructure quality.
Poor road conditions such as potholes and cracks can damage vehicles
and create dangerous driving situations.

This system uses **Convolutional Neural Networks (CNN)**, a deep learning
technique specialized for **computer vision and image recognition**,
to automatically detect road damage from images.

### Industry Applications
- Smart City Infrastructure Monitoring
- Highway Maintenance Systems
- Municipal Road Inspection
- AI-based Transportation Safety
""")

st.markdown("---")

# -----------------------------
# SECTION 3 — UPLOAD AREA
# -----------------------------
st.header("Upload Road Image")

uploaded_file = st.file_uploader(
    "Choose a road image",
    type=["jpg", "jpeg", "png"]
)

# -----------------------------
# IMAGE PROCESSING FUNCTION
# -----------------------------
def preprocess_image(image):
    image = image.resize((224, 224))  # change if your model input differs
    image = np.array(image)

    if image.shape[-1] == 4:
        image = image[:, :, :3]

    image = image / 255.0
    image = np.expand_dims(image, axis=0)

    return image


if uploaded_file is not None:

    image = Image.open(uploaded_file)

    # -----------------------------
    # SECTION 4 — IMAGE PREVIEW
    # -----------------------------
    st.header("Uploaded Image Preview")

    st.image(
        image,
        caption="Uploaded Road Image",
        use_container_width=True
    )

    # -----------------------------
    # PREDICTION
    # -----------------------------
    processed_image = preprocess_image(image)

    prediction = model.predict(processed_image)

    predicted_index = np.argmax(prediction)
    predicted_label = class_names[predicted_index]
    confidence = float(np.max(prediction)) * 100

    # Severity logic
    if predicted_label == "Pothole":
        severity = "High"
    elif predicted_label == "Crack":
        severity = "Medium"
    else:
        severity = "Low"

    # -----------------------------
    # SECTION 5 — PREDICTION AREA
    # -----------------------------
    st.markdown("---")
    st.header("Prediction Results")

    st.success(f"Prediction: {predicted_label}")
    st.info(f"Confidence: {confidence:.2f}%")
    st.warning(f"Severity: {severity}")

    # -----------------------------
    # SECTION 6 — VISUALIZATION
    # -----------------------------
    st.markdown("---")
    st.header("Visualization Area")

    probabilities = prediction[0] * 100

    fig, ax = plt.subplots(figsize=(8, 4))

    ax.bar(class_names, probabilities)

    ax.set_ylabel("Confidence (%)")
    ax.set_xlabel("Damage Type")
    ax.set_title("Class Confidence Graph")

    st.pyplot(fig)

    # -----------------------------
    # SECTION 7 — RECOMMENDATIONS
    # -----------------------------
    st.markdown("---")
    st.header("Recommendations")

    if predicted_label == "Pothole":
        st.error("""
        Immediate maintenance recommended.
        High-risk road condition detected.
        Drivers should proceed with caution.
        """)

    elif predicted_label == "Crack":
        st.warning("""
        Road repair should be scheduled soon.
        Moderate safety risk detected.
        Continuous monitoring is recommended.
        """)

    else:
        st.success("""
        No urgent maintenance required.
        Road condition appears safe.
        Continue regular monitoring.
        """)
