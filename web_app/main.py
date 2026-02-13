import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2
import tempfile

# Page config
st.set_page_config(page_title="PV Defect Detection", layout="centered")

st.title("🔍 PV Panel Defect Detection")
st.write("Upload an image to detect **Broken / Clean / Dirty** panels")

# Load model (cached so it doesn't reload every time)
@st.cache_resource
def load_model():
    return YOLO(r"C:/Users/asus/runs/detect/train9/weights/best.pt")

model = load_model()

# Confidence slider
conf = st.slider("Confidence Threshold", 0.1, 1.0, 0.25, 0.05)

# Image uploader
uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    # Load image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Save temporarily (YOLO needs a file path)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        image.save(tmp.name)
        temp_path = tmp.name

    if st.button("Run Detection"):
        with st.spinner("Running YOLOv8 inference..."):
            results = model.predict(
                temp_path,
                conf=conf,
                device="cpu"  # change to 0 if you want GPU
            )

            # Plot result
            result_img = results[0].plot()
            result_img = cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR)

            st.image(result_img, caption="Prediction Result", use_column_width=True)

            # Show detections summary
            boxes = results[0].boxes
            if boxes is not None and len(boxes) > 0:
                st.success(f"Detected {len(boxes)} object(s)")
            else:
                st.warning("No objects detected")
