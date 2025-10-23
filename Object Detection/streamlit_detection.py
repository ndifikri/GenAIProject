import numpy as np
import streamlit as st
import supervision as sv
from ultralytics import YOLO
from PIL import Image 
from io import BytesIO

def detector_pipeline_pillow(image_bytes):
    pil_image = Image.open(BytesIO(image_bytes)).convert("RGB") # Pastikan mode RGB
    image_np_rgb = np.array(pil_image)
    results = model(image_np_rgb, verbose=False)[0] 
    detections = sv.Detections.from_ultralytics(results).with_nms()
    box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()
    annotated_image = pil_image.copy()
    annotated_image = box_annotator.annotate(scene=annotated_image, detections=detections)
    annotated_image = label_annotator.annotate(scene=annotated_image, detections=detections)
    annotated_image_np = np.asarray(annotated_image)
    classcounts = {}
    for c in detections.data.get("class_name"):
        if c not in classcounts:
            classcounts[str(c)] = 0
        classcounts[c] += 1
    
    return annotated_image_np, classcounts


# --- Bagian Streamlit Utama ---

st.title("Object Detection")

selected_model = st.selectbox("Select Usecase", ("Construction Equipment", "Vehicle", "Fruit"))
# Asumsi path model Anda sudah benar
if selected_model == "Construction Equipment":
    model = YOLO('./Object Detection/best_construction.pt')
elif selected_model == "Fruit":
    model = YOLO('./Object Detection/best_fruit.pt')
else:
    model = YOLO('./Object Detection/best_vehicle2.pt')

uploaded_file = st.file_uploader("Upload Image", accept_multiple_files=False, type=["jpg", "jpeg", "png", "webp"])
if uploaded_file is not None:
    # Untuk membaca file sebagai bytes:
    bytes_data = uploaded_file.getvalue()
    if st.button("Detect Object"):
        # Panggil fungsi yang dimodifikasi
        annotated_image_rgb, classcounts = detector_pipeline_pillow(bytes_data)
        st.image(annotated_image_rgb)
        st.write(classcounts)