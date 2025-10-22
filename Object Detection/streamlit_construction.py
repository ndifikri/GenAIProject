import cv2
import numpy as np
import streamlit as st
import supervision as sv
from ultralytics import YOLO

def detector_pipeline(image_bytes):
  np_array = np.frombuffer(image_bytes, np.uint8)
  image = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
  results = model(image, verbose=False)[0]
  detections = sv.Detections.from_ultralytics(results).with_nms()
  box_annotator = sv.BoxAnnotator()
  label_annotator = sv.LabelAnnotator()
  annotated_image = image.copy()
  annotated_image = box_annotator.annotate(scene=annotated_image, detections=detections)
  annotated_image = label_annotator.annotate(scene=annotated_image, detections=detections)
  annotated_image_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
  classcounts = {}
  for c in detections.data.get("class_name"):
    if c not in classcounts:
      classcounts[str(c)] = 0
    classcounts[c] += 1
  for c in classcounts:
    # st.write(f"{c}: {classcounts[c]}")
    return annotated_image_rgb, classcounts


st.title("Construction Safety Equipment Checking")

selected_model = st.selectbox("Select Usecase", ("Construction Equipment", "Vehicle"))
if selected_model == "Construction Equipment":
    model = YOLO('./Object Detection/best_construction.pt')
else:
    model = YOLO('./Object Detection/best_vehicle2.pt')

uploaded_file = st.file_uploader("Upload Image", accept_multiple_files=False, type=["jpg", "jpeg", "png", "webp"])
if uploaded_file is not None:
    # To read file as bytes:
    bytes_data = uploaded_file.getvalue()
    if st.button("Detect Object"):
        annotated_image_rgb, classcounts = detector_pipeline(bytes_data)
        st.image(annotated_image_rgb)
        st.write(classcounts)