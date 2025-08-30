import streamlit as st
from ultralytics import YOLO
import cv2
import os
import numpy as np
import logging
import datetime

# Generate a unique log file name for each run
run_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = f"logs/detection_log_{run_timestamp}.txt"
os.makedirs(os.path.dirname(log_file), exist_ok=True)
logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s - %(message)s')

st.title("Motorbike Detection")

# Model path (relative to the app.py location)
model_path = "models/best_motorbike.pt"

# Load the local model
if os.path.exists(model_path):
    print(f"Loading local model from {model_path}")
    model = YOLO(model_path)
else:
    st.error(f"Local model not found at {model_path}. Please ensure best_motorbike.pt is in the models/ folder.")
    st.stop()

# Function to log detection results
def log_detection(image_source, num_detections):
    log_message = f"Processed {image_source} - {num_detections} motorbikes detected"
    logging.info(log_message)
    print(log_message)

# Function to save annotated output
def save_annotated_output(annotated_img, source_name, frame_num=None, video_writer=None, save_as_frames=False, fps=None):
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(source_name)[0]
    
    if video_writer is not None and not save_as_frames:
        video_writer.write(annotated_img)
    elif save_as_frames and frame_num is not None:
        if fps is None or frame_num % max(1, int(30 / fps)) == 0:
            filename = f"{base_name}_frame_{frame_num}.jpg"
            output_path = os.path.join(output_dir, filename)
            cv2.imwrite(output_path, annotated_img)
            st.success(f"Annotated frame saved to: {output_path}")
            log_detection(f"Annotated frame {filename}", 0)
    else:
        if frame_num is not None:
            filename = f"{base_name}_frame_{frame_num}.jpg"
        else:
            filename = f"{base_name}_annotated.jpg"
        output_path = os.path.join(output_dir, filename)
        cv2.imwrite(output_path, annotated_img)
        st.success(f"Annotated output saved to: {output_path}")
        log_detection(f"Annotated output {filename}", 0)

# Input options (Webcam disabled for cloud)
input_type = st.radio("Select input type:", ("Image", "Video"))

if input_type == "Image":
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        file_path = os.path.join("data", uploaded_file.name)
        os.makedirs("data", exist_ok=True)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        if uploaded_file.name.endswith((".jpg", ".jpeg", ".png")):
            st.image(file_path, caption="Uploaded Image", width='stretch')
            try:
                img = cv2.imread(file_path)
                if img is None:
                    st.error(f"Failed to load image: {file_path}")
                    raise ValueError("Image loading failed")

                results = model(img)[0]
                annotated_img = img.copy()
                detections = []
                for det in results.boxes:
                    if det.cls == 0:  # motorbike class
                        x, y, w, h = det.xywh[0].numpy()
                        cv2.rectangle(annotated_img, (int(x - w/2), int(y - h/2)), (int(x + w/2), int(y + h/2)), (0, 255, 0), 2)
                        detections.append("Motorbike Detected")

                if detections:
                    st.image(annotated_img, caption="Annotated Image with Motorbike Detections", width='stretch', channels="BGR")
                    save_annotated_output(annotated_img, uploaded_file.name)
                    log_detection(uploaded_file.name, len(detections))
                else:
                    st.write("No motorbikes detected.")
                    save_annotated_output(annotated_img, uploaded_file.name)
                    log_detection(uploaded_file.name, 0)
            except ValueError as e:
                st.error(f"Detection error: {e}")
        else:
            st.write("Invalid image format. Use .jpg, .jpeg, or .png.")

elif input_type == "Video":
    uploaded_video = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])
    if uploaded_video is not None:
        video_path = os.path.join("data", uploaded_video.name)
        os.makedirs("data", exist_ok=True)
        with open(video_path, "wb") as f:
            f.write(uploaded_video.getbuffer())

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            st.error(f"Error: Could not open video {video_path}")
        else:
            frame_placeholder = st.empty()
            frame_count = 0
            save_option = st.radio("Save as:", ("Video", "Frames"), key="video_save_option")
            fps = st.number_input("Frames per second (for frames option)", min_value=1, max_value=30, value=5, key="fps_input") if save_option == "Frames" else None
            
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            fps_video = cap.get(cv2.CAP_PROP_FPS) if cap.get(cv2.CAP_PROP_FPS) > 0 else 30.0
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            output_video_path = f"output/{uploaded_video.name.split('.')[0]}_annotated_{run_timestamp}.mp4"
            video_writer = cv2.VideoWriter(output_video_path, fourcc, fps_video, (frame_width, frame_height)) if save_option == "Video" else None
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    st.write("Video processing completed.")
                    break
                results = model(frame)[0]
                annotated_frame = frame.copy()
                detections = []
                for det in results.boxes:
                    if det.cls == 0:  # motorbike class
                        x, y, w, h = det.xywh[0].numpy()
                        cv2.rectangle(annotated_frame, (int(x - w/2), int(y - h/2)), (int(x + w/2), int(y + h/2)), (0, 255, 0), 2)
                        detections.append("Motorbike Detected")

                if detections:
                    frame_placeholder.image(annotated_frame, caption="Video - Motorbike Detections", width='stretch', channels="BGR")
                    save_annotated_output(annotated_frame, uploaded_video.name, frame_count, video_writer, save_option == "Frames", fps)
                    log_detection(f"Video Frame {frame_count}", len(detections))
                else:
                    frame_placeholder.image(annotated_frame, caption="Video - No Motorbikes Detected", width='stretch', channels="BGR")
                    save_annotated_output(annotated_frame, uploaded_video.name, frame_count, video_writer, save_option == "Frames", fps)
                    log_detection(f"Video Frame {frame_count}", 0)

                frame_count += 1

            cap.release()
            if video_writer is not None:
                video_writer.release()
                st.success(f"Annotated video saved to: {output_video_path}")
            elif save_option == "Frames":
                st.success("Annotated frames saved to output directory.")

st.write("Select an input type and upload/process to detect motorbikes!")