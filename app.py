import streamlit as st
import cv2
import numpy as np
import tempfile
from ultralytics import YOLO
import torch
import time

# Define COCO dataset classes
CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
    'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
    'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
    'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'fire', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
    'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
    'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush','gun'
]

# Configure Streamlit page settings
st.set_page_config(initial_sidebar_state='expanded')
def detect_objects(image, model, classes, conf):
    results = model(image, conf=conf, classes=classes)
    annotated_image = results[0].plot()
    return annotated_image, results[0].boxes.cls.tolist()

def is_cuda_available():
    return torch.cuda.is_available()

def main():
    # Initialize session state variables for metrics tracking
    if 'frame_rate' not in st.session_state:
        st.session_state.frame_rate = 0
    if 'tracked_objects' not in st.session_state:
        st.session_state.tracked_objects = 0
    if 'detected_classes' not in st.session_state:
        st.session_state.detected_classes = 0

    # Set up main page title and divider
    st.markdown("<h1 style='text-align: center;'>Vision X - Real Time Object Counter</h1>", unsafe_allow_html=True)
    st.markdown("---")

    # Configure sidebar settings
    st.sidebar.title("⚙️ Settings")
    cuda_available = is_cuda_available()

    # Add user configuration options
    conf_threshold = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.3, 0.1)
    enable_gpu = st.sidebar.checkbox("🤖 Enable GPU", value=False, disabled=not cuda_available)

    # Add input source selection options
    use_webcam = st.sidebar.button("Use Webcam")
    selected_classes = st.sidebar.multiselect("Select Category", CLASSES, default=['person'])

    # Display CUDA availability warning if needed
    if not cuda_available:
        st.sidebar.warning("CUDA is not available. GPU acceleration is disabled.")

    # Add video upload option
    uploaded_video = st.sidebar.file_uploader("Upload Video", type=['mp4', 'avi', 'mov'])

    st.sidebar.markdown('''
     by TEAM CHOICE_MATRIX ❤️
    ''')

    # Convert selected class names to their corresponding indices
    class_indices = [CLASSES.index(cls) for cls in selected_classes]

    # Initialize YOLO model with appropriate device (GPU/CPU)
    model = YOLO('yolov8n.pt')
    model.to('cuda' if enable_gpu and cuda_available else 'cpu')

    # Create video display container
    video_placeholder = st.empty()

    # Create metrics display layout
    kpi_col1, kpi_col2, kpi_col3 = st.columns(3)
    tracked_objects_metric = kpi_col1.empty()
    frame_rate_metric = kpi_col2.empty()
    classes_metric = kpi_col3.empty()

    # Initialize metrics with default values
    tracked_objects_metric.metric("Tracked Objects", "0")
    frame_rate_metric.metric("Frame Rate", "0.00 FPS")
    classes_metric.metric("Classes", "0")

    # Create placeholder for object count table
    object_count_placeholder = st.empty()

    # Handle webcam input
    if use_webcam:
        cap = cv2.VideoCapture(0)
        frame_count = 0
        start_time = time.time()

        while True:
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to capture frame from webcam.")
                break

            # Process and display frame
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            annotated_frame, detected_classes = detect_objects(frame, model, class_indices, conf_threshold)
            video_placeholder.image(annotated_frame, channels="RGB")

            # Update performance metrics
            frame_count += 1
            elapsed_time = time.time() - start_time
            if elapsed_time >= 1.0:  # Update metrics every second
                frame_rate = frame_count / elapsed_time
                tracked_objects_metric.metric("Tracked Objects", str(len(detected_classes)))
                frame_rate_metric.metric("Frame Rate", f"{frame_rate:.2f} FPS")
                unique_classes = len(np.unique(detected_classes))
                classes_metric.metric("Classes", str(unique_classes))
                frame_count = 0
                start_time = time.time()

            # Update and display object count table
            unique_classes, counts = np.unique(detected_classes, return_counts=True)
            object_data = [{"Class": CLASSES[int(cls)], "Count": count} for cls, count in zip(unique_classes, counts)]
            object_count_placeholder.markdown(
                "<table class='detected-object-table'>" +
                "<tr><th>Class</th><th>Count</th></tr>" +
                "".join([f"<tr><td>{item['Class']}</td><td>{item['Count']}</td></tr>" for item in object_data]) +
                "</table>",
                unsafe_allow_html=True
            )

        cap.release()

    # Handle uploaded video input
    elif uploaded_video is not None:
        # Create temporary file for video processing
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_video.read())

        vf = cv2.VideoCapture(tfile.name)
        frame_count = 0
        start_time = time.time()

        while vf.isOpened():
            ret, frame = vf.read()
            if not ret:
                break

            # Process and display frame
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            annotated_frame, detected_classes = detect_objects(frame, model, class_indices, conf_threshold)
            video_placeholder.image(annotated_frame, channels="RGB")

            # Update performance metrics
            frame_count += 1
            elapsed_time = time.time() - start_time
            if elapsed_time >= 0.5:  # Update metrics every half second
                frame_rate = frame_count / elapsed_time
                tracked_objects_metric.metric("Tracked Objects", str(len(detected_classes)))
                frame_rate_metric.metric("Frame Rate", f"{frame_rate:.2f} FPS")
                unique_classes = len(np.unique(detected_classes))
                classes_metric.metric("Classes", str(unique_classes))
                frame_count = 0
                start_time = time.time()

            # Update and display object count table
            unique_classes, counts = np.unique(detected_classes, return_counts=True)
            object_data = [{"Class": CLASSES[int(cls)], "Count": count} for cls, count in zip(unique_classes, counts)]
            object_count_placeholder.markdown(
                "<table class='detected-object-table'>" +
                "<tr><th>Class</th><th>Count</th></tr>" +
                "".join([f"<tr><td>{item['Class']}</td><td>{item['Count']}</td></tr>" for item in object_data]) +
                "</table>",
                unsafe_allow_html=True
            )

        vf.release()

if __name__ == "__main__":
    main()
