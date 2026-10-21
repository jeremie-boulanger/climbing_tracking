FROM nvidia/cuda:12.2.0-base-ubuntu22.04
# Set environment variables
ENV TZ=Europe/London
ENV DEBIAN_FRONTEND=noninteractive

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3-pip \
    libgl1 \
    libglib2.0-0 \
    ffmpeg \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# GPU build: replace the CPU onnxruntime with the CUDA build (falls back
# to CPU execution automatically when no GPU is available)
RUN pip install --no-cache-dir onnxruntime-gpu

# Copy YOLO model and code
COPY models/yolov8x-pose.pt /app/models/yolov8x-pose.pt
COPY scripts/ /app/scripts/
#COPY config.yaml /app/config.yaml

# Prepare folder for Mediapipe models (code reads models/mediapipe_models/)
RUN mkdir -p /app/models/mediapipe_models && chmod -R a+rwx /app/models/mediapipe_models

# Create non-root user
ARG LOCAL_UID=1000
ARG LOCAL_GID=1000
RUN groupadd -g ${LOCAL_GID} appgroup && \
    useradd -m -u ${LOCAL_UID} -g ${LOCAL_GID} appuser

# Make a writable folder for input/output videos
RUN mkdir -p /app/data && chown -R ${LOCAL_UID}:${LOCAL_GID} /app/data

# Writable folder for the DWPose ONNX models (downloaded at build)
RUN mkdir -p /app/models/dwpose_models && chown -R ${LOCAL_UID}:${LOCAL_GID} /app/models/dwpose_models

# Switch to non-root user
USER appuser

# Environment variables
ENV MEDIAPIPE_MODEL_DIR=/app/models/mediapipe_models
#ENV MEDIAPIPE_DISABLE_GPU=1   # optional if no GPU

# Preload Mediapipe Pose as appuser (downloads cached in /app/.mediapipe)
RUN python3 -c "from ultralytics import YOLO; model = YOLO('models/yolov8x-pose.pt')"
#RUN python3 -c "import mediapipe as mp; mp.solutions.pose.Pose(static_image_mode=True)"
#RUN python3 -c "from mediapipe.python import solutions as mp_solutions; mp_solutions.pose.Pose(static_image_mode=True)"

RUN wget -O /app/models/mediapipe_models/pose_landmarker_heavy.task https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task

RUN python3 - <<'EOF'
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.vision import RunningMode

BaseOptions = python.BaseOptions
PoseLandmarker = python.vision.PoseLandmarker
PoseLandmarkerOptions = python.vision.PoseLandmarkerOptions

options = PoseLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path="/app/models/mediapipe_models/pose_landmarker_heavy.task"
    ),
    running_mode=RunningMode.IMAGE
)

with PoseLandmarker.create_from_options(options) as landmarker:
    print("✅ Mediapipe PoseLandmarker ready (Python 3.11)")
EOF

# Pre-download the DWPose ONNX models (yolox_l + dw-ll_ucoco_384)
RUN python3 -c "import sys; sys.path.insert(0, 'scripts'); from scripts_tracking_dwpose import DWPose; DWPose().load(); print('DWPose ONNX models ready')"




# Command to run Streamlit
CMD ["streamlit", "run", "scripts/main.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]

