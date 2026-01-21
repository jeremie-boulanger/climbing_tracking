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
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY requirements.txt .

# Set temp directory on a bigger partition
ENV TMPDIR=/home/merry251/docker_tracking/tmp
RUN mkdir -p $TMPDIR
RUN pip install --no-cache-dir -r requirements.txt

#RUN pip install --no-cache-dir -r requirements.txt

# Copy YOLO model and code
COPY yolov8x-pose.pt /app/yolov8x-pose.pt
COPY scripts/ /app/scripts/
#COPY config.yaml /app/config.yaml

# Prepare folder for Mediapipe models
#COPY mediapipe_models/pose_landmark_heavy.tflite /app/.mediapipe/
RUN mkdir -p /app/.mediapipe && chmod -R a+rwx /app/.mediapipe

# Create non-root user
ARG LOCAL_UID=1000
ARG LOCAL_GID=1000
RUN groupadd -g ${LOCAL_GID} appgroup && \
    useradd -m -u ${LOCAL_UID} -g ${LOCAL_GID} appuser

# Make a writable folder for input/output videos
RUN mkdir -p /app/data && chown -R ${LOCAL_UID}:${LOCAL_GID} /app/data

# Switch to non-root user
USER appuser

# Environment variables
ENV MEDIAPIPE_MODEL_DIR=/app/.mediapipe
#ENV MEDIAPIPE_DISABLE_GPU=1   # optional if no GPU

# Preload Mediapipe Pose as appuser (downloads cached in /app/.mediapipe)
RUN python3 -c "from ultralytics import YOLO; model = YOLO('yolov8x-pose.pt')"
#RUN python3 -c "import mediapipe as mp; mp.solutions.pose.Pose(static_image_mode=True)"
#RUN python3 -c "from mediapipe.python import solutions as mp_solutions; mp_solutions.pose.Pose(static_image_mode=True)"
RUN python3 - <<'EOF'
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.vision import RunningMode

BaseOptions = python.BaseOptions
PoseLandmarker = python.vision.PoseLandmarker
PoseLandmarkerOptions = python.vision.PoseLandmarkerOptions

options = PoseLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path="/app/.mediapipe/pose_landmarker_lite.task"
    ),
    running_mode=RunningMode.IMAGE
)

with PoseLandmarker.create_from_options(options) as landmarker:
    print("✅ Mediapipe PoseLandmarker ready (Python 3.11)")
EOF




# Command to run Streamlit
CMD ["streamlit", "run", "scripts/main.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]

