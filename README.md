# climbing_tracking

Streamlit app to track climbers (YOLO + MediaPipe) and detect hold usage
(small network trained on annotated sessions).

## Project layout

```
climbing_tracking/
├── scripts/            # streamlit app + all processing code
│   ├── main.py         # entry point (streamlit run scripts/main.py)
│   ├── pages/          # app pages
│   └── scripts_*.py    # tracking / training / mask modules
├── configs/            # session config JSONs (uploaded in the app pages)
├── models/             # all model files (gitignored)
│   ├── yolov8x-pose.pt
│   ├── mediapipe_models/   # pose_landmarker_heavy.task
│   └── dwpose_models/      # ONNX models for the DWPose page
├── data/               # videos, tracking, annotations, holds (gitignored)
├── requirements.txt
└── Dockerfile
```

## Path conventions

- Config files are **uploaded** in the app; the paths inside them
  (`./data/...`) are resolved **relative to the directory where
  Streamlit is launched** (the project root, or `/app` in Docker) — NOT
  relative to the config file location.
- Model paths in the code are resolved relative to the project root
  (independent of the launch directory).

## Run

```bash
pip install -r requirements.txt
streamlit run scripts/main.py
```

Or with Docker (see README.txt): paths inside the container are
relative to `/app`, so all config paths must look like
`./data/videos/myvideo.mp4`.
