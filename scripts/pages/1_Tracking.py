# Page Title: Tracking

import streamlit as st
import json
import os
import pickle
from scripts_tracking import *

# --------------------------------------------------
# Page config (must be FIRST!)
# --------------------------------------------------
st.set_page_config(layout="wide")
st.title("Tracking App")

# --------------------------------------------------
# Session State Initialization
# --------------------------------------------------
defaults = {
    "title": "",
    "video": "",
    "tracking_yolo": "",
    "image_tracking": "",
    "tracking_mediapipe": "",
    "tracking_pickle": "",
    "tracking_xlsx": "",
    "wall_image": "",
    "video_tracking": "",
    "list_climbers_tracking": "",
    "pixel_size_x": 0.0,
    "pixel_size_y": 0.0,
    "freq_cut": 0.5,
    "delta_t": 1,
    "option": "Option 1",
}

for key, value in defaults.items():
    st.session_state.setdefault(key, value)

# --------------------------------------------------
# Columns Layout
# --------------------------------------------------
col_yolo, col_mediapipe, col_video = st.columns(3)

# --------------------------------------------------
# YOLO COLUMN
# --------------------------------------------------
with col_yolo:

    st.header("YOLO tracking: track every person in the video")
    show_example = st.toggle("Show JSON example")

    if show_example:
        st.json({
            "session":[
            {
                "title": "Seq21",
                "video": "./data/videos/Seq21.mp4",
                "tracking_yolo": "./data/tracking/Seq21_tracking_yolo.pickle",
                "image_tracking": "./data/tracking/Seq21_tracking.png",
                "tracking_mediapipe": "./data/tracking/Seq21_tracking_mediapipe.pickle",
                "tracking_pickle": "./data/tracking/Seq21_tracking.pickle",
                "tracking_xlsx": "./data/tracking/Seq21_tracking.xlsx",
                "wall image": "./data/tracking/wall_image21.png",
                "video_tracking": "./data/tracking/Seq21_tracking.mp4",
                "pixel size X (m/p)": 0.01,
                "pixel size Y (m/p)": 0.01,
                "frequency cut-off": 0.1,
                "delta t": 0.02,
            },
            {
                "title": "Seq22",
                "video": "./data/videos/Seq22.mp4",
                "tracking_yolo": "./data/tracking/Seq22_tracking_yolo.pickle",
                "image_tracking": "./data/tracking/Seq22_tracking.png",
                "tracking_mediapipe": "./data/tracking/Seq22_tracking_mediapipe.pickle",
                "tracking_pickle": "./data/tracking/Seq22_tracking.pickle",
                "tracking_xlsx": "./data/tracking/Seq22_tracking.xlsx",
                "wall image": "./data/tracking/wall_image22.png",
                "video_tracking": "./data/tracking/Seq22_tracking.mp4",
                "pixel size X (m/p)": 0.01,
                "pixel size Y (m/p)": 0.01,
                "frequency cut-off": 0.1,
                "delta t": 0.02,
            }]
        })

    # ---------------- JSON uploader ----------------
    uploaded_file = st.file_uploader("Upload session JSON", type=["json"])

    if uploaded_file:
        try:
            data = json.load(uploaded_file)
            sessions = data.get("session", [])

            if not sessions:
                st.warning("No session found.")
            else:
                selected = st.selectbox(
                    "Choose a session:", [s["title"] for s in sessions]
                )
                s = next(x for x in sessions if x["title"] == selected)

                # Update state safely
                st.session_state.title = s.get("title", "")
                st.session_state.video = s.get("video", "")
                st.session_state.tracking_yolo = s.get("tracking_yolo", "")
                st.session_state.image_tracking = s.get("image_tracking", "")
                st.session_state.tracking_mediapipe = s.get("tracking_mediapipe", "")
                st.session_state.tracking_pickle = s.get("tracking_pickle", "")
                st.session_state.tracking_xlsx = s.get("tracking_xlsx", "")
                st.session_state.wall_image = s.get("wall image", "")
                st.session_state.video_tracking = s.get("video_tracking", "")
                st.session_state.pixel_size_x = s.get("pixel size X (m/p)", 0.0)
                st.session_state.pixel_size_y = s.get("pixel size Y (m/p)", 0.0)
                st.session_state.freq_cut = s.get("frequency cut-off", 0.5)
                st.session_state.delta_t = s.get("delta t", 1.0)

                st.success("Fields imported successfully.")

                submitted_yolo_all = st.button("Run YOLO Tracking for all sessions")
                if submitted_yolo_all:
                    with st.spinner("Tracking all sessions"):
                        text_session_track = st.empty()
                        for i,s in enumerate(sessions):
                            with text_session_track:
                                t = s.get("title","")
                                st.write(f"Processing: {t}")
                                if not os.path.isfile(s.get("tracking_yolo","")):
                                    detections, people, heatmap_acc = yolo_tracking(
                                        st.session_state.video,
                                        st.session_state.tracking_yolo,
                                        title=st.session_state.title,
                                        out_image_name=st.session_state.image_tracking,
                                        heatmap_image_name=st.session_state.heatmap_tracking,
                                        model_name='yolov8x-pose.pt',
                                        display=False,
                                        stream=True
                                    )

                        st.success("YOLO on all sessions complete.")
                        


        except Exception as e:
            st.error(f"JSON error: {e}")
    else:
        for key, value in defaults.items():
            st.session_state[key] = value
#

    # ---------------- CLEAR FIELDS ----------------
    if st.button("Clear All Fields"):
        for key, value in defaults.items():
            st.session_state[key] = value
        st.success("Fields cleared.")

# ---------------- YOLO FORM ----------------
with col_yolo:
    placeholder_yolo_user_form = st.empty()
    with placeholder_yolo_user_form:
        with st.form("yolo_form"):
            st.text_input("Title", key="title")
            st.text_input("Video IN", key="video")
            st.text_input("YOLO Tracking OUT", key="tracking_yolo")
            st.text_input("Tracking Image OUT", key="image_tracking")
            st.text_input("Heatmap Image OUT", key="heatmap_tracking", value="./data/tracking/heatmap.png")
            submitted_yolo = st.form_submit_button("Run YOLO Tracking")

    if submitted_yolo:
        with st.spinner("Running YOLO tracking..."):
            # Vérifier si le pickle existe
            if not os.path.isfile(st.session_state.tracking_yolo):
                # Appel de yolo_tracking avec la nouvelle signature
                detections, people, heatmap_acc = yolo_tracking(
                    st.session_state.video,
                    st.session_state.tracking_yolo,
                    title=st.session_state.title,
                    out_image_name=st.session_state.image_tracking,
                    heatmap_image_name=st.session_state.heatmap_tracking,
                    model_name='yolov8x-pose.pt',
                    display=False,
                    stream=True
                )
            else:
                # Charger les données existantes
                with open(st.session_state.tracking_yolo, "rb") as f:
                    data = pickle.load(f)
                detections = data["list_detection"]
                people = data["tracking_person"]
                if os.path.isfile(st.session_state.heatmap_tracking):
                    heatmap_acc = cv2.imread(st.session_state.heatmap_tracking, cv2.IMREAD_UNCHANGED)
                else:
                    heatmap_acc = None

            st.session_state.list_climbers_tracking = ",".join(map(str, people))
            st.success("YOLO complete.")

    # ---------------- Preview YOLO Image & Heatmap ----------------
    placeholder_yolo_image = st.empty()
    with placeholder_yolo_image.container():  # garde tout dans col_yolo
        # Image tracking
        image_width = 400  # largeur en pixels, ajuste comme tu veux

        # Image tracking
        if os.path.isfile(st.session_state.image_tracking):
            img_tracking = cv2.imread(st.session_state.image_tracking)
            img_tracking_rgb = cv2.cvtColor(img_tracking, cv2.COLOR_BGR2RGB)
            st.image(
                img_tracking_rgb,
                caption=f"Tracking Image: {st.session_state.title}",
                width=image_width
            )

        # Heatmap
        if st.session_state.get("heatmap_tracking") and os.path.isfile(st.session_state.heatmap_tracking):
            heatmap_img = cv2.imread(st.session_state.heatmap_tracking)
            heatmap_rgb = cv2.cvtColor(heatmap_img, cv2.COLOR_BGR2RGB)
            st.image(
                heatmap_rgb,
                caption="Accumulated Heatmap",
                width=image_width
            )



# --------------------------------------------------
# MEDIAPIPE COLUMN
# --------------------------------------------------
with col_mediapipe:

    st.header("Mediapipe Pose Tracking")
    placeholder_mediapipe = st.empty()
    with placeholder_mediapipe:
        with st.form("mediapipe_form"):
            st.text_input("Climbers to track, e.g: 2,3,6", key="list_climbers_tracking")
            st.text_input("Mediapipe Tracking OUT", key="tracking_mediapipe")
            submitted_mediapipe = st.form_submit_button("Run Mediapipe")

    if submitted_mediapipe:
        if len(st.session_state.list_climbers_tracking)==0:
            st.warning("Climbers to track is empty, use list of int seperated by commas.")
        if len(st.session_state.list_climbers_tracking)>0: #with placeholders["status_mediapipe"]:
            with st.spinner("Running Mediapipe tracking..."):

                climbers = [int(x) for x in st.session_state.list_climbers_tracking.split(",")]

                # Load previous YOLO tracking
                with open(st.session_state.tracking_yolo, "rb") as f:
                    data = pickle.load(f)
                detections = data["list_detection"]

                prev = data["tracking_person"]
                if climbers != prev:
                    with open(st.session_state.tracking_yolo, "wb") as f:
                        pickle.dump(
                            {"list_detection": detections, "tracking_person": climbers},
                            f,
                            protocol=pickle.HIGHEST_PROTOCOL,
                        )

                mediapipe_tracking(
                    st.session_state.video,
                    st.session_state.tracking_yolo,
                    st.session_state.tracking_mediapipe,
                )

                st.success("Mediapipe complete.")

    # -------------- EXTRACTION ------------------
    st.header("Extract Pose & Export")
    placeholder_extraction = st.empty()
    with placeholder_extraction:
        with st.form("extract_form"):
            st.text_input("Pickle OUT", key="tracking_pickle")
            st.text_input("Excel OUT", key="tracking_xlsx")
            st.number_input("Pixel Size X (m/p)", key="pixel_size_x")
            st.number_input("Pixel Size Y (m/p)", key="pixel_size_y")
            st.number_input("Cut-off Frequency", key="freq_cut")
            submitted_extract = st.form_submit_button("Extract Tracking")

    if submitted_extract:
        if 1:#with placeholders["status_extract"]:
            with st.spinner("Extracting pose data..."):
                extract_tracking(
                    st.session_state.tracking_mediapipe,
                    st.session_state.tracking_pickle,
                    st.session_state.tracking_xlsx,
                    st.session_state.freq_cut,
                    st.session_state.pixel_size_x,
                    st.session_state.pixel_size_y
                )
                st.success("Extraction complete.")

# --------------------------------------------------
# VIDEO COLUMN
# --------------------------------------------------
with col_video:

    st.header("Generate Media")
    placeholder_image_wall_form = st.empty()
    with placeholder_image_wall_form:
        with st.form("media_form"):
            st.text_input("Wall Image OUT", key="wall_image")
            wall_submit = st.form_submit_button("Extract Wall Image")

            st.text_input("Video OUT", key="video_tracking")
            video_submit = st.form_submit_button("Create Tracking Video")

    if wall_submit:
        if 1:#with placeholders["status_wall"]:
            with st.spinner("Extracting wall image..."):
                extract_median_image(st.session_state.video, st.session_state.wall_image)
                st.success("Wall image extracted.")
    
    placeholder_wall_image = st.empty()

    # Preview YOLO image
    if os.path.isfile(st.session_state.wall_image):
        placeholder_wall_image.image(
            st.session_state.wall_image,
            caption=f"Climbing wall: {st.session_state.title}",
            width="stretch",
        )

    if video_submit:
        if 1:#with placeholders["status_video"]:
            with st.spinner("Generating video..."):
                write_on_video(
                    st.session_state.video,
                    st.session_state.video_tracking,
                    st.session_state.tracking_pickle,
                )
                st.success("Video generated.")

    placeholder_video = st.empty()
    if os.path.isfile(st.session_state.video_tracking):
#        video_bytes = open(st.session_state.video_tracking, "rb").read()
        placeholder_video.video(st.session_state.video_tracking)

