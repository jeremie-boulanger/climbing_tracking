# Page Title: Tracking

import streamlit as st
import json
import os
import pickle
from scripts_tracking_auto import *
from scripts_segmentation import *

# --------------------------------------------------
# Page config (must be FIRST!)
# --------------------------------------------------
st.set_page_config(layout="wide")
st.title("Automatic Tracking App")

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
    "mask": "",
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
col_yolo, col_video, col_auto = st.columns(3)

# --------------------------------------------------
# YOLO COLUMN
# --------------------------------------------------
with col_yolo:

    st.header("YOLO tracking automatic : track every person in the video")
    show_example = st.toggle("Show JSON example", key="toggle_json_yolo")

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
    
    submitted_media_all = st.toggle("Generate Media for all sessions", key="toggle_media_yolo")

    # ---------------- JSON uploader ----------------
    uploaded_file = st.file_uploader("Upload session JSON", type=["json"])

    if uploaded_file:
        try:
            data = json.load(uploaded_file)
            sessions = data.get("session", [])

            if not sessions:
                st.warning("No session found.")
            else:
                selected = st.selectbox("Choose a session:", [s["title"] for s in sessions], key="select_yolo")
                s = next(x for x in sessions if x["title"] == selected)

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

                submitted_yolo_all = st.button("Run YOLO Tracking for all sessions", key="btn_yolo_all")
                if submitted_yolo_all:
                    with st.spinner("Tracking all sessions"):
                        text_session_track = st.empty()
                        for i,s in enumerate(sessions):
                            with text_session_track:
                                t = s.get("title","")
                                st.write(f"Processing: {t}")
                                current_video = s.get("video", "")
                                current_tracking_yolo = s.get("tracking_yolo", "")
                                current_image_tracking = s.get("image_tracking", "")
                                h = s.get("image_tracking", "")
                                current_heatmap = h[:-4] + "_heatmap" + h[-4:]
                                if not os.path.isfile(s.get("tracking_yolo","")):
                                    detections, people, heatmap_acc = yolo_tracking(
                                        current_video,
                                        current_tracking_yolo,
                                        title=t,
                                        out_image_name=current_image_tracking,
                                        heatmap_image_name=current_heatmap,
                                        model_name='yolov8x-pose.pt',
                                        display=False,
                                        stream=True
                                    )
                                    mediapipe_tracking(
                                        s.get("video", ""),
                                        s.get("tracking_yolo", ""),
                                        s.get("tracking_mediapipe", ""),
                                    )
                                    extract_tracking(
                                        s.get("tracking_mediapipe", ""),
                                        s.get("tracking_pickle", ""),
                                        s.get("tracking_xlsx", ""),
                                        s.get("frequency cut-off", 0.5),
                                        s.get("pixel size X (m/p)", 0.0),
                                        s.get("pixel size Y (m/p)", 0.0),
                                    )
                        st.success("YOLO on all sessions complete.")
                
                    if submitted_media_all:
                        with st.spinner("Run media for all sessions"):
                            
                            for i, s in enumerate(sessions):
                                t = s.get("title", "")
                                current_video = s.get("video", "")
                                current_wall = s.get("wall image", "")
                                current_video_out = s.get("video_tracking", "")
                                current_pickle = s.get("tracking_pickle", "")

                                # Wall Image
                                if current_video and current_wall:
                                    if not os.path.isfile(current_wall):
                                        extract_median_image(current_video, current_wall)

                                # Video
                                if current_video and current_video_out and current_pickle:
                                    if not os.path.isfile(current_video_out) and os.path.isfile(current_pickle):
                                        write_on_video(current_video,current_video_out,current_pickle)
                                        
                            st.success("Media and YOLO for all sessions complete.")
                        


        except Exception as e:
            st.error(f"JSON error: {e}")
#

    # ---------------- CLEAR FIELDS ----------------
    if st.button("Clear All Fields"):
        for key, value in defaults.items():
            st.session_state[key] = value
        st.success("Fields cleared.")

    # ---------------- Preview YOLO Image & Heatmap ----------------

with col_video:
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
                width = 'stretch'
            )

        # Heatmap
        if st.session_state.get("heatmap_tracking") and os.path.isfile(st.session_state.heatmap_tracking):
            heatmap_img = cv2.imread(st.session_state.heatmap_tracking)
            heatmap_rgb = cv2.cvtColor(heatmap_img, cv2.COLOR_BGR2RGB)
            st.image(
                heatmap_rgb,
                caption="Accumulated Heatmap",
                width = 'stretch'
            )

        if os.path.isfile(st.session_state.wall_image):
            st.image(
                st.session_state.wall_image,
                caption=f"Climbing wall: {st.session_state.title}",
                width="stretch",
            )
            
        if os.path.isfile(st.session_state.video_tracking):
            st.video(st.session_state.video_tracking)



with col_auto:

    st.header("YOLO tracking automatic : track every person in the video with video segmentation")
    show_example = st.toggle("Show JSON example", key="toggle_json_auto")

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
                "mask": "./data/holds/",
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
                "mask": "./data/holds/",
                "video_tracking": "./data/tracking/Seq22_tracking.mp4",
                "pixel size X (m/p)": 0.01,
                "pixel size Y (m/p)": 0.01,
                "frequency cut-off": 0.1,
                "delta t": 0.02,
            }]
        })
    
    submitted_media_all = st.toggle("Generate Media for all sessions", key="toggle_media_auto")

    # ---------------- JSON uploader ----------------
    uploaded_file = st.file_uploader("Upload session JSON", type=["json"], key="uploader_auto")

    if uploaded_file:
        try:
            data = json.load(uploaded_file)
            sessions = data.get("session", [])

            if not sessions:
                st.warning("No session found.")
            else:
                selected = st.selectbox("Choose a session:", [s["title"] for s in sessions], key="select_auto")
                s = next(x for x in sessions if x["title"] == selected)

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

                submitted_yolo_all = st.button("Run YOLO Tracking for all sessions", key="btn_auto_all")
                if submitted_yolo_all:
                    lst_essaie = []
                    with st.spinner("Tracking all sessions"):
                        text_session_track = st.empty()
                        for i,s in enumerate(sessions):
                            with text_session_track:
                                t = s.get("title","")
                                current_video = s.get("video", "")
                                current_tracking_yolo = s.get("tracking_yolo", "")
                                current_image_tracking = s.get("image_tracking", "")
                                h = s.get("image_tracking", "")
                                current_heatmap = h[:-4] + "_heatmap" + h[-4:]
                                if True or not os.path.isfile(s.get("tracking_yolo","")):
                                    st.write(f"Processing: {t}, tracking yolo")
                                    detections, people, heatmap_acc = yolo_tracking(
                                        current_video,
                                        current_tracking_yolo,
                                        title=t,
                                        out_image_name=current_image_tracking,
                                        heatmap_image_name=current_heatmap,
                                        model_name='yolov8x-pose.pt',
                                        display=False,
                                        stream=True
                                    )
                                if True or not os.path.isfile(s.get("tracking_mediapipe","")) :
                                    st.write(f"Processing: {t}, tracking mediapipe")
                                    mediapipe_tracking(
                                        s.get("video", ""),
                                        s.get("tracking_yolo", ""),
                                        s.get("tracking_mediapipe", "")
                                    )
                                st.write(f"Processing: {t}, extract tracking")
                                extract_tracking(
                                    s.get("tracking_mediapipe", ""),
                                    s.get("tracking_pickle", ""),
                                    s.get("tracking_xlsx", ""),
                                    s.get("frequency cut-off", 0.5),
                                    s.get("pixel size X (m/p)", 0.0),
                                    s.get("pixel size Y (m/p)", 0.0)
                                )
                                st.write(f"Processing: {t}, segmentation")
                                lst_essaie += segmentation(
                                    s.get("video", ""), 
                                    s.get("tracking_pickle", ""),
                                    s.get("tracking_xlsx", ""),
                                    s.get("mask", "")
                                )
                                print(lst_essaie)

                        st.success("YOLO on all sessions complete.")
                
                    if submitted_media_all:
                        with st.spinner("Run media for all sessions"):
                            text = st.empty()

                            for i, s in enumerate(sessions):
                                t = s.get("title", "")
                                current_video = s.get("video", "")
                                current_wall = s.get("wall image", "")
                                current_video_out = s.get("video_tracking", "")
                                current_pickle = s.get("tracking_pickle", "")

                                # Wall Image
                                if current_video and current_wall:
                                    if not os.path.isfile(current_wall):
                                        st.write(f"Processing: {t}, wall image")
                                        extract_median_image(current_video, current_wall)

                                # Video
                                if current_video and current_video_out and current_pickle:
                                    if not os.path.isfile(current_video_out) and os.path.isfile(current_pickle):
                                        st.write(f"Processing: {t}, video")
                                        write_on_video(current_video,current_video_out,current_pickle)

                            for i, s in enumerate(lst_essaie):
                                t = s[0][:-4]
                                current_video = s[0]
                                current_video_out = s[1][:-7] + "_tracking.mp4"
                                current_pickle = s[1]

                                # Video
                                if current_video and current_video_out and current_pickle:
                                    if not os.path.isfile(current_video_out) and os.path.isfile(current_pickle):
                                        st.write(f"Processing: {t}, video")
                                        write_on_video(current_video,current_video_out,current_pickle)
                            
                            st.success("Media and YOLO for all sessions complete.")

        except Exception as e:
            st.error(f"JSON error: {e}")
