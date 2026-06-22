import streamlit as st
import json
import os
import numpy as np
from scripts_mask import *
from scripts_segmentation import * 
from scripts_tracking import *
import shutil
import ast

# --------------------------------------------------
# Page config (must be FIRST!)
# --------------------------------------------------
st.set_page_config(layout="wide")
st.title("Mask & Segmentation")

# --------------------------------------------------
# Session State Initialization
# --------------------------------------------------
defaults = {
    "title": "",
    "image_source": "",
    "image_destination": "",
    "mask": "",
    "output": "",
    "mask" : "",
}

for key, value in defaults.items():
    st.session_state.setdefault(key, value)

# --------------------------------------------------
# Columns Layout
# --------------------------------------------------
col_mask, col_seg, col_view = st.columns(3)


with col_seg :
    
    st.header("Video segmentation, cut the video for each essay")
    show_example = st.toggle("Show JSON example", key="toggle_json_seg")

    if show_example:
        st.json({
            "session":[
            {
                "title": "Seq21 without specified intervals",
                "video": "./data/videos/Seq21.mp4",
                "mask": "./data/wall_beginning_image21.png"
            },
            {
                "title": "Seq22 with specified intervals",
                "video": "./data/videos/Seq22.mp4",
                "intervals": "[(2,17),(204,500)]"
            }]
        })

    submitted_interval = st.toggle("Video segmentation with specified intervals", key="toggle_interval_auto")

    # ---------------- JSON uploader ----------------
    uploaded_file = st.file_uploader("Upload session JSON", type=["json"], key="uploader_seg")

    if uploaded_file:
        try:
            data = json.load(uploaded_file)
            sessions = data.get("session", [])

            if not sessions:
                st.warning("No session found.")
            else:
                selected = st.selectbox("Choose a session:", [s["title"] for s in sessions], key="select_seg")
                s = next(x for x in sessions if x["title"] == selected)

                st.session_state.title = s.get("title", "")
                st.session_state.video = s.get("video", "")
                st.session_state.wall = s.get("mask", "")

                st.success("Fields imported successfully.")

                submitted_yolo_all = st.button("Run video segmentation for all sessions", key="btn_seg_all")
                if submitted_yolo_all:
                    if not submitted_interval :
                        with st.spinner("Video segmentation for all sessions"):
                            text_session_track = st.empty()
                            for i,s in enumerate(sessions):
                                with text_session_track:
                                    t = s.get("title","")
                                    current_video = s.get("video", "")
                                    current_tracking_yolo = "./temp/yolo_temp.pickle"
                                    current_tracking_mediapipe = "./temp/mediapipe_temp.pickle"
                                    current_pickle = "./temp/pickle_temp.pickle"
                                    current_xlsx = "./temp/xlsx_temp.xlsx"
                                    current_wall = s.get("mask", "")
                                    st.write(f"Processing: {t}, tracking yolo")
                                    detections, people, heatmap_acc = yolo_tracking(
                                        current_video,
                                        current_tracking_yolo,
                                        title=t,
                                        out_image_name="./temp/image_temp.png",
                                        heatmap_image_name="./temp/image_temp2.png",
                                        model_name='yolov8n-pose.pt',
                                        display=False,
                                        stream=True
                                    )
                                    st.write(f"Processing: {t}, tracking mediapipe")
                                    mediapipe_tracking(
                                        current_video,
                                        current_tracking_yolo,
                                        current_tracking_mediapipe,
                                        MODEL_PATH = "./mediapipe_models/pose_landmarker_lite.task"
                                    )
                                    st.write(f"Processing: {t}, extract tracking")
                                    extract_tracking(
                                        current_tracking_mediapipe,
                                        current_pickle,
                                        current_xlsx,
                                        0.1,
                                        0.01,
                                        0.01
                                    )
                                    st.write(f"Processing: {t}, segmentation")
                                    segmentation(
                                        s.get("video", ""), 
                                        current_pickle,
                                        current_xlsx,
                                        s.get("mask", ""),
                                        save = False
                                    )
                        shutil.rmtree("./temp")
                    else :
                        with st.spinner("Video segmentation for all sessions"):
                            text_session_track = st.empty()
                            for i,s in enumerate(sessions):
                                with text_session_track:
                                    t = s.get("title","")

                                    st.write(f"Processing: {t}, segmentation")
                                    segmentation_intervals(
                                        s.get("video", ""), 
                                        ast.literal_eval(s.get("intervals", ""))
                                    )     
            st.success("Video segmentation on all sessions complete.")

        except Exception as e:
            st.error(f"JSON error: {e}")

# --------------------------------------------------
# MASK COLUMN
# --------------------------------------------------
with col_mask:
    st.header("Creating from existing masks to new wall view")
    example_train = st.toggle("Show json training example", key="toggle_json_mask")
    if example_train:
        st.json({
        "sessions":[
            {"title": "seq1 mask to seq2",
            "image_source": "./data/tracking/wall_image_seq1.png",
            "image_destination": "./data/tracking/wall_image_seq2.png",
            "mask": "./data/mask/B1/",
            "output": "./data/mask/B1_mask_to_seq2/"
            },
            {"title": "seq1 mask to seq3",
            "image_source": "./data/tracking/wall_image_seq1.png",
            "image_destination": "./data/tracking/wall_image_seq3.png",
            "mask": "./data/mask/B1/",
            "output": "./data/mask/B1_mask_to_seq3/"
            }
        ]
    })
    # ---------------- JSON uploader ----------------
    uploaded_file = st.file_uploader("Upload session JSON", type=["json"])

    if uploaded_file:
        try:
            data = json.load(uploaded_file)
            sessions = data.get("sessions", [])

            if not sessions:
                st.warning("No session found in JSON.")
            else:
                selected = st.selectbox("Choose a session:", [s["title"] for s in sessions], key="select_mask")
                s = next(x for x in sessions if x["title"] == selected)

                st.session_state.title = s.get("title", "")
                st.session_state.image_source = s.get("image_source", "")
                st.session_state.image_destination = s.get("image_destination", "")
                st.session_state.mask = s.get("mask", "")
                st.session_state.output = s.get("output", "")

                st.success("Fields imported successfully.")

                # ---------- Run ALL sessions ----------
                submitted_mask_all = st.button("Run Mask for all sessions", key="btn_mask_all")
                if submitted_mask_all:
                    with st.spinner("Processing all sessions…"):
                        errors = []
                        text_session = st.empty()
                        for sess in sessions:
                            t = sess.get("title", "?")
                            with text_session:
                                st.write(f"Processing: {t}")
                            try:
                                processed = transfo_mask_folder(
                                    sess["mask"],
                                    sess["image_destination"],
                                    sess["image_source"],
                                    sess["output"],
                                )
                            except Exception as e:
                                errors.append(t)

                        if errors:
                            st.warning(f"Finished with errors in: {', '.join(errors)}")
                        else:
                            st.success("All sessions complete.")

        except Exception as e:
            st.error(f"JSON error: {e}")
    else:
        for key, value in defaults.items():
            st.session_state[key] = value

    # ---------------- CLEAR FIELDS ----------------
    if st.button("Clear All Fields"):
        for key, value in defaults.items():
            st.session_state[key] = value
        st.success("Fields cleared.")

    # ---------------- MASK FORM ----------------
    placeholder_mask_form = st.empty()
    with placeholder_mask_form:
        with st.form("mask_form"):
            st.text_input("Title", key="title")
            st.text_input("Image Source", key="image_source")
            st.text_input("Image Destination", key="image_destination")
            st.text_input("Mask Folder", key="mask")
            st.text_input("Output Folder", key="output")
            submitted_mask = st.form_submit_button("Run Mask")

    if submitted_mask:
        with st.spinner("Running transform"):
            try:
                processed = transfo_mask_folder(
                    st.session_state.mask,
                    st.session_state.image_destination,
                    st.session_state.image_source,
                    st.session_state.output,
                )
                st.success(f"Done — {len(processed)} mask(s) saved to: {st.session_state.output}")
            except Exception as e:
                st.error(f"Error: {e}")
