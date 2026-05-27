import streamlit as st
import json
import os
import numpy as np
from scripts_mask import *

# --------------------------------------------------
# Page config (must be FIRST!)
# --------------------------------------------------
st.set_page_config(layout="wide")
st.title("Mask")

# --------------------------------------------------
# Session State Initialization
# --------------------------------------------------
defaults = {
    "title": "",
    "image_source": "",
    "source_point": "",
    "image_destination": "",
    "destination_point": "",
    "mask": "",
    "output": "",
}

for key, value in defaults.items():
    st.session_state.setdefault(key, value)

# --------------------------------------------------
# Columns Layout
# --------------------------------------------------
col_mask, col_preview, _ = st.columns(3)

# --------------------------------------------------
# MASK COLUMN
# --------------------------------------------------
with col_mask:
    st.header("Creating from existing masks to new wall view")
    example_train = st.toggle("Show json training example")
    if example_train:
        st.json({
        "sessions":[
            {"title": "seq1 mask to seq2",
            "image_source": "./data/tracking/wall_image_seq1.png",
            "source_point": "./data/point/point_seq1.png",
            "image_destination": "./data/tracking/wall_image_seq2.png",
            "destination_point": "./data/point/point_seq2.png",
            "mask": "./data/mask/B1/",
            "output": "./data/mask/B1_mask_to_seq2/"
            },
            {"title": "seq1 mask to seq3",
            "image_source": "./data/tracking/wall_image_seq1.png",
            "source_point": "./data/point/point_seq1.png",
            "image_destination": "./data/tracking/wall_image_seq3.png",
            "destination_point": "./data/point/point_seq3.png",
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
                selected = st.selectbox(
                    "Choose a session:", [s["title"] for s in sessions]
                )
                s = next(x for x in sessions if x["title"] == selected)

                st.session_state.title = s.get("title", "")
                st.session_state.image_source = s.get("image_source", "")
                src_pts = s.get("source_point", "")
                st.session_state.source_point = src_pts if isinstance(src_pts, str) else json.dumps(src_pts)
                st.session_state.image_destination = s.get("image_destination", "")
                dst_pts = s.get("destination_point", "")
                st.session_state.destination_point = dst_pts if isinstance(dst_pts, str) else json.dumps(dst_pts)
                st.session_state.mask = s.get("mask", "")
                st.session_state.output = s.get("output", "")

                st.success("Fields imported successfully.")

                # ---------- Run ALL sessions ----------
                submitted_mask_all = st.button("Run Mask for all sessions")
                if submitted_mask_all:
                    with st.spinner("Processing all sessions…"):
                        errors = []
                        text_session = st.empty()
                        for sess in sessions:
                            t = sess.get("title", "?")
                            with text_session:
                                st.write(f"Processing: {t}")
                            try:
                                val_src = sess["source_point"]
                                val_dst = sess["destination_point"]
                                
                                if isinstance(val_src, str): val_src = json.loads(val_src)
                                if isinstance(val_dst, str): val_dst = json.loads(val_dst)

                                pts_src = np.array(val_src, dtype=np.float32)
                                pts_dst = np.array(val_dst, dtype=np.float32)
                                processed = transfo_mask_folder(
                                    pts_src,
                                    pts_dst,
                                    sess["mask"],
                                    sess["image_destination"],
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
            st.text_input("Source Points", key="source_point")
            st.text_input("Image Destination", key="image_destination")
            st.text_input("Destination Points", key="destination_point")
            st.text_input("Mask Folder", key="mask")
            st.text_input("Output Folder", key="output")
            submitted_mask = st.form_submit_button("Run Mask")

    if submitted_mask:
        with st.spinner("Running transform"):
            try:
                pts_src = np.array(json.loads(st.session_state.source_point), dtype=np.float32)
                pts_dst = np.array(json.loads(st.session_state.destination_point), dtype=np.float32)
                processed = transfo_mask_folder(
                    pts_src,
                    pts_dst,
                    st.session_state.mask,
                    st.session_state.image_destination,
                    st.session_state.output,
                )
                st.success(f"Done — {len(processed)} mask(s) saved to: {st.session_state.output}")
            except Exception as e:
                st.error(f"Error: {e}")