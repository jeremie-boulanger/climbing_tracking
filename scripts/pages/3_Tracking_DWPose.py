# Page Title: Tracking DWPose (single pass)

import streamlit as st
import json
import os
from scripts_tracking import extract_tracking, extract_median_image, write_on_video
from scripts_tracking_dwpose import dwpose_tracking

# --------------------------------------------------
# Page config (must be FIRST!)
# --------------------------------------------------
st.set_page_config(layout="wide")
st.title("Tracking App — DWPose (single pass)")

st.caption("One ONNX pipeline (YOLOX person detection + DWPose wholebody "
           "pose, 133 keypoints incl. hands and feet) instead of "
           "YOLO + MediaPipe. Multi-person videos are supported: the "
           "climber is selected automatically as the person who climbs "
           "the highest.")

# --------------------------------------------------
# Session State Initialization
# --------------------------------------------------
defaults = {
    "title": "",
    "video": "",
    "tracking_dwpose": "",
    "tracking_pickle": "",
    "tracking_xlsx": "",
    "wall_image": "",
    "video_tracking": "",
    "pixel_size_x": 0.0,
    "pixel_size_y": 0.0,
    "freq_cut": 0.1,
}

for key, value in defaults.items():
    st.session_state.setdefault(key, value)

sessions = []
selected_session = None


def load_session_into_state(s):
    st.session_state.title = s.get("title", "")
    st.session_state.video = s.get("video", "")
    st.session_state.tracking_dwpose = s.get("tracking_dwpose",
                                             s.get("tracking_mediapipe", ""))
    st.session_state.tracking_pickle = s.get("tracking_pickle", "")
    st.session_state.tracking_xlsx = s.get("tracking_xlsx", "")
    st.session_state.wall_image = s.get("wall image", "")
    st.session_state.video_tracking = s.get("video_tracking", "")
    st.session_state.pixel_size_x = s.get("pixel size X (m/p)", 0.0)
    st.session_state.pixel_size_y = s.get("pixel size Y (m/p)", 0.0)
    st.session_state.freq_cut = s.get("frequency cut-off", 0.1)


def run_session(s, do_wall_image=False, do_video=False):
    """DWPose tracking + extraction for one session entry of the config.
    The pixel sizes and cut-off come from the session dict itself — not
    from widget state."""
    title = s.get("title", "?")
    st.write(f"Processing: {title}")
    dwpose_tracking(s["video"].strip(), s["tracking_dwpose"].strip())
    extract_tracking(
        s["tracking_dwpose"].strip(),
        s["tracking_pickle"].strip(),
        s["tracking_xlsx"].strip(),
        s.get("frequency cut-off", 0.1),
        s.get("pixel size X (m/p)", 0.0),
        s.get("pixel size Y (m/p)", 0.0),
    )
    st.write(f"Extraction complete ({s['tracking_pickle']}) — "
             f"pixel size {s.get('pixel size X (m/p)')} m/p applied.")
    if do_wall_image and s.get("wall image", "").strip():
        extract_median_image(s["video"].strip(), s["wall image"].strip())
    if do_video and s.get("video_tracking", "").strip():
        write_on_video(s["video"].strip(), s["video_tracking"].strip(),
                       s["tracking_pickle"].strip())


# --------------------------------------------------
# JSON uploader
# --------------------------------------------------
st.header("Sessions")
example = st.toggle("Show JSON example")
if example:
    st.json({
        "session": [
            {
                "title": "CL_B1E1",
                "video": "./data/videos/CL_B1E1_h265_2.mp4",
                "tracking_dwpose": "./data/tracking_dwpose/CL_B1E1_pose.pickle",
                "tracking_pickle": "./data/tracking_dwpose/CL_B1E1_tracking.pickle",
                "tracking_xlsx": "./data/tracking_dwpose/CL_B1E1_tracking.xlsx",
                "wall image": "./data/tracking_dwpose/CL_B1E1_wall_image.png",
                "video_tracking": "./data/tracking_dwpose/CL_B1E1_tracking.mp4",
                "pixel size X (m/p)": 0.00483,
                "pixel size Y (m/p)": 0.00483,
                "frequency cut-off": 0.1,
                "delta t": 0.02
            }
        ]
    })

uploaded_file = st.file_uploader("Upload session JSON", type=["json"])
if uploaded_file:
    try:
        data = json.load(uploaded_file)
        sessions = data.get("session", [])
        if not sessions:
            st.warning("No session found in the JSON (use the 'session' key).")
    except Exception as e:
        st.error(f"JSON error: {e}")

# --------------------------------------------------
# Session selection + run buttons
# --------------------------------------------------
if sessions:
    selected = st.selectbox("Choose a session:",
                            [s.get("title", f"session{i}")
                             for i, s in enumerate(sessions)])
    selected_session = next(
        s for s in sessions if s.get("title", "") == selected)
    load_session_into_state(selected_session)

    col_a, col_b = st.columns(2)
    with col_a:
        run_one = st.button("Run DWPose tracking + extraction "
                            "(selected session)")
    with col_b:
        run_all = st.button("Run for all sessions")

    # optional overrides — NOT in a form, so values set from the JSON
    # actually show up and are used (st.form widgets ignore session_state)
    with st.expander("Parameters (loaded from the JSON, editable)"):
        px = st.number_input("Pixel Size X (m/p)",
                             value=float(st.session_state.pixel_size_x),
                             key="pixel_size_x", min_value=0.0, step=0.0001,
                             format="%.5f")
        py = st.number_input("Pixel Size Y (m/p)",
                             value=float(st.session_state.pixel_size_y),
                             key="pixel_size_y", min_value=0.0, step=0.0001,
                             format="%.5f")
        fc = st.number_input("Cut-off Frequency",
                             value=float(st.session_state.freq_cut),
                             key="freq_cut", min_value=0.001, step=0.01,
                             format="%.3f")
        do_wall_image = st.checkbox("Also extract the wall image", True)
        do_video = st.checkbox("Also write the tracking video", False)

    if run_one and selected_session is not None:
        s = dict(selected_session)
        # the editable overrides win over the JSON values
        s["pixel size X (m/p)"] = st.session_state.pixel_size_x
        s["pixel size Y (m/p)"] = st.session_state.pixel_size_y
        s["frequency cut-off"] = st.session_state.freq_cut
        with st.spinner("DWPose tracking + extraction..."):
            run_session(s, do_wall_image, do_video)
        st.success(f"DWPose complete: {selected}")

    if run_all:
        with st.spinner("DWPose on all sessions..."):
            for s in sessions:
                try:
                    run_session(s, do_wall_image, do_video)
                except Exception as e:
                    st.error(f"{s.get('title', '?')}: {e}")
        st.success("DWPose on all sessions complete.")

    st.divider()
    st.subheader("Media")
    col_img, col_vid = st.columns(2)
    with col_img:
        if st.button("Extract wall image (selected session)"):
            extract_median_image(st.session_state.video,
                                  st.session_state.wall_image)
            st.success("Wall image extracted.")
        if os.path.isfile(st.session_state.wall_image):
            st.image(st.session_state.wall_image,
                     caption=f"Climbing wall: {st.session_state.title}",
                     width="stretch")
    with col_vid:
        if st.button("Create tracking video (selected session)"):
            write_on_video(st.session_state.video,
                           st.session_state.video_tracking,
                           st.session_state.tracking_pickle)
            st.success("Video generated.")
        if os.path.isfile(st.session_state.video_tracking):
            st.video(st.session_state.video_tracking)
else:
    st.info("Upload a sessions JSON to start (see the example above).")
