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


def run_full_pipeline(s, do_image, do_video):
    """YOLO (if not already tracked) -> default climber (highest track)
    -> MediaPipe -> extraction, for one session entry of the config.
    The pixel sizes and cut-off come from the session dict itself."""
    video = s["video"].strip()
    ty = s["tracking_yolo"].strip()
    if not os.path.isfile(ty):
        yolo_tracking(
            video, ty,
            title=s.get("title", ""),
            out_image_name=s.get("image_tracking", "").strip() or None,
            model_name=YOLO_MODEL,
            display=False,
            stream=True)
    with open(ty, "rb") as f:
        data = pickle.load(f)
    ranked, stats = rank_climber_tracks(data["list_detection"])
    if not ranked:
        raise RuntimeError("No person track long enough after YOLO.")
    top = ranked[0]
    st.write(f"Default climber: track {top} "
             f"({stats[top]['frames']} frames, "
             f"top y={stats[top]['top_y']})")
    # the MediaPipe pass follows the ids stored in the yolo pickle
    with open(ty, "wb") as f:
        pickle.dump({"list_detection": data["list_detection"],
                     "tracking_person": [top]}, f,
                    protocol=pickle.HIGHEST_PROTOCOL)
    mediapipe_tracking(video, ty, s["tracking_mediapipe"].strip())
    extract_tracking(
        s["tracking_mediapipe"].strip(),
        s["tracking_pickle"].strip(),
        s["tracking_xlsx"].strip(),
        s.get("frequency cut-off", 0.1),
        s.get("pixel size X (m/p)", 0.0),
        s.get("pixel size Y (m/p)", 0.0))
    st.write(f"Extraction complete: {s['tracking_pickle'].strip()} — "
             f"pixel size {s.get('pixel size X (m/p)')} m/p applied")
    if do_image and s.get("wall image", "").strip():
        extract_median_image(video, s["wall image"].strip())
    if do_video and s.get("video_tracking", "").strip():
        write_on_video(video, s["video_tracking"].strip(),
                       s["tracking_pickle"].strip())

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

                # ---------- FULL PIPELINE (all sessions) ----------
                st.subheader("Full pipeline on all sessions")
                c1, c2 = st.columns(2)
                with c1:
                    do_image = st.checkbox("Wall image OUT", value=True,
                                          key="pipeline_wall_image")
                with c2:
                    do_video = st.checkbox("Tracking video OUT", value=False,
                                           key="pipeline_video")
                submitted_pipeline = st.button(
                    "Run YOLO + MediaPipe + extract on all sessions "
                    "(default = highest climber)")
                if submitted_pipeline:
                    with st.spinner("Running full pipeline..."):
                        for i, s in enumerate(sessions):
                            title = s.get("title", "?")
                            st.write(f"Processing: {title}")
                            try:
                                run_full_pipeline(s, do_image, do_video)
                            except Exception as e:
                                st.error(f"{title}: {e}")
                        st.success("Full pipeline complete on all sessions.")
                        


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
                    model_name=YOLO_MODEL,
                    display=False,
                    stream=True
                )
            else:
                # Charger les données existantes
                with open(st.session_state.tracking_yolo, "rb") as f:
                    data = pickle.load(f)
                detections = data["list_detection"]
                if os.path.isfile(st.session_state.heatmap_tracking):
                    heatmap_acc = cv2.imread(st.session_state.heatmap_tracking, cv2.IMREAD_UNCHANGED)
                else:
                    heatmap_acc = None

            st.success("YOLO complete. Select the climber(s) in the "
                       "MediaPipe column.")

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



# --------------------------------------------------
# MEDIAPIPE COLUMN
# --------------------------------------------------
with col_mediapipe:

    st.header("Mediapipe Pose Tracking")

    # climber candidates from the YOLO pickle, as checkboxes: the top
    # track is pre-checked; tick extra tracks if the climber leaves the
    # camera view and comes back re-detected with a new id
    selected = []
    ty = st.session_state.tracking_yolo
    if ty and os.path.isfile(ty):
        with open(ty, "rb") as f:
            data = pickle.load(f)
        ranked, stats = rank_climber_tracks(data["list_detection"])
        if ranked:
            st.caption("Climber candidates, best first (highest point "
                       "reached). The top track is pre-checked; tick "
                       "more if needed.")
            for tid in ranked:
                if st.checkbox(
                        f"track {tid} — {stats[tid]['frames']} frames, "
                        f"top y = {stats[tid]['top_y']} px",
                        value=(tid == ranked[0]),
                        key=f"climber_{st.session_state.title}_{tid}"):
                    selected.append(tid)
        else:
            st.info("No person track long enough in the YOLO tracking "
                    "(100 frames minimum). Run the YOLO step again.")
    else:
        st.info("Run the YOLO step first — the climber candidates come "
                "from its pickle.")
    st.session_state.list_climbers_tracking = ",".join(map(str, selected))

    placeholder_mediapipe = st.empty()
    with placeholder_mediapipe:
        with st.form("mediapipe_form"):
            st.text_input("Mediapipe Tracking OUT", key="tracking_mediapipe")
            submitted_mediapipe = st.form_submit_button("Run Mediapipe")

    if submitted_mediapipe:
        if len(st.session_state.list_climbers_tracking)==0:
            st.warning("No climber selected — tick at least one track above.")
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
    # NOTE: pixel size / cut-off inputs are OUTSIDE the form on purpose —
    # Streamlit does not update widgets inside a st.form from
    # session_state, so values loaded from the JSON config were lost
    # (pixel size fell back to 0 and the (m) columns were wrong).
    st.number_input("Pixel Size X (m/p)", key="pixel_size_x", min_value=0.0,
                    step=0.0001, format="%.5f")
    st.number_input("Pixel Size Y (m/p)", key="pixel_size_y", min_value=0.0,
                    step=0.0001, format="%.5f")
    st.number_input("Cut-off Frequency", key="freq_cut", min_value=0.001,
                    step=0.01, format="%.3f")
    placeholder_extraction = st.empty()
    with placeholder_extraction:
        with st.form("extract_form"):
            st.text_input("Pickle OUT", key="tracking_pickle")
            st.text_input("Excel OUT", key="tracking_xlsx")
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

