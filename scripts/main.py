# Page Title: Main

import streamlit as st
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

st.set_page_config(page_title="Climbing tracking", page_icon="🧗‍♀️", layout="wide")

st.title("🧗‍♀️ Welcome to the climbing tracking and hold detection app.")

st.markdown("---")

st.header("⚡ Tracking full automatic")
st.write("Run the full algorithm to process the video automatically.")

if st.button("Run YOLO + MediaPipe", type="primary", use_container_width=True):
    st.switch_page("pages/1_Tracking_Auto.py")

st.markdown("---")

st.header("🧱 Hold detection")
st.write("Train a model or do the inference for hold detection.")

if st.button("Run hold detection", type="primary", use_container_width=True):
    st.switch_page("pages/4_Hold_usage_detection.py")

st.markdown("---")

st.header("🛠️ Manual correction")
st.caption("A step in the automatic tracking process failed ? Use this modules below:")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Tracking not working ?")
    st.write("Restart climber tracking only (YOLO and/or MediaPipe).")
    if st.button("Run manual tracking"):
        st.switch_page("pages/2_Tracking.py")

with col2:
    st.subheader("Problem with the mask or video segmentation ?")
    st.write("Manually correct the video segmentation or create new masks.")
    if st.button("Go to Mask & Segmentation"):
        st.switch_page("pages/3_Mask_&_Segmentation.py")