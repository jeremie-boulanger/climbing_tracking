# Page Title: Hold usage detection

import streamlit as st
import json
import os
from scripts_train import *


# Title of your app
st.set_page_config(layout="wide")
st.title("Hold detection")


defaults = {
    "sessions_train": [],
    "sessions_test": [],
    "dataset_train": "",
    "dataset_train2": "",
    "network_out": "",
    "dataset_test": "",
    "network_in": ""
}

for key, value in defaults.items():
    st.session_state.setdefault(key, value)

# --------------------------------------------------
# Columns Layout
# --------------------------------------------------
col_train, col_test, _ = st.columns(3)
    
    
submitted_dataset = False
submitted_training = False
submitted_inference = False


with col_train:
    st.header("Training")
    example_train = st.toggle("Show json training example")
    if example_train:
        st.json({
        "sessions":[
            {"position": "./data/tracking/Seq21_tracking.pickle",
            "annotation": "./data/annotation/Seq21_annotation.xlsx",
            "mask": "./data/holds/"},
            {"position": "./data/tracking/Seq22_tracking.pickle",
            "annotation": "./data/annotation/Seq22_annotation.xlsx",
            "mask": "./data/holds/"}
            ],
        "dataset": "./data/dataset.pickle",
        "network": "./data/u_net.pt"
    })

    # File uploader
    uploaded_file_train = st.file_uploader("Upload a JSON file for the training fields", type=["json"])

    if uploaded_file_train:
        try:
            data = json.load(uploaded_file_train)
            st.session_state.sessions_train = data.get("sessions", st.session_state.sessions_train)
            st.session_state.dataset_train = data.get("dataset", st.session_state.dataset_train)
            st.session_state.dataset_train2 = data.get("dataset", st.session_state.dataset_train2)
            st.session_state.network_out = data.get("network", st.session_state.network_out)
            st.success("Fields auto-filled from file!")
        except Exception as e:
            st.error(f"Error reading file: {e}")




    st.subheader("Generate a dataset to train")
    placeholder_generating_dataset = st.empty()
    with placeholder_generating_dataset:
        # Create a form for user input
        with st.form("generate dataset"):
            dataset_train = st.text_input("Enter dataset (OUT) path:", key="dataset_train")
            submitted_dataset = st.form_submit_button("Generate training dataset from sessions.")

    if submitted_dataset:
        with st.spinner("Generating dataset..."):
            get_dataset(st.session_state.sessions_train, dataset_train)
            st.success("Dataset generation for training complete.")


    st.subheader("Train the network from an annotated dataset")
    placeholder_training = st.empty()
    with placeholder_training:
        with st.form("train"):
            dataset_train2 = st.text_input("Enter dataset (IN) path:", key="dataset_train2")
            network_out = st.text_input("Enter network (OUT) path", key="network_out")
            submitted_training = st.form_submit_button("Train from generated dataset")

    fig = None
    if submitted_training and os.path.isfile(st.session_state.dataset_train2):
        with st.spinner("Training the network..."):
            loss, fig = train_NN(dataset_train2, network_out)
            st.success(f"Training complete, final F1-score: {-loss[-1]:.2f}.")
    if submitted_training and not os.path.isfile(st.session_state.dataset_train2):
        st.warning(f"Dataset for training not generated.")

    placeholder_training_fig = st.empty()
    if fig is not None:
        with placeholder_training_fig:
            st.pyplot(fig)

with col_test:
    st.header("Inference")
    example_test = st.toggle("Show json inference example")
    if example_test:
        st.json({
    "sessions":[
        {"position": "./data/tracking/Seq21_tracking.pickle",
        "annotation": "./data/annotation/Seq21_annotation.xlsx",
        "mask": "./data/holds/",
        "prediction": "./data/inference/Seq21.pickle",
        "plot": "./data/inference/Seq21.png"},

        {"position": "./data/tracking/Seq22_tracking.pickle",
        "mask": "./data/holds/",
        "prediction": "./data/inference/Seq22.pickle",
        "annotation": "./data/annotation/Seq22_annotation.xlsx",
        "plot": "./data/inference/Seq22.png"}
        ],
    "dataset": "./data/dataset_test.pickle",
    "network": "./data/u_net.pt"
    })

    # File uploader
    uploaded_file_test = st.file_uploader("Upload a JSON file for the inference fields", type=["json"])

    if uploaded_file_test:
        try:
            data = json.load(uploaded_file_test)
            st.session_state.sessions_test = data.get("sessions", st.session_state.sessions_test)
            st.session_state.dataset_test = data.get("dataset", st.session_state.dataset_test)
            st.session_state.network_in = data.get("network", st.session_state.network_in)
            st.success("Fields auto-filled from file!")
        except Exception as e:
            st.error(f"Error reading file: {e}")


    # Create a form for user input
    placeholder_inference_form = st.empty()
    with placeholder_inference_form:
        with st.form("inference"):
            dataset_test = st.text_input("Enter dataset test (OUT) path:", key="dataset_test")
            network_in = st.text_input("Enter network (IN) path", key="network_in")
            submitted_inference = st.form_submit_button("Perform inference from sessions.")

    if submitted_inference:
#        get_dataset(st.session_state.sessions_test, dataset_test)
        with st.spinner("Performing inference on dataset"):
            f1 = get_detection(st.session_state.sessions_test, dataset_test, network_in)
            st.success(f"Inference and dataset generation complete. Final F1-score is {f1:.2f}")
