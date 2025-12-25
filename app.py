
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import glob
from tensorflow.keras.applications.resnet_v2 import preprocess_input
from tensorflow.keras.models import load_model

# --- Configuration ---
MODEL_PATH = 'retriever_model.h5'
IMAGE_SIZE = (224, 224)
CLASSES = ['Golden_Retriever', 'Labrador_Retriever']
EXAMPLE_DIR = './Test/'

# --- Model Loading ---
@st.cache_resource
def load_keras_model():

    """

    Loads the pre-trained Keras model.

    The @st.cache_resource decorator ensures the model is loaded only once.

    """

    try:

        model = load_model(MODEL_PATH)

        return model

    except Exception as e:

        st.error(f"Error loading model: {e}")

        return None

# --- Logic ---
def main():
    st.title("Golden vs. Labrador Retriever Classifier")
    
    model = load_keras_model()
    if model is None:
        st.warning("Model not found.")
        return

    # --- UI 佈局 ---
    # 1. 範例選擇器 (不使用 callback 修改 state)
    example_images = glob.glob(os.path.join(EXAMPLE_DIR, '*.jpg'))
    example_filenames = ["None"] + [os.path.basename(p) for p in example_images]
    
    selected_example = st.selectbox(
        "Select an example image:",
        options=example_filenames,
        key='example_selector'
    )

    # 2. 檔案上傳器
    uploaded_file = st.file_uploader(
        "Or upload your own image...",
        type=["jpg", "jpeg", "png"],
        key='uploaded_file'
    )

    # --- 優先權邏輯控制 ---
    image_to_process = None

    # 邏輯：如果有上傳檔案，以上傳檔案優先；否則看是否有選擇範例
    if uploaded_file is not None:
        image_to_process = Image.open(uploaded_file)
        st.info("Using uploaded image.")
    elif selected_example != "None":
        example_path = os.path.join(EXAMPLE_DIR, selected_example)
        if os.path.exists(example_path):
            image_to_process = Image.open(example_path)
            st.info(f"Using example: {selected_example}")

    # --- 執行預測 ---
    if image_to_process:
        predict_and_display(image_to_process, model)

if __name__ == "__main__":
    main()
