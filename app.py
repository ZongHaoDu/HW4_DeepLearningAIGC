import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import glob
from tensorflow.keras.applications.resnet_v2 import preprocess_input
from tensorflow.keras.models import load_model

# --- 1. 配置 ---
MODEL_PATH = 'retriever_model.h5'
IMAGE_SIZE = (224, 224)
CLASSES = ['Golden_Retriever', 'Labrador_Retriever']
EXAMPLE_DIR = './Test/'

# --- 2. 功能函數 ---
@st.cache_resource
def load_keras_model():
    try:
        return load_model(MODEL_PATH) if os.path.exists(MODEL_PATH) else None
    except:
        return None

def preprocess_image(image):
    img = image.resize(IMAGE_SIZE)
    img_array = np.array(img)
    return preprocess_input(np.expand_dims(img_array, axis=0))

def predict_and_display(image, model, source_type):
    st.info(f"📍 當前顯示：{source_type}")
    st.image(image, use_container_width=True)
    
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)
    
    score = np.max(prediction)
    class_name = CLASSES[np.argmax(prediction)].replace('_', ' ')

    st.success(f"**辨識結果:** {class_name} (信心度: {score:.2%})")

# --- 3. Session State 初始化與回呼函數 ---

# 初始化優先權狀態，預設為 'upload'
if 'last_action' not in st.session_state:
    st.session_state.last_action = 'upload'

def mark_upload():
    st.session_state.last_action = 'upload'

def mark_example():
    st.session_state.last_action = 'example'

# --- 4. 主要 UI 與邏輯 ---
def main():
    st.title("Labrador、Labrador 狗品種辨識器 ")
    model = load_keras_model()
    
    if model is None:
        st.error("找不到模型檔案，請確認 retriever_model.h5 位置。")
        return

    # 建立兩欄 UI
    col1, col2 = st.columns(2)

    with col1:
        # 當使用者上傳或清除檔案時，觸發 mark_upload
        uploaded_file = st.file_uploader(
            "上傳圖片", 
            type=["jpg", "png"], 
            key="user_upload",
            on_change=mark_upload
        )

    with col2:
        example_images = glob.glob(os.path.join(EXAMPLE_DIR, '*.jpg'))
        example_filenames = ["None"] + [os.path.basename(p) for p in example_images]
        # 當使用者切換選單時，觸發 mark_example
        selected_example = st.selectbox(
            "或選擇範例", 
            options=example_filenames,
            key="example_select",
            on_change=mark_example
        )

    # --- 關鍵決策邏輯 ---
    image_to_process = None
    source_label = ""

    # 1. 如果最後動作是「上傳」且檔案存在
    if st.session_state.last_action == 'upload' and uploaded_file is not None:
        # 修改點：加上 .convert('RGB')
        image_to_process = Image.open(uploaded_file).convert('RGB')
        source_label = "您上傳的照片"
    
    # 2. 如果最後動作是「選範例」且選了有效範例
    elif st.session_state.last_action == 'example' and selected_example != "None":
        example_path = os.path.join(EXAMPLE_DIR, selected_example)
        if os.path.exists(example_path):
            # 修改點：加上 .convert('RGB')
            image_to_process = Image.open(example_path).convert('RGB')
            source_label = f"範例圖片: {selected_example}"
    
    # 3. 備援機制：如果最後動作與內容不匹配（例如刪除了上傳圖，但範例還選著），自動切換
    else:
        if uploaded_file is not None:
            image_to_process = Image.open(uploaded_file)
            source_label = "您上傳的照片"
        elif selected_example != "None":
            example_path = os.path.join(EXAMPLE_DIR, selected_example)
            image_to_process = Image.open(example_path)
            source_label = f"範例圖片: {selected_example}"

    # --- 執行辨識 ---
    if image_to_process:
        predict_and_display(image_to_process, model, source_label)
    else:
        st.write("💡 請先上傳圖片或從右側選單挑選範例。")

if __name__ == "__main__":
    main()