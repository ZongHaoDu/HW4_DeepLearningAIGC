
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

def preprocess_image(image):
    """
    Preprocesses the image to fit the model's input requirements.
    """
    img = image.resize(IMAGE_SIZE)
    img_array = np.array(img)
    img_array_expanded = np.expand_dims(img_array, axis=0)
    return preprocess_input(img_array_expanded)

def predict_and_display(image, model):
    """
    Displays the image and the model's prediction.
    """
    st.image(image, caption='Selected Image', use_container_width=True)
    st.write("")
    st.write("Classifying...")

    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)
    
    confidence_score = np.max(prediction)
    predicted_class_index = np.argmax(prediction)
    predicted_class_name = CLASSES[predicted_class_index]

    st.subheader("Prediction")
    st.write(f"**Breed:** {predicted_class_name}")
    st.write(f"**Confidence:** {confidence_score:.2%}")

# --- Streamlit App ---
st.title("Golden vs. Labrador Retriever Classifier")
st.write("Upload an image of a dog, or select one of the examples below.")

model = load_keras_model()

if model is not None:
    # --- Example Selector ---
    example_images = glob.glob(os.path.join(EXAMPLE_DIR, '*.jpg'))
    example_filenames = ["None"] + [os.path.basename(p) for p in example_images]
    selected_example = st.selectbox("Or select an example:", options=example_filenames)

    # --- File Uploader ---
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    image_to_process = None

    if selected_example != "None":
        example_path = os.path.join(EXAMPLE_DIR, selected_example)
        if os.path.exists(example_path):
            image_to_process = Image.open(example_path)
            st.write("Using selected example.")
    elif uploaded_file is not None:
        image_to_process = Image.open(uploaded_file)
        st.write("Using uploaded image.")

    if image_to_process:
        predict_and_display(image_to_process, model)

else:
    st.warning(f"Model file `{MODEL_PATH}` not found or could not be loaded. Please ensure the model is trained and available.")

st.sidebar.info(
    "**About**\n"
    "This app uses a deep learning model (ResNet50V2) to classify images of "
    "Golden Retrievers and Labrador Retrievers."
)
