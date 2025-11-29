
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.keras.applications.resnet_v2 import preprocess_input
from tensorflow.keras.models import load_model

# --- Configuration ---
MODEL_PATH = 'retriever_model.h5'
IMAGE_SIZE = (224, 224)
# This should correspond to the output of your training script
CLASSES = ['Golden_Retriever', 'Labrador_Retriever']

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
    Preprocesses the uploaded image to fit the model's input requirements.
    """
    # Resize the image
    img = image.resize(IMAGE_SIZE)
    # Convert image to numpy array
    img_array = np.array(img)
    # Expand dimensions to create a batch of 1
    img_array_expanded = np.expand_dims(img_array, axis=0)
    # Apply ResNetV2 specific preprocessing
    return preprocess_input(img_array_expanded)

# --- Streamlit App ---
st.title("Golden vs. Labrador Retriever Classifier")
st.write("Upload an image of a dog, and the model will predict if it's a Golden Retriever or a Labrador Retriever.")

# Load the model
model = load_keras_model()

if model is not None:
    # Image uploader
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_container_width=True)
        st.write("")
        st.write("Classifying...")

        # Preprocess the image and make a prediction
        processed_image = preprocess_image(image)
        prediction = model.predict(processed_image)
        
        # Get the highest confidence score and its index
        confidence_score = np.max(prediction)
        predicted_class_index = np.argmax(prediction)
        predicted_class_name = CLASSES[predicted_class_index]

        # Display the result
        st.subheader("Prediction")
        st.write(f"**Breed:** {predicted_class_name}")
        st.write(f"**Confidence:** {confidence_score:.2%}")
else:
    st.warning("Model file `retriever_model.h5` not found. Please train the model first by running `train_model.py`.")

st.sidebar.info(
    "**About**\n"
    "This app uses a deep learning model (ResNet50V2) to classify images of "
    "Golden Retrievers and Labrador Retrievers."
)
