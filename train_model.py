
import os
import glob
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.applications import ResNet50V2
from tensorflow.keras.applications.resnet_v2 import preprocess_input
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# --- Configuration ---
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 8 # Small batch size for a small dataset
EPOCHS = 20
DATA_DIR = './'
CLASSES = ['Golden_Retriever', 'Labrador_Retriever']

# --- 1. Load Data ---
print("Loading and preprocessing data...")
image_paths = []
labels = []

for class_name in CLASSES:
    class_dir = os.path.join(DATA_DIR, class_name)
    for img_path in glob.glob(os.path.join(class_dir, '*.jpg')):
        image_paths.append(img_path)
        labels.append(class_name)

# --- 2. Preprocess Data ---
# Convert labels to numbers
le = LabelEncoder()
labels_encoded = le.fit_transform(labels)
labels_one_hot = tf.keras.utils.to_categorical(labels_encoded, num_classes=len(CLASSES))

# Load images
images = np.array([img_to_array(load_img(path, target_size=IMAGE_SIZE)) for path in image_paths])

# Create a class dictionary
class_indices = {i: name for i, name in enumerate(le.classes_)}
print(f"Class Indices Found: {class_indices}")


# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(
    images, labels_one_hot, test_size=0.2, random_state=42, stratify=labels_one_hot
)

# Apply ResNetV2 specific preprocessing
X_train = preprocess_input(X_train)
X_val = preprocess_input(X_val)

# --- 3. Data Augmentation ---
print("Setting up data augmentation...")
datagen = ImageDataGenerator(
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

train_generator = datagen.flow(X_train, y_train, batch_size=BATCH_SIZE)

# --- 4. Build the Model ---
print("Building the transfer learning model...")
# Load base model with pre-trained ImageNet weights, excluding the top classification layer
base_model = ResNet50V2(weights='imagenet', include_top=False, input_shape=IMAGE_SIZE + (3,))

# Freeze the base model layers
base_model.trainable = False

# Add custom layers on top
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
predictions = Dense(len(CLASSES), activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

# --- 5. Compile the Model ---
print("Compiling model...")
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# --- 6. Train the Model ---
print("Training model...")
history = model.fit(
    train_generator,
    steps_per_epoch=len(X_train) // BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=(X_val, y_val),
    verbose=1
)

# --- 7. Save the Model ---
MODEL_PATH = 'retriever_model.h5'
print(f"Training complete. Saving model to {MODEL_PATH}...")
model.save(MODEL_PATH)
print("Model saved successfully.")
print(f"Class Indices to remember: {class_indices}")
