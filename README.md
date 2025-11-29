# Golden vs. Labrador Retriever Image Classifier

This project is a deep learning-based image classifier built to distinguish between Golden Retrievers and Labrador Retrievers. It uses transfer learning with the ResNet50V2 architecture and is deployed as an interactive web application using Streamlit.

## ABSTRACT

This project presents a deep learning solution for binary image classification, specifically designed to differentiate between two similar dog breeds: the Golden Retriever and the Labrador Retriever. Given the significant visual overlap between these breeds, this task poses a notable challenge, which is further compounded by the constraint of a very limited dataset, comprising only about ten images per class. To address the high risk of overfitting associated with small datasets, this project employs a robust two-fold strategy.

The core of the solution is transfer learning, leveraging the powerful ResNet50V2 architecture pre-trained on the extensive ImageNet dataset. By using these pre-trained weights, the model capitalizes on generalized features learned from millions of images, providing a strong foundation for this specialized task. The top classification layer of the ResNet50V2 model is removed and replaced with a custom head, consisting of a Global Average Pooling 2D layer followed by a Dense layer with a softmax activation function, tailored for our binary classification problem.

To further mitigate overfitting and enhance the model's ability to generalize, extensive data augmentation is implemented using Keras's `ImageDataGenerator`. This technique artificially expands the training set by applying a variety of random transformations to the existing images, such as rotation, zooming, and horizontal flipping. This process creates a more diverse set of training examples, teaching the model to be invariant to changes in orientation and scale. The model is trained on this augmented data, learning to discern the subtle features that distinguish the two retriever breeds.

The final trained model is deployed as an interactive web application using Streamlit, providing a user-friendly interface for real-time predictions. Users can upload an image, and the application, powered by the cached TensorFlow/Keras model, will return the predicted breed along with a confidence score, demonstrating a practical application of deep learning in a real-world scenario despite the initial data limitations.

## 摘要

本專案旨在為二元圖像分類提供一個深度學習解決方案，專門用於區分黃金獵犬與拉布拉多獵犬這兩種外觀相似的犬種。此任務本身已具挑戰，而每類僅約十張圖像的極端有限資料集更劇了此挑戰。為了解決小型資料集帶來的高度過擬合風險，本專案採用了雙重策略。

方案的核心是遷移學習，我們利用在龐大ImageNet資料集上預先訓練的強大ResNet50V2架構。透過使用這些預訓練權重，模型得以運用從數百萬張圖像中學到的通用特徵，為此特定任務奠定堅實基礎。我們移除了ResNet50V2的頂層分類層，並換上一個客製化的分類頭，包含一個全局平均池化層（GlobalAveragePooling2D）和一個使用softmax激活函數的Dense層，以適應我們的二元分類目標。

為了進一步緩解過擬合並提升模型的泛化能力，我們使用Keras的`ImageDataGenerator`進行了大規模的資料增強。此技術透過對現有圖像進行多種隨機轉換（如旋轉、縮放、水平翻轉）來人工擴增訓練集，從而產生更多樣化的訓練樣本，讓模型學會對方向和尺寸的變化保持不變性。

最終訓練好的模型被部署為一個使用Streamlit開發的互動式Web應用程式，為即時預測提供了一個用戶友好的界面。使用者可以上傳圖像，應用程式將利用快取的TensorFlow/Keras模型回傳預測的品種及置信度分數，展示了在資料受限的情況下，深度學習的實際應用價值。

## Features

- **Transfer Learning:** Utilizes a pre-trained `ResNet50V2` model on ImageNet for high-accuracy feature extraction.
- **Data Augmentation:** Employs `ImageDataGenerator` to prevent overfitting on a small dataset.
- **Streamlit Web App:** Provides an easy-to-use web interface for uploading images and viewing predictions.
- **High Performance:** Caches the TensorFlow/Keras model in Streamlit for fast subsequent predictions.

## Directory Structure

```
.
├── Golden_Retriever/
│   └── (Images of Golden Retrievers)
├── Labrador_Retriever/
│   └── (Images of Labrador Retrievers)
├── app.py                  # Streamlit web application
├── train_model.py          # Script to train the classification model
├── retriever_model.h5      # The saved, trained Keras model
├── requirements.txt        # Project dependencies
└── README.md               # This file
```

## Setup and Installation

1.  **Clone the repository (or download the files).**

2.  **Create and activate a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Usage

There are two main parts to this project: training the model and running the application.

### 1. Training the Model

If you want to re-train the model (e.g., with new data), run the training script. This will generate a new `retriever_model.h5` file.

```bash
python train_model.py
```

### 2. Running the Application

To start the web application, run the following command. Make sure the `retriever_model.h5` file exists in the same directory.

```bash
streamlit run app.py
```

Your web browser should open with the application running. You can now upload a JPG or PNG image of a dog to get a prediction.
