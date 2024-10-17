---

### 1. Project Overview

This project contains three primary Python applications and associated model files. It utilizes neural networks for image classification and produces visualizations such as GradCAM results and confusion matrices. The three `.pth` model files store pre-trained models used for making predictions.

### 2. File and Folder Structure

- **Files:**
  - `cnntest.py`: This script tests a Convolutional Neural Network (CNN) for classification tasks.
  - `fastapi_app.py`: This script runs a FastAPI app to expose an API for image classification.
  - `streamlit_app.py`: This script runs a Streamlit app to create a web-based UI for interacting with the model.

- **Folders:**
  - `resnet_gradcam`: Contains GradCAM visualizations for the ResNet model.
  - `densnet_gradcam`: Contains GradCAM visualizations for the DenseNet model.
  - `confusion_matrix_images`: Contains confusion matrix images generated from the classification results.

- **Model Files (`.pth`):**
  - These files contain the pre-trained models. You will load these models in the test scripts or apps for predictions. The models are typically trained using CNN architectures like ResNet or DenseNet.
  
- **Idle Shell File:**
  - This file captures the execution of the program and provides logs or output from the execution process.

### 3. Installation and Setup

Follow these steps to set up the project on your machine:

1. Clone or download the project repository.
2. Ensure you have Python installed (version 3.x recommended).
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   The `requirements.txt` file should include the libraries used, such as `torch`, `fastapi`, `streamlit`, etc.

4. Download the trained model files and place them in the appropriate folder, if not already present.

### 4. Running the Project

You can run the individual components as follows:

- **Running the CNN Test Script** (`cnntest.py`):
  ```bash
  python cnntest.py --model resnet --input_image <path_to_image>
  ```
  This script will load the corresponding trained model and test it on an input image. You can specify different models like ResNet or DenseNet using command-line arguments.

- **Running the FastAPI App** (`fastapi_app.py`):
  ```bash
  uvicorn fastapi_app:app --reload
  ```
  This will start a local FastAPI server, allowing you to send POST requests to classify images.

- **Running the Streamlit App** (`streamlit_app.py`):
  ```bash
  streamlit run streamlit_app.py
  ```
  This will start a web application where you can interactively upload images and get classification results, as well as visualize GradCAM and confusion matrices.

### 5. Usage Examples

- **Using FastAPI to Classify an Image**:
  Send a POST request with the image data to the FastAPI server (usually `http://localhost:8000/predict`):
  ```bash
  curl -X POST "http://localhost:8000/predict" -H "accept: application/json" -H "Content-Type: multipart/form-data" -F "file=@<image_path>"
  ```

- **Using the Streamlit App**:
  Upload an image via the Streamlit app interface and get predictions along with visualizations of the model's activations (GradCAM).

### 6. Results

- **GradCAM Visualizations**:  
  - The folders `resnet_gradcam` and `densnet_gradcam` store the GradCAM results for images processed through ResNet and DenseNet models respectively. GradCAM helps interpret the decisions made by the CNN by highlighting important regions in the image.

- **Confusion Matrix Images**:  
  - The `confusion_matrix_images` folder contains the confusion matrices generated after testing the model on a set of images. This helps in visualizing how well the model performs across different classes.

### 7. Future Work/Customization

- **Model Customization**: You can easily swap in new models by saving them as `.pth` files and modifying the script to load them accordingly.
- **Expand Dataset**: The project can be extended to handle more datasets or different image types. You can retrain the models or fine-tune existing ones for new datasets.
- **Improve UI**: The Streamlit or FastAPI apps can be further developed with features like real-time updates, batch processing of images, or additional visualizations.

--- 
