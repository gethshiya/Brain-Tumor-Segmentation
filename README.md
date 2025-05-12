# 🧠 Brain Tumor Classifier using CNN & Gradio
This project is a deep learning-based Brain Tumor Classification System that uses a Convolutional Neural Network (CNN) to classify MRI brain scans into one of four categories:

Glioma

Meningioma

No Tumor

Pituitary Tumor

It also provides a Gradio web interface for easy image uploads and predictions.

# 🚀 Features
Image classification using CNN

Four-class tumor detection

Interactive Gradio interface

Dataset loaded from Google Drive

# 📁 Dataset
The dataset is structured as follows inside Google Drive:
/Brain Tumor Segmentation/Training/
    ├── glioma/
    ├── meningioma/
    ├── notumor/
    └── pituitary/
Each folder contains MRI images belonging to the corresponding tumor class.

# 🧰 Dependencies
Install the required Python libraries:

pip install gradio opencv-python-headless numpy tensorflow scikit-learn
For Google Colab:
from google.colab import drive
drive.mount('/content/drive')
# 🧠 Model Architecture
A CNN model built using TensorFlow/Keras with the following layers:

3× Conv2D + MaxPooling + BatchNorm

GlobalAveragePooling

Dense Layer (ReLU)

Dropout

Output Layer (Softmax)

# 🏗️ How It Works
Loads and preprocesses images from the Google Drive dataset.

Trains a CNN model to classify into 4 categories.

Provides a Gradio interface to upload and classify a new brain scan image.

# 🖼️ Using the Gradio Interface
Once you run the script:
interface.launch()
A Gradio interface will appear. You can:

Upload an MRI image

Get the predicted tumor type

# 🧪 Example Prediction
Upload a sample image like below:

sample_image.jpg
You’ll get a prediction like:
Prediction: Glioma (92% confidence)

# 📊 Model Training
The model is trained with:

Image size: 128×128

Batch size: 16

Epochs: 10

Loss: Categorical Crossentropy

Optimizer: Adam

# 📎 File Structure
├── brain_tumor_classifier.py   # Main Python script
├── README.md                   # Project README

# ✅ Conclusion
This project demonstrates the power of deep learning in the field of medical image analysis. By leveraging a Convolutional Neural Network (CNN), we can effectively classify brain MRI scans into four tumor categories with high accuracy. The use of Gradio enhances usability by providing a simple, interactive interface for real-time predictions. With further improvements, such as training on a larger dataset or optimizing the architecture, this system has the potential to assist medical professionals in early tumor detection and diagnosis.
