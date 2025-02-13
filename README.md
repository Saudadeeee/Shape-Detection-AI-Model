# 🎯 Project AI and Microprocessor

## 📌 Topic: Shape Recognition Using CNN
This project uses a Convolutional Neural Network (CNN) to recognize basic shapes such as squares, circles, triangles, and stars.

---

## 📖 Table of Contents
- [1️⃣ Introduction](#1%EF%B8%8F%C2%A0introduction)
- [2️⃣ CNN Architecture](#2%EF%B8%8F%C2%A0cnn-architecture)
- [3️⃣ Usage Guide](#3%EF%B8%8F%C2%A0usage-guide)
- [4️⃣ Directory Structure](#4%EF%B8%8F%C2%A0directory-structure)
- [5️⃣ Installation & Run](#5%EF%B8%8F%C2%A0installation--run)
- [6️⃣ System Requirements](#6%EF%B8%8F%C2%A0system-requirements)
- [7️⃣ License](#7%EF%B8%8F%C2%A0license)

---

## 1️⃣ Introduction
This project is designed to detect common shapes from input images using a CNN model. The workflow is as follows:
1. Receive input images from users.
2. Preprocess and normalize the images.
3. Classify the shape using the CNN model.
4. Return the prediction through a backend API.

---

## 2️⃣ CNN Architecture
The CNN model is defined in [`cnn_model.py`](cnn_model.py) with the following key components:

| Layer            | Description                                                 |
|------------------|-------------------------------------------------------------|
| **Conv2d**       | 2D convolution layer for feature extraction.                |
| **BatchNorm2d**  | Batch normalization for more stable training.               |
| **ReLU**         | Non-linear activation to learn complex representations.     |
| **MaxPool2d**    | Pools features to reduce spatial dimensions.                |
| **Linear**       | Fully connected layer for final shape classification.       |

---

## 3️⃣ Usage Guide

### 🛠 **Overall Workflow**  
1. **Prepare images**: Input images must be in `.PNG` format.  
2. **Send images to server**: Use a `POST` request to the backend API.  
3. **Preprocessing**: The image is temporarily stored and prepared for the CNN model.  
4. **Prediction**: The CNN model classifies the shape.  
5. **Result**: The backend returns a JSON response with the predicted shape.

### 🚀 **Run the Backend Server**
```bash
python app.py
```

---

## 4️⃣ Directory Structure
Below is a simple overview of the project layout:
- cnn_model.py (defines the CNN architecture)

---

## 5️⃣ Installation & Run
Use the following commands to install dependencies and run the project:
```
pip install -r requirements.txt
python app.py
```

---

## 6️⃣ System Requirements
- Python 3.7+  
- CPU or GPU with CUDA support for faster training

---

## 7️⃣ License
This project is licensed under the MIT License.
