# import os
# import cv2
# import numpy as np

# def save_image_to_binary(image_path, output_path):
#     image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
#     image = cv2.resize(image, (64, 64))
#     image = image / 255.0
#     image = image.reshape(1, 64, 64, 1)
#     image.tofile(output_path)

# if __name__ == "__main__":
#     TEST_IMAGE_PATH = "d:/Code/SourceCode/CNN_ModelAI/test.png"
#     OUTPUT_IMAGE_PATH = "d:/Code/SourceCode/CNN_ModelAI/test.bin"
#     print("Saving test.png to binary format...")
#     save_image_to_binary(TEST_IMAGE_PATH, OUTPUT_IMAGE_PATH)
#     print("test.png saved to binary format.")
    
#     print("Data preparation completed.")


import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

def load_data(data_dir):
    X, y = [], []
    label_map = {"circle": 0, "square": 1, "star": 2, "triangle": 3}
    
    for label, class_idx in label_map.items():
        class_dir = os.path.join(data_dir, label)
        for file_name in os.listdir(class_dir):
            file_path = os.path.join(class_dir, file_name)
            image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE) 
            image = cv2.resize(image, (64, 64))  
            X.append(image)
            y.append(class_idx)

    return np.array(X), np.array(y)

def preprocess_data(X, y):
    X = X / 255.0 
    X = X.reshape(-1, 64, 64, 1)
    y = np.array(y)
    return train_test_split(X, y, test_size=0.2, random_state=42)

def save_to_binary(X, y, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    X.tofile(os.path.join(output_dir, "X.bin"))
    y.tofile(os.path.join(output_dir, "y.bin"))

def save_test_data(X_test, y_test, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    X_test.tofile(os.path.join(output_dir, "X.bin"))
    y_test.tofile(os.path.join(output_dir, "Y.bin"))

if __name__ == "__main__":
    DATA_DIR = "d:/Code/SourceCode/CNN_ModelAI/shapes"
    OUTPUT_DIR = "processed_data"
    print("Loading data...")
    X, y = load_data(DATA_DIR)
    print(f"Loaded {len(X)} samples.")
    print("Preprocessing data...")
    X_train, X_test, y_train, y_test = preprocess_data(X, y)
    print(f"X_train dtype: {X_train.dtype}, shape: {X_train.shape}")
    print(f"X_test dtype: {X_test.dtype}, shape: {X_test.shape}")

    print("Saving processed data...")
    save_to_binary(X_train, y_train, os.path.join(OUTPUT_DIR, "train"))
    save_to_binary(X_test, y_test, os.path.join(OUTPUT_DIR, "test"))

    print("Saving test data...")
    save_test_data(X_test, y_test, os.path.join(OUTPUT_DIR, "test"))
    
    print("Data preparation completed.")
