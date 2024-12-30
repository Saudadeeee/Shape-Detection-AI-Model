import os
import cv2
import numpy as np

def save_image_to_binary(image_path, output_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (64, 64))
    image = image / 255.0
    image = image.reshape(1, 64, 64, 1)
    image.tofile(output_path)

if __name__ == "__main__":
  
    TEST_IMAGE_PATH = "d:/Code/SourceCode/CNN_ModelAI/test.png"
    OUTPUT_IMAGE_PATH = "d:/Code/SourceCode/CNN_ModelAI/test.bin"
    print("Saving test.png to binary format...")
    save_image_to_binary(TEST_IMAGE_PATH, OUTPUT_IMAGE_PATH)
    print("test.png saved to binary format.")
    
    print("Data preparation completed.")
