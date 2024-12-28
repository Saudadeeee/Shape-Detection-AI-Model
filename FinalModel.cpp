#include "cnn_model.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>

const int ORIGINAL_IMAGE_SIZE = 200;
const int NEW_IMAGE_SIZE = 64;

int main() {
    CNN model;
    model.load_weights_from_csv("cnn_weights.csv");

    // Load and preprocess the test image from binary file
    std::ifstream file("test_binary.bin", std::ios::binary);
    if (!file) {
        std::cerr << "Error: Could not load binary image 'test_binary.bin'" << std::endl;
        return -1;
    }

    // Read the entire file into a buffer
    std::vector<unsigned char> buffer((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    if (buffer.size() != ORIGINAL_IMAGE_SIZE * ORIGINAL_IMAGE_SIZE) {
        std::cerr << "Error: Image size does not match expected dimensions" << std::endl;
        return -1;
    }

    // Resize to 64x64
    std::vector<unsigned char> resized_buffer(NEW_IMAGE_SIZE * NEW_IMAGE_SIZE);
    for (int i = 0; i < NEW_IMAGE_SIZE; ++i) {
        for (int j = 0; j < NEW_IMAGE_SIZE; ++j) {
            int orig_i = i * ORIGINAL_IMAGE_SIZE / NEW_IMAGE_SIZE;
            int orig_j = j * ORIGINAL_IMAGE_SIZE / NEW_IMAGE_SIZE;
            resized_buffer[i * NEW_IMAGE_SIZE + j] = buffer[orig_i * ORIGINAL_IMAGE_SIZE + orig_j];
        }
    }

    float input[NEW_IMAGE_SIZE][NEW_IMAGE_SIZE];
    for (int i = 0; i < NEW_IMAGE_SIZE; ++i) {
        for (int j = 0; j < NEW_IMAGE_SIZE; ++j) {
            input[i][j] = resized_buffer[i * NEW_IMAGE_SIZE + j] / 255.0f;
        }
    }

    // Debug: Print the input values
    std::cout << "Input values: ";
    for (int i = 0; i < NEW_IMAGE_SIZE; ++i) {
        for (int j = 0; j < NEW_IMAGE_SIZE; ++j) {
            std::cout << input[i][j] << " ";
        }
    }
    std::cout << std::endl;

    auto output = model.forward(input);

    // Debug: Print the output values
    std::cout << "Model output values: ";
    for (const auto& value : output) {
        std::cout << value << " ";
    }
    std::cout << std::endl;

    // Map the model output to shape categories
    std::vector<std::string> shape_categories = {"Square", "Circle", "Triangle", "Star"};
    int predicted_index = std::distance(output.begin(), std::max_element(output.begin(), output.end()));
    // Debug: Print the predicted index
    std::cout << "Predicted index: " << predicted_index << std::endl;

    std::cout << "Model output: " << shape_categories[predicted_index] << std::endl;

    return 0;
}
