#include <iostream>
#include "cnn_model.h"
#include "data_loader.h" 
#include <vector>
#include <fstream>

void evaluate(CNN& model, const std::vector<Image>& test_data) {
    int correct = 0;
    for (const auto& img : test_data) {
        int prediction = model.predict(img);
        if (prediction == img.label) {
            ++correct;
        }
    }
    float accuracy = 100.0f * correct / test_data.size();
    std::cout << "Model accuracy: " << accuracy << "%" << std::endl;
}

Image load_captured_image(const std::string& file_path) {
    std::ifstream file(file_path, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open captured image file." << std::endl;
        exit(1);
    }

    Image img;
    file.read(reinterpret_cast<char*>(img.data), IMAGE_SIZE * IMAGE_SIZE * sizeof(float));
    file.close();
    return img;
}

int main() {
    std::string train_X_path = "processed_data/train/X.bin";
    std::string train_y_path = "processed_data/train/y.bin";
    std::string test_X_path = "processed_data/test/X.bin";
    std::string test_y_path = "processed_data/test/Y.bin";

    std::vector<Image> train_data = load_data(train_X_path, train_y_path);
    std::vector<Image> test_data = load_data(test_X_path, test_y_path);

    CNN model;
    model.train(train_data, 50, 0.0001f, 32);
    model.save_weights("cnn_weights.bin");

    std::cout << "Model training completed." << std::endl;

    model.load_weights("cnn_weights.bin");
    evaluate(model, test_data);

    // Load the captured image from ESP32 and predict its label
    std::string captured_image_path = "path_to_captured_image.bin"; // Update with the actual path
    Image captured_img = load_captured_image(captured_image_path);
    int captured_predicted_label = model.predict(captured_img);
    std::cout << "Predicted label for the captured image: " << captured_predicted_label << std::endl;

    return 0;
}