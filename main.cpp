#include <iostream>
#include "cnn_model.h"
#include "data_loader.h"
#include <vector>
#include <fstream>

void train(CNN& model, const std::vector<Image>& train_data, int epochs, float learning_rate, size_t batch_size) {
    for (int epoch = 0; epoch < epochs; ++epoch) {
        std::cout << "Epoch " << epoch + 1 << "/" << epochs << " started." << std::endl; // Add this line
        for (size_t i = 0; i < train_data.size(); i += batch_size) {
            size_t end = std::min(i + batch_size, train_data.size());
            std::vector<Image> batch(train_data.begin() + i, train_data.begin() + end);
            augment_data(batch); // Apply data augmentation
            model.train_batch(batch, learning_rate);
            std::cout << "Processed batch " << i / batch_size + 1 << "/" << (train_data.size() + batch_size - 1) / batch_size << std::endl; // Add this line
        }
        std::cout << "Epoch " << epoch + 1 << "/" << epochs << " completed." << std::endl;
    }
}

void evaluate(CNN& model, const std::vector<Image>& test_data) {
    int correct = 0;
    for (const auto& img : test_data) {
        auto output = model.forward(img.data);
        int prediction = std::distance(output.begin(), std::max_element(output.begin(), output.end()));
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
    std::cout << "Starting main function..." << std::endl;

    std::string train_X_path = "processed_data/train/X.bin";
    std::string train_y_path = "processed_data/train/y.bin";
    std::string test_X_path = "processed_data/test/X.bin";
    std::string test_y_path = "processed_data/test/Y.bin";

    std::cout << "Loading training data..." << std::endl;
    std::vector<Image> train_data = load_data(train_X_path, train_y_path);
    std::cout << "Training data loaded." << std::endl;

    std::cout << "Loading test data..." << std::endl;
    std::vector<Image> test_data = load_data(test_X_path, test_y_path);
    std::cout << "Test data loaded." << std::endl;

    CNN model;

    std::cout << "Starting training..." << std::endl;
    train(model, train_data, 50, 0.0001f, 32);
    model.save_weights("cnn_weights.bin");
    std::cout << "Model training completed." << std::endl;

    std::cout << "Loading model weights..." << std::endl;
    model.load_weights("cnn_weights.bin");
    std::cout << "Evaluating model..." << std::endl;
    evaluate(model, test_data);

    std::cout << "Main function completed." << std::endl;
    return 0;
}