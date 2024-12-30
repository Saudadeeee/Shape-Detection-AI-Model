#include <iostream>
#include "cnn_model.h"
#include "data_loader.h"
#include <vector>
#include <fstream>
#include <cmath>
#include <algorithm>

// ...existing code...

void train(CNN& model, std::vector<Image>& train_data, int epochs, float learning_rate) {
    // Adam optimizer parameters
    float beta1 = 0.9f;
    float beta2 = 0.999f;
    float epsilon = 1e-8f;
    std::vector<float> m1(model.get_fc2_weights().size(), 0.0f);
    std::vector<float> v1(model.get_fc2_weights().size(), 0.0f);

    for (int epoch = 0; epoch < epochs; ++epoch) {
        std::cout << "Epoch " << epoch + 1 << "/" << epochs << " started." << std::endl;
        for (auto& img : train_data) {
            auto output = model.forward(img.data);
            std::vector<float> target(NUM_CLASSES, 0.0f);
            target[img.label] = 1.0f;

            // Compute loss (CrossEntropyLoss)
            float loss = 0.0f;
            for (int i = 0; i < NUM_CLASSES; ++i) {
                if (output[i] > 0) {
                    loss -= target[i] * std::log(output[i]);
                }
            }

            // Print loss for debugging
            std::cout << "Loss: " << loss << std::endl;

            // Backpropagation (simplified)
            std::vector<float> grad_output(NUM_CLASSES);
            for (int i = 0; i < NUM_CLASSES; ++i) {
                grad_output[i] = output[i] - target[i];
            }

            // Update weights (simplified)
            auto fc2_weights = model.get_fc2_weights();
            for (size_t i = 0; i < fc2_weights.size(); ++i) {
                fc2_weights[i] -= learning_rate * grad_output[i % NUM_CLASSES];
            }
            model.set_fc2_weights(fc2_weights);
        }
        std::cout << "Epoch " << epoch + 1 << "/" << epochs << " completed." << std::endl;
    }
}

void evaluate(CNN& model, std::vector<Image>& test_data) {
    int correct = 0;
    for (auto& img : test_data) {
        auto output = model.forward(img.data);
        int predicted_index = std::distance(output.begin(), std::max_element(output.begin(), output.end()));
        if (predicted_index == img.label) {
            ++correct;
        }
    }
    float accuracy = 100.0f * correct / test_data.size();
    std::cout << "Model accuracy: " << accuracy << "%" << std::endl;
}

int main() {
    CNN model;  // Initialize model from scratch

    std::string train_X_path = "processed_data/train/X.bin";
    std::string train_y_path = "processed_data/train/y.bin";
    std::string test_X_path = "processed_data/test/X.bin";
    std::string test_y_path = "processed_data/test/Y.bin";

    int batch_size = 1000;
    std::vector<Image> train_data = load_data(train_X_path, train_y_path, batch_size);
    std::vector<Image> test_data = load_data(test_X_path, test_y_path, batch_size);

    std::cout << "Training model..." << std::endl;
    train(model, train_data, 30, 0.0001);
    model.save_weights_binary("cnn_weights.bin");
    std::cout << "Model training completed." << std::endl;

    std::cout << "Evaluating model..." << std::endl;
    evaluate(model, test_data);

    return 0;
}