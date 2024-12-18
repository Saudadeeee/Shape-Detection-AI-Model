#include <iostream>
#include "cnn_model.h"
#include "data_loader.h" 

int main() {
    std::string train_X_path = "processed_data/train/X.bin";
    std::string train_y_path = "processed_data/train/y.bin";

    std::vector<Image> train_data = load_data(train_X_path, train_y_path);

    CNN model;
    model.train(train_data, 20, 0.0005); 

    std::cout << "Model training completed." << std::endl;
    std::string test_X_path = "processed_data/test/X.bin";
    std::string test_y_path = "processed_data/test/Y.bin";

    std::vector<Image> test_data = load_data(test_X_path, test_y_path);

    int correct_predictions = 0;
    for (const auto& img : test_data) {
        int predicted_label = model.predict(img);
        if (predicted_label == img.label) {
            correct_predictions++;
        }
    }

    float accuracy = static_cast<float>(correct_predictions) / test_data.size();
    std::cout << "Model accuracy: " << accuracy * 100 << "%" << std::endl;

    return 0;
}