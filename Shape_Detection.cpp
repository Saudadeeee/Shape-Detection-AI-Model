#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <filesystem>
#include <map>
#include <cmath>

namespace fs = std::filesystem;

void resize_image(const std::vector<unsigned char>& input, int input_width, int input_height, std::vector<unsigned char>& output, int output_width, int output_height) {
    output.resize(output_width * output_height);
    for (int y = 0; y < output_height; ++y) {
        for (int x = 0; x < output_width; ++x) {
            int src_x = x * input_width / output_width;
            int src_y = y * input_height / output_height;
            output[y * output_width + x] = input[src_y * input_width + src_x];
        }
    }
}

void load_image(const std::string& file_path, std::vector<double>& image) {
    std::ifstream file(file_path, std::ios::binary);
    if (file) {
        file.seekg(0, std::ios::end);
        size_t size = file.tellg();
        file.seekg(0, std::ios::beg);
        std::vector<unsigned char> buffer(size);
        file.read(reinterpret_cast<char*>(buffer.data()), size);

        std::cout << "File size: " << size << " bytes" << std::endl;

        int input_width = 200;  // Original image width
        int input_height = 200; // Original image height
        std::vector<unsigned char> resized_buffer;
        resize_image(buffer, input_width, input_height, resized_buffer, 64, 64);

        for (size_t i = 0; i < resized_buffer.size(); ++i) {
            image.push_back(static_cast<double>(resized_buffer[i]) / 255.0);
        }
    } else {
        std::cerr << "Failed to open file: " << file_path << std::endl;
    }
}

void load_data(const std::string& data_dir, std::vector<std::vector<double>>& X, std::vector<int>& y) {
    std::map<std::string, int> label_map = {{"circle", 0}, {"square", 1}, {"triangle", 2}, {"star", 3}};
    
    for (const auto& [label, class_idx] : label_map) {
        std::string class_dir = data_dir + "/" + label;
        std::cout << "Loading images from: " << class_dir << std::endl;
        for (const auto& entry : fs::directory_iterator(class_dir)) {
            std::cout << "Processing file: " << entry.path().string() << std::endl;
            std::vector<double> image;
            load_image(entry.path().string(), image);
            if (!image.empty()) {
                X.push_back(image);
                y.push_back(class_idx);
                std::cout << "Loaded image: " << entry.path().string() << std::endl;
            } else {
                std::cerr << "Failed to load image: " << entry.path().string() << std::endl;
            }
        }
    }
}

void preprocess_data(const std::vector<std::vector<double>>& X, const std::vector<int>& y, std::vector<std::vector<double>>& X_train, std::vector<std::vector<double>>& X_test, std::vector<int>& y_train, std::vector<int>& y_test) {
    size_t total_size = X.size();
    size_t test_size = static_cast<size_t>(total_size * 0.2);
    size_t train_size = total_size - test_size;

    for (size_t i = 0; i < total_size; ++i) {
        if (i < train_size) {
            X_train.push_back(X[i]);
            y_train.push_back(y[i]);
        } else {
            X_test.push_back(X[i]);
            y_test.push_back(y[i]);
        }
    }
}

void save_to_binary(const std::vector<std::vector<double>>& X, const std::vector<int>& y, const std::string& output_dir) {
    fs::create_directories(output_dir);

    std::ofstream X_file(output_dir + "/X.bin", std::ios::binary);
    std::ofstream y_file(output_dir + "/y.bin", std::ios::binary);

    for (const auto& vec : X) {
        X_file.write(reinterpret_cast<const char*>(vec.data()), vec.size() * sizeof(double));
    }
    y_file.write(reinterpret_cast<const char*>(y.data()), y.size() * sizeof(int));
}

void save_test_data(const std::vector<std::vector<double>>& X_test, const std::vector<int>& y_test, const std::string& output_dir) {
    fs::create_directories(output_dir);

    std::ofstream X_file(output_dir + "/X.bin", std::ios::binary);
    std::ofstream y_file(output_dir + "/Y.bin", std::ios::binary);

    for (const auto& vec : X_test) {
        X_file.write(reinterpret_cast<const char*>(vec.data()), vec.size() * sizeof(double));
    }
    y_file.write(reinterpret_cast<const char*>(y_test.data()), y_test.size() * sizeof(int));
}

int main() {
    std::string DATA_DIR = "d:/Code/SourceCode/CNN_ModelAI/shapes";
    std::string OUTPUT_DIR = "processed_data";
    std::cout << "Loading data..." << std::endl;

    std::vector<std::vector<double>> X;
    std::vector<int> y;
    load_data(DATA_DIR, X, y);

    std::cout << "Loaded " << X.size() << " samples." << std::endl;
    std::cout << "Preprocessing data..." << std::endl;

    std::vector<std::vector<double>> X_train, X_test;
    std::vector<int> y_train, y_test;
    preprocess_data(X, y, X_train, X_test, y_train, y_test);

    // Print data sizes for verification
    std::cout << "X_train size: " << X_train.size() << ", first element size: " << (X_train.empty() ? 0 : X_train[0].size()) << std::endl;
    std::cout << "X_test size: " << X_test.size() << ", first element size: " << (X_test.empty() ? 0 : X_test[0].size()) << std::endl;

    // Ensure the shapes match the Python output
    std::cout << "Expected X_train shape: (11976, 64 * 64)" << std::endl;
    std::cout << "Expected X_test shape: (2994, 64 * 64)" << std::endl;

    std::cout << "Saving processed data..." << std::endl;
    save_to_binary(X_train, y_train, OUTPUT_DIR + "/train");
    save_to_binary(X_test, y_test, OUTPUT_DIR + "/test");

    std::cout << "Saving test data..." << std::endl;
    save_test_data(X_test, y_test, OUTPUT_DIR + "/test");

    std::cout << "Data preparation completed." << std::endl;
    return 0;
}
