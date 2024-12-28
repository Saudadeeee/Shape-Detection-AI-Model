#ifndef CNN_MODEL_H
#define CNN_MODEL_H

#include <vector>
#include <cmath>
#include <algorithm>
#include <random>
#include <fstream> 
#include <string>
#include "activation_functions.h"

const int IMAGE_SIZE = 64;
const int NUM_CLASSES = 4;

struct Image {
    float data[IMAGE_SIZE][IMAGE_SIZE];
    int label;
};

class CNN {
public:
    CNN() {
        initialize_weights();
    }

    std::vector<float> forward(const float input[IMAGE_SIZE][IMAGE_SIZE]) {
        std::cout << "Forward pass started." << std::endl; // Add this line
        // Convolutional layer 1
        std::vector<float> conv_output1(32 * 32 * 32, 0.0f);
        for (int f = 0; f < 32; ++f) {
            for (int i = 0; i < 32; ++i) {
                for (int j = 0; j < 32; ++j) {
                    float sum = 0.0f;
                    for (int ki = 0; ki < 3; ++ki) {
                        for (int kj = 0; kj < 3; ++kj) {
                            sum += input[i + ki][j + kj] * conv1_weights[f][ki * 3 + kj];
                        }
                    }
                    conv_output1[f * 32 * 32 + i * 32 + j] = sum;
                }
            }
        }
        conv_output1 = relu(conv_output1);
        conv_output1 = max_pool2d(conv_output1, 32, 2);

        // Convolutional layer 2
        std::vector<float> conv_output2(64 * 16 * 16, 0.0f);
        for (int f = 0; f < 64; ++f) {
            for (int i = 0; i < 16; ++i) {
                for (int j = 0; j < 16; ++j) {
                    float sum = 0.0f;
                    for (int ki = 0; ki < 3; ++ki) {
                        for (int kj = 0; kj < 3; ++kj) {
                            sum += conv_output1[(i + ki) * 32 + (j + kj)] * conv2_weights[f][ki * 3 + kj];
                        }
                    }
                    conv_output2[f * 16 * 16 + i * 16 + j] = sum;
                }
            }
        }
        conv_output2 = relu(conv_output2);
        conv_output2 = max_pool2d(conv_output2, 16, 2);

        // Convolutional layer 3
        std::vector<float> conv_output3(128 * 8 * 8, 0.0f);
        for (int f = 0; f < 128; ++f) {
            for (int i = 0; i < 8; ++i) {
                for (int j = 0; j < 8; ++j) {
                    float sum = 0.0f;
                    for (int ki = 0; ki < 3; ++ki) {
                        for (int kj = 0; kj < 3; ++kj) {
                            sum += conv_output2[(i + ki) * 16 + (j + kj)] * conv3_weights[f][ki * 3 + kj];
                        }
                    }
                    conv_output3[f * 8 * 8 + i * 8 + j] = sum;
                }
            }
        }
        conv_output3 = relu(conv_output3);
        conv_output3 = max_pool2d(conv_output3, 8, 2);

        // Fully connected layer 1
        std::vector<float> fc1_output(256, 0.0f);
        for (int i = 0; i < 256; ++i) {
            float sum = 0.0f;
            for (int j = 0; j < 128 * 8 * 8; ++j) {
                sum += conv_output3[j] * fc1_weights[j];
            }
            fc1_output[i] = sum;
        }
        fc1_output = relu(fc1_output);

        // Fully connected layer 2
        std::vector<float> fc2_output(128, 0.0f);
        for (int i = 0; i < 128; ++i) {
            float sum = 0.0f;
            for (int j = 0; j < 256; ++j) {
                sum += fc1_output[j] * fc2_weights[j];
            }
            fc2_output[i] = sum;
        }
        fc2_output = relu(fc2_output);

        // Fully connected layer 3
        std::vector<float> output(NUM_CLASSES, 0.0f);
        for (int i = 0; i < NUM_CLASSES; ++i) {
            float sum = 0.0f;
            for (int j = 0; j < 128; ++j) {
                sum += fc2_output[j] * fc3_weights[j];
            }
            output[i] = sum;
        }

        std::cout << "Forward pass completed." << std::endl; // Add this line
        return output;
    }

    void backward(const float input[IMAGE_SIZE][IMAGE_SIZE], const std::vector<float>& output, int label, float learning_rate) {
        std::cout << "Backward pass started." << std::endl; // Add this line
        std::vector<float> output_grad(NUM_CLASSES, 0.0f);
        output_grad[label] = 1.0f - output[label];
        std::cout << "output_grad[" << label << "] = " << output_grad[label] << std::endl; // Add this line

        std::vector<float> fc3_grad(128, 0.0f);
        for (int i = 0; i < NUM_CLASSES; ++i) {
            for (int j = 0; j < 128; ++j) {
                fc3_grad[j] += output_grad[i];
                fc3_weights[j] += learning_rate * output_grad[i];
            }
        }
        std::cout << "FC3 gradient calculated." << std::endl; // Add this line

        std::vector<float> fc2_grad(256, 0.0f);
        for (int i = 0; i < 128; ++i) {
            for (int j = 0; j < 256; ++j) {
                fc2_grad[j] += fc3_grad[i];
                fc2_weights[j] += learning_rate * fc3_grad[i];
            }
        }
        std::cout << "FC2 gradient calculated." << std::endl; // Add this line

        std::vector<float> conv3_grad(128 * 8 * 8, 0.0f);
        for (int f = 0; f < 128; ++f) {
            for (int i = 0; i < 8; ++i) {
                for (int j = 0; j < 8; ++j) {
                    for (int ki = 0; ki < 3; ++ki) {
                        for (int kj = 0; kj < 3; ++kj) {
                            conv3_grad[f * 8 * 8 + i * 8 + j] += fc2_grad[f * 8 * 8 + i * 8 + j];
                            conv3_weights[f][ki * 3 + kj] += learning_rate * conv3_grad[f * 8 * 8 + i * 8 + j];
                        }
                    }
                }
            }
        }
        std::cout << "Conv3 gradient calculated." << std::endl; // Add this line

        std::vector<float> conv2_grad(64 * 16 * 16, 0.0f);
        for (int f = 0; f < 64; ++f) {
            std::cout << "Calculating gradient for conv2 filter " << f + 1 << "/64" << std::endl; // Add this line
            for (int i = 0; i < 16; ++i) {
                for (int j = 0; j < 16; ++j) {
                    for (int ki = 0; ki < 3; ++ki) {
                        for (int kj = 0; kj < 3; ++kj) {
                            conv2_grad[f * 16 * 16 + i * 16 + j] += conv3_grad[f * 16 * 16 + i * 16 + j];
                            conv2_weights[f][ki * 3 + kj] += learning_rate * conv2_grad[f * 16 * 16 + i * 16 + j];
                            std::cout << "output_grad[" << label << "] = " << output_grad[label] << std::endl; // Add this line
                            std::cout << "fc3_grad[" << j << "] = " << fc3_grad[j] << std::endl; // Add this line
                            std::cout << "fc2_grad[" << f << "][" << i << "][" << j << "] = " << fc2_grad[f * 8 * 8 + i * 8 + j] << std::endl; // Add this line
                            std::cout << "conv3_grad[" << f << "][" << i << "][" << j << "] = " << conv3_grad[f * 8 * 8 + i * 8 + j] << std::endl; // Add this line
                            std::cout << "conv2_grad[" << f << "][" << i << "][" << j << "] updated: " << conv2_grad[f * 16 * 16 + i * 16 + j] << std::endl; // Add this line
                        }
                    }
                }
            }
        }
        std::cout << "Conv2 gradient calculated." << std::endl; // Add this line

        std::vector<float> conv1_grad(32 * 32 * 32, 0.0f);
        for (int f = 0; f < 32; ++f) {
            for (int i = 0; i < 32; ++i) {
                for (int j = 0; j < 32; ++j) {
                    for (int ki = 0; ki < 3; ++ki) {
                        for (int kj = 0; kj < 3; ++kj) {
                            conv1_grad[f * 32 * 32 + i * 32 + j] += conv2_grad[f * 32 * 32 + i * 32 + j];
                            conv1_weights[f][ki * 3 + kj] += learning_rate * conv1_grad[f * 32 * 32 + i * 32 + j];
                        }
                    }
                }
            }
        }
        std::cout << "Conv1 gradient calculated." << std::endl; // Add this line
        std::cout << "Backward pass completed." << std::endl; // Add this line
    }

    void train_batch(const std::vector<Image>& batch, float learning_rate) {
        std::cout << "Training batch of size " << batch.size() << std::endl; // Add this line
        for (size_t i = 0; i < batch.size(); ++i) {
            const auto& img = batch[i];
            auto output = forward(img.data);
            backward(img.data, output, img.label, learning_rate);
            std::cout << "Processed image " << i + 1 << "/" << batch.size() << std::endl; // Add this line
        }
    }

    void save_weights(const std::string& file_path) {
        std::ofstream file(file_path, std::ios::binary);
        if (!file.is_open()) {
            std::cerr << "Error: Could not open file to save weights." << std::endl;
            return;
        }
        save_vector(file, conv1_weights);
        save_vector(file, conv2_weights);
        save_vector(file, conv3_weights);
        save_vector(file, fc1_weights);
        save_vector(file, fc2_weights);
        save_vector(file, fc3_weights);
        file.close();
    }

    void load_weights(const std::string& file_path) {
        std::ifstream file(file_path, std::ios::binary);
        if (!file.is_open()) {
            std::cerr << "Error: Could not open file to load weights." << std::endl;
            return;
        }
        load_vector(file, conv1_weights);
        load_vector(file, conv2_weights);
        load_vector(file, conv3_weights);
        load_vector(file, fc1_weights);
        load_vector(file, fc2_weights);
        load_vector(file, fc3_weights);
        file.close();
    }

private:
    std::vector<std::vector<float>> conv1_weights;
    std::vector<std::vector<float>> conv2_weights;
    std::vector<std::vector<float>> conv3_weights;
    std::vector<float> fc1_weights;
    std::vector<float> fc2_weights;
    std::vector<float> fc3_weights;

    void initialize_weights() {
        conv1_weights = std::vector<std::vector<float>>(32, std::vector<float>(9, 0.01f));
        conv2_weights = std::vector<std::vector<float>>(64, std::vector<float>(32 * 3 * 3, 0.01f));
        conv3_weights = std::vector<std::vector<float>>(128, std::vector<float>(64 * 3 * 3, 0.01f));
        fc1_weights = std::vector<float>(128 * 8 * 8, 0.01f);
        fc2_weights = std::vector<float>(256, 0.01f);
        fc3_weights = std::vector<float>(128, 0.01f);
    }

    template <typename T>
    void save_vector(std::ofstream& file, const std::vector<T>& vec) {
        size_t size = vec.size();
        file.write(reinterpret_cast<const char*>(&size), sizeof(size));
        file.write(reinterpret_cast<const char*>(vec.data()), size * sizeof(T));
    }

    template <typename T>
    void load_vector(std::ifstream& file, std::vector<T>& vec) {
        size_t size;
        file.read(reinterpret_cast<char*>(&size), sizeof(size));
        vec.resize(size);
        file.read(reinterpret_cast<char*>(vec.data()), size * sizeof(T));
    }

    std::vector<float> relu(const std::vector<float>& input) {
        std::vector<float> output(input.size());
        std::transform(input.begin(), input.end(), output.begin(), [](float x) { return std::max(0.0f, x); });
        return output;
    }

    std::vector<float> max_pool2d(const std::vector<float>& input, int input_size, int pool_size) {
        int output_size = input_size / pool_size;
        std::vector<float> output(output_size * output_size * (input.size() / (input_size * input_size)), 0.0f);
        for (size_t c = 0; c < input.size() / (input_size * input_size); ++c) {
            for (int i = 0; i < output_size; ++i) {
                for (int j = 0; j < output_size; ++j) {
                    float max_val = -std::numeric_limits<float>::infinity();
                    for (int ki = 0; ki < pool_size; ++ki) {
                        for (int kj = 0; kj < pool_size; ++kj) {
                            max_val = std::max(max_val, input[c * input_size * input_size + (i * pool_size + ki) * input_size + (j * pool_size + kj)]);
                        }
                    }
                    output[c * output_size * output_size + i * output_size + j] = max_val;
                }
            }
        }
        return output;
    }
};

#endif // CNN_MODEL_H