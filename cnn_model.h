#ifndef CNN_MODEL_H
#define CNN_MODEL_H

#include <vector>
#include <cmath>
#include <algorithm>
#include <fstream>
#include <string>
#include <iostream>
#include <sstream>
#include <random>
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
        // Convolutional layer 1
        std::vector<float> conv_output1(8 * 64 * 64, 0.0f);
        for (int f = 0; f < 8; ++f) {
            for (int i = 0; i < 64; ++i) {
                for (int j = 0; j < 64; ++j) {
                    float sum = 0.0f;
                    for (int ki = 0; ki < 3; ++ki) {
                        for (int kj = 0; kj < 3; ++kj) {
                            int ni = i + ki - 1;
                            int nj = j + kj - 1;
                            if (ni >= 0 && ni < 64 && nj >= 0 && nj < 64) {
                                sum += input[ni][nj] * conv1_weights[f][ki * 3 + kj];
                            }
                        }
                    }
                    conv_output1[f * 64 * 64 + i * 64 + j] = sum + conv1_bias[f];
                }
            }
        }
        conv_output1 = batch_norm(conv_output1, bn1_mean, bn1_var, bn1_gamma, bn1_beta);
        conv_output1 = relu(conv_output1);
        conv_output1 = max_pool2d(conv_output1, 64, 2);

        // Convolutional layer 2
        std::vector<float> conv_output2(16 * 32 * 32, 0.0f);
        for (int f = 0; f < 16; ++f) {
            for (int i = 0; i < 32; ++i) {
                for (int j = 0; j < 32; ++j) {
                    float sum = 0.0f;
                    for (int ki = 0; ki < 3; ++ki) {
                        for (int kj = 0; kj < 3; ++kj) {
                            int ni = i + ki - 1;
                            int nj = j + kj - 1;
                            if (ni >= 0 && ni < 32 && nj >= 0 && nj < 32) {
                                sum += conv_output1[(f / 2) * 64 * 64 + ni * 64 + nj] * conv2_weights[f][ki * 3 + kj];
                            }
                        }
                    }
                    conv_output2[f * 32 * 32 + i * 32 + j] = sum + conv2_bias[f];
                }
            }
        }
        conv_output2 = batch_norm(conv_output2, bn2_mean, bn2_var, bn2_gamma, bn2_beta);
        conv_output2 = relu(conv_output2);
        conv_output2 = max_pool2d(conv_output2, 32, 2);

        // Fully connected layer 1
        std::vector<float> fc1_output(64, 0.0f);
        for (int i = 0; i < 64; ++i) {
            float sum = 0.0f;
            for (int j = 0; j < 16 * 16 * 16; ++j) {
                sum += conv_output2[j] * fc1_weights[j * 64 + i];
            }
            fc1_output[i] = sum;
        }
        fc1_output = relu(fc1_output);

        // Fully connected layer 2
        std::vector<float> output(NUM_CLASSES, 0.0f);
        for (int i = 0; i < NUM_CLASSES; ++i) {
            float sum = 0.0f;
            for (int j = 0; j < 64; ++j) {
                sum += fc1_output[j] * fc2_weights[j * NUM_CLASSES + i];
            }
            output[i] = sum;
        }
        return output; // Removed softmax
    }

    void save_weights_binary(const std::string& file_path) {
        std::ofstream file(file_path, std::ios::binary);
        if (!file.is_open()) {
            std::cerr << "Error: Could not open file to save weights." << std::endl;
            return;
        }
        save_vector_binary(file, conv1_weights);
        save_vector_binary(file, conv1_bias);
        save_vector_binary(file, bn1_gamma);
        save_vector_binary(file, bn1_beta);
        save_vector_binary(file, bn1_mean);
        save_vector_binary(file, bn1_var);
        save_vector_binary(file, conv2_weights);
        save_vector_binary(file, conv2_bias);
        save_vector_binary(file, bn2_gamma);
        save_vector_binary(file, bn2_beta);
        save_vector_binary(file, bn2_mean);
        save_vector_binary(file, bn2_var);
        save_vector_binary(file, fc1_weights);
        save_vector_binary(file, fc2_weights);
        file.close();
    }

    // Getter and setter for fc1_weights
    std::vector<float>& get_fc1_weights() {
        return fc1_weights;
    }

    void set_fc1_weights(const std::vector<float>& weights) {
        fc1_weights = weights;
    }

    // Getter and setter for fc2_weights
    std::vector<float>& get_fc2_weights() {
        return fc2_weights;
    }

    void set_fc2_weights(const std::vector<float>& weights) {
        fc2_weights = weights;
    }

private:
    std::vector<std::vector<float>> conv1_weights;
    std::vector<float> conv1_bias;
    std::vector<std::vector<float>> conv2_weights;
    std::vector<float> conv2_bias;
    std::vector<float> fc1_weights;
    std::vector<float> fc2_weights;
    std::vector<float> bn1_mean;
    std::vector<float> bn1_var;
    std::vector<float> bn1_gamma;
    std::vector<float> bn1_beta;
    std::vector<float> bn2_mean;
    std::vector<float> bn2_var;
    std::vector<float> bn2_gamma;
    std::vector<float> bn2_beta;

    void initialize_weights() {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> dis(-0.01, 0.01);

        conv1_weights = std::vector<std::vector<float>>(8, std::vector<float>(9));
        for (auto& vec : conv1_weights) {
            for (auto& val : vec) {
                val = dis(gen);
            }
        }
        conv1_bias = std::vector<float>(8, 0.01f);

        conv2_weights = std::vector<std::vector<float>>(16, std::vector<float>(8 * 3 * 3));
        for (auto& vec : conv2_weights) {
            for (auto& val : vec) {
                val = dis(gen);
            }
        }
        conv2_bias = std::vector<float>(16, 0.01f);

        fc1_weights = std::vector<float>(16 * 16 * 16 * 64);
        for (auto& val : fc1_weights) {
            val = dis(gen);
        }
        fc2_weights = std::vector<float>(64 * NUM_CLASSES);
        for (auto& val : fc2_weights) {
            val = dis(gen);
        }

        bn1_mean = std::vector<float>(8, 0.0f);
        bn1_var = std::vector<float>(8, 1.0f);
        bn1_gamma = std::vector<float>(8, 1.0f);
        bn1_beta = std::vector<float>(8, 0.0f);

        bn2_mean = std::vector<float>(16, 0.0f);
        bn2_var = std::vector<float>(16, 1.0f);
        bn2_gamma = std::vector<float>(16, 1.0f);
        bn2_beta = std::vector<float>(16, 0.0f);
    }

    template <typename T>
    void save_vector_binary(std::ofstream& file, const std::vector<T>& vec) {
        file.write(reinterpret_cast<const char*>(vec.data()), vec.size() * sizeof(T));
    }

    void save_vector_binary(std::ofstream& file, const std::vector<std::vector<float>>& v) {
        for (const auto& vec : v) {
            save_vector_binary(file, vec);
        }
    }

    std::vector<float> batch_norm(const std::vector<float>& input, const std::vector<float>& mean, const std::vector<float>& var, const std::vector<float>& gamma, const std::vector<float>& beta) {
        std::vector<float> output(input.size());
        for (size_t i = 0; i < input.size(); ++i) {
            output[i] = gamma[i] * (input[i] - mean[i]) / std::sqrt(var[i] + 1e-5) + beta[i];
        }
        return output;
    }

};

#endif // CNN_MODEL_H