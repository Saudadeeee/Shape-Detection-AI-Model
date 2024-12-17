#ifndef CNN_MODEL_H
#define CNN_MODEL_H

#include <vector>
#include <string>
#include <fstream>
#include <cassert>
#include <algorithm>
#include <iostream>
#include <cmath>


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

    void train(const std::vector<Image>& train_data, int epochs, float learning_rate) {
        std::cout << "Starting training with " << train_data.size() << " images." << std::endl;
        for (int epoch = 0; epoch < epochs; ++epoch) {
            for (const auto& img : train_data) {
                auto output = forward(img.data);
                backward(img.data, output, img.label, learning_rate);
            }
            std::cout << "Epoch " << epoch + 1 << " completed." << std::endl;
        }
    }

    int predict(const Image& img) {
        auto output = forward(img.data);
        int prediction = std::distance(output.begin(), std::max_element(output.begin(), output.end()));
        std::cout << "Predicted label: " << prediction << ", Actual label: " << img.label << std::endl;
        return prediction;
    }

private:
    std::vector<std::vector<float>> conv1_weights;
    std::vector<std::vector<float>> conv2_weights; 
    std::vector<float> fc1_weights;
    std::vector<float> fc2_weights;
    std::vector<float> fc3_weights; 

    void initialize_weights() {
        conv1_weights = std::vector<std::vector<float>>(32, std::vector<float>(9, 0.01f));
        conv2_weights = std::vector<std::vector<float>>(64, std::vector<float>(32 * 3 * 3, 0.01f));
        fc1_weights = std::vector<float>(32 * 31 * 31, 0.01f);
        fc2_weights = std::vector<float>(64 * 29 * 29, 0.01f); 
        fc3_weights = std::vector<float>(128, 0.01f); 
    }

    std::vector<float> forward(const float input[IMAGE_SIZE][IMAGE_SIZE]) {
        std::vector<float> conv_output(32 * 31 * 31, 0.0f);
        for (int f = 0; f < 32; ++f) {
            for (int i = 0; i < 31; ++i) {
                for (int j = 0; j < 31; ++j) {
                    float sum = 0.0f;
                    for (int ki = 0; ki < 3; ++ki) {
                        for (int kj = 0; kj < 3; ++kj) {
                            sum += input[i + ki][j + kj] * conv1_weights[f][ki * 3 + kj];
                        }
                    }
                    conv_output[f * 31 * 31 + i * 31 + j] = std::max(0.0f, sum);
                }
            }
        }

        std::vector<float> conv2_output(64 * 29 * 29, 0.0f);
        for (int f = 0; f < 64; ++f) {
            for (int i = 0; i < 29; ++i) {
                for (int j = 0; j < 29; ++j) {
                    float sum = 0.0f;
                    for (int ki = 0; ki < 3; ++ki) {
                        for (int kj = 0; kj < 3; ++kj) {
                            sum += conv_output[(i + ki) * 31 + (j + kj)] * conv2_weights[f][ki * 3 + kj];
                        }
                    }
                    conv2_output[f * 29 * 29 + i * 29 + j] = std::max(0.0f, sum);
                }
            }
        }

        std::vector<float> fc1_output(64, 0.0f);
        for (int i = 0; i < 64; ++i) {
            float sum = 0.0f;
            for (int j = 0; j < 32 * 31 * 31; ++j) {
                sum += conv_output[j] * fc1_weights[j];
            }
            fc1_output[i] = std::max(0.0f, sum);
        }

        std::vector<float> fc2_output(128, 0.0f);
        for (int i = 0; i < 128; ++i) {
            float sum = 0.0f;
            for (int j = 0; j < 64 * 29 * 29; ++j) {
                sum += conv2_output[j] * fc2_weights[j];
            }
            fc2_output[i] = std::max(0.0f, sum);
        }

        std::vector<float> output(NUM_CLASSES, 0.0f);
        for (int i = 0; i < NUM_CLASSES; ++i) {
            float sum = 0.0f;
            for (int j = 0; j < 128; ++j) {
                sum += fc2_output[j] * fc3_weights[j];
            }
            output[i] = sum;
        }

        return output;
    }

    void backward(const float input[IMAGE_SIZE][IMAGE_SIZE], const std::vector<float>& output, int label, float learning_rate) {
        std::vector<float> output_grad(NUM_CLASSES, 0.0f);
        output_grad[label] = 1.0f - output[label];

        std::vector<float> fc3_grad(128, 0.0f);
        for (int i = 0; i < NUM_CLASSES; ++i) {
            for (int j = 0; j < 128; ++j) {
                fc3_grad[j] += output_grad[i];
                fc3_weights[j] += learning_rate * output_grad[i];
            }
        }

        std::vector<float> fc2_grad(64 * 29 * 29, 0.0f);
        for (int i = 0; i < 128; ++i) {
            for (int j = 0; j < 64 * 29 * 29; ++j) {
                fc2_grad[j] += fc3_grad[i];
                fc2_weights[j] += learning_rate * fc3_grad[i];
            }
        }

        std::vector<float> conv2_grad(64 * 29 * 29, 0.0f);
        for (int f = 0; f < 64; ++f) {
            for (int i = 0; i < 29; ++i) {
                for (int j = 0; j < 29; ++j) {
                    for (int ki = 0; ki < 3; ++ki) {
                        for (int kj = 0; kj < 3; ++kj) {
                            conv2_grad[f * 29 * 29 + i * 29 + j] += fc2_grad[f * 29 * 29 + i * 29 + j];
                            conv2_weights[f][ki * 3 + kj] += learning_rate * conv2_grad[f * 29 * 29 + i * 29 + j];
                        }
                    }
                }
            }
        }

        std::vector<float> fc1_grad(32 * 31 * 31, 0.0f);
        for (int i = 0; i < 64; ++i) {
            for (int j = 0; j < 32 * 31 * 31; ++j) {
                fc1_grad[j] += conv2_grad[i];
                fc1_weights[j] += learning_rate * conv2_grad[i];
            }
        }
        std::vector<float> conv1_grad(32 * 31 * 31, 0.0f);
        for (int f = 0; f < 32; ++f) {
            for (int i = 0; i < 31; ++i) {
                for (int j = 0; j < 31; ++j) {
                    for (int ki = 0; ki < 3; ++ki) {
                        for (int kj = 0; kj < 3; ++kj) {
                            conv1_grad[f * 31 * 31 + i * 31 + j] += fc1_grad[f * 31 * 31 + i * 31 + j];
                            conv1_weights[f][ki * 3 + kj] += learning_rate * conv1_grad[f * 31 * 31 + i * 31 + j];
                        }
                    }
                }
            }
        }
    }
};

#endif // CNN_MODEL_H