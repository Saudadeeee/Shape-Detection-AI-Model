#ifndef MLP_H
#define MLP_H

#include <vector>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <algorithm>
#include <numeric>
#include <random>
#include "ActivationFunctions.h"

class MLP {
public:
    MLP(const std::vector<int>& layer_sizes, float lr);
    void train(const std::vector<std::vector<float>>& training_data, const std::vector<float>& labels, int epochs, const std::vector<std::vector<float>>& validation_data, const std::vector<float>& validation_labels);
    std::vector<float> predict(const std::vector<float>& inputs) const;
    void setLearningRate(float lr); // Add method to set learning rate

private:
    std::vector<std::vector<std::vector<float>>> weights_hidden;
    std::vector<std::vector<float>> biases_hidden;
    std::vector<std::vector<float>> hidden_layers;
    std::vector<std::vector<float>> weights_output;
    std::vector<float> bias_output;
    float learning_rate;
    float l2_lambda = 0.01; // L2 regularization parameter
};

// Implementation of MLP methods

MLP::MLP(const std::vector<int>& layer_sizes, float lr) {
    std::srand(std::time(0)); // Khởi tạo seed cho số ngẫu nhiên
    int num_layers = layer_sizes.size();
    weights_hidden.resize(num_layers - 1);
    biases_hidden.resize(num_layers - 1);
    hidden_layers.resize(num_layers - 1);
    for (int i = 0; i < num_layers - 1; ++i) {
        weights_hidden[i].resize(layer_sizes[i], std::vector<float>(layer_sizes[i + 1]));
        biases_hidden[i].resize(layer_sizes[i + 1]);
        hidden_layers[i].resize(layer_sizes[i + 1], 0.0);
        for (int j = 0; j < layer_sizes[i]; ++j) {
            for (int k = 0; k < layer_sizes[i + 1]; ++k) {
                weights_hidden[i][j][k] = static_cast<float>(std::rand()) / RAND_MAX * 0.01f; // Smaller weight initialization
            }
        }
        for (int j = 0; j < layer_sizes[i + 1]; ++j) {
            biases_hidden[i][j] = static_cast<float>(std::rand()) / RAND_MAX * 0.01f; // Smaller bias initialization
        }
    }
    weights_output.resize(layer_sizes.back(), std::vector<float>(3));
    for (int i = 0; i < layer_sizes.back(); ++i) {
        for (int j = 0; j < 3; ++j) {
            weights_output[i][j] = static_cast<float>(std::rand()) / RAND_MAX * 0.01f; // Smaller weight initialization
        }
    }
    bias_output.resize(3);
    for (int j = 0; j < 3; ++j) {
        bias_output[j] = static_cast<float>(std::rand()) / RAND_MAX * 0.01f; // Smaller bias initialization
    }
    learning_rate = lr;
}

void MLP::setLearningRate(float lr) {
    learning_rate = lr;
}

void MLP::train(const std::vector<std::vector<float>>& training_data , const std::vector<float>& labels, int epochs, const std::vector<std::vector<float>>& validation_data, const std::vector<float>& validation_labels) {
    std::vector<int> indices(training_data.size());
    std::iota(indices.begin(), indices.end(), 0); // Fill indices with 0, 1, ..., training_data.size()-1
    std::default_random_engine engine(std::random_device{}());

    for (int epoch = 0; epoch < epochs; ++epoch) {
        std::shuffle(indices.begin(), indices.end(), engine); // Shuffle the indices

        float total_loss = 0.0; // Initialize total loss for the epoch

        // Decay learning rate
        if (epoch % 1000 == 0 && epoch != 0) {
            setLearningRate(learning_rate * 0.9); // Decay learning rate by 10% every 1000 epochs
        }

        for (int idx : indices) {
            const auto& input = training_data[idx];
            float label = labels[idx];
    
            // Forward pass
            std::vector<std::vector<float>> layer_outputs(hidden_layers.size() + 1);
            layer_outputs[0] = input;
            for (int j = 0; j < hidden_layers.size(); ++j) {
                std::vector<float> next_layer(hidden_layers[j].size(), 0.0);
                for (int k = 0; k < hidden_layers[j].size(); ++k) {
                    float sum = biases_hidden[j][k];
                    for (int l = 0; l < layer_outputs[j].size(); ++l) {
                          if (l >= weights_hidden[j].size() || k >= weights_hidden[j][l].size()) {
                            std::cerr << "Index out of bounds: weights_hidden[" << j << "][" << l << "][" << k << "]" << std::endl;
                            return;
                        }
                        sum += layer_outputs[j][l] * weights_hidden[j][l][k];
                    }
                    next_layer[k] = relu(sum);
                }
                layer_outputs[j + 1] = next_layer;
            }

            std::vector<float> output_layer(3, 0.0); // 3 lớp đầu ra cho 3 hình dạng
            for (int j = 0; j < layer_outputs.back().size(); ++j) {
                for (int k = 0; k < 3; ++k) {
                    output_layer[k] += layer_outputs.back()[j] * weights_output[j][k];
                }
            }

            output_layer = softmax(output_layer);

            // Calculate error
            std::vector<float> target_output(3, 0.0);
            target_output[label] = 1.0;
            std::vector<float> error_output(3, 0.0);
            for (int k = 0; k < 3; ++k) {
                error_output[k] = target_output[k] - output_layer[k];
                total_loss += error_output[k] * error_output[k]; // Accumulate loss
            }

            // Backpropagation
            for (int j = 0; j < layer_outputs.back().size(); ++j) {
                for (int k = 0; k < 3; ++k) {
                    float delta_output = error_output[k] * output_layer[k] * (1 - output_layer[k]);
                    weights_output[j][k] += learning_rate * delta_output * layer_outputs.back()[j];
                }
            }
            for (int k = 0; k < 3; ++k) {
                bias_output[k] += learning_rate * error_output[k] * output_layer[k] * (1 - output_layer[k]);
            }

            std::vector<std::vector<float>> error_hidden_next(hidden_layers.size(), std::vector<float>(hidden_layers.back().size(), 0.0));
            for (int j = hidden_layers.size() - 1; j >= 0; --j) {
                std::vector<float> error_hidden(hidden_layers[j].size(), 0.0);
                for (int k = 0; k < hidden_layers[j].size(); ++k) {
                    float error = 0.0;
                    if (j == hidden_layers.size() - 1) {
                        for (int l = 0; l < 3; ++l) {
                            error += error_output[l] * weights_output[k][l];
                        }
                    } else {
                        for (int l = 0; l < hidden_layers[j + 1].size(); ++l) {
                            error += error_hidden_next[j + 1][l] * weights_hidden[j + 1][k][l];
                        }
                    }
                    error_hidden[k] = error * relu_derivative(layer_outputs[j + 1][k]);
                    for (int l = 0; l < layer_outputs[j].size(); ++l) {
                        weights_hidden[j][l][k] += learning_rate * error_hidden[k] * layer_outputs[j][l];
                    }
                    biases_hidden[j][k] += learning_rate * error_hidden[k];
                }
                error_hidden_next[j] = error_hidden;
            }

            // Apply L2 regularization
            for (auto& layer : weights_hidden) {
                for (auto& neuron : layer) {
                    for (auto& weight : neuron) {
                        weight -= learning_rate * l2_lambda * weight;
                    }
                }
            }
            for (auto& neuron : weights_output) {
                for (auto& weight : neuron) {
                    weight -= learning_rate * l2_lambda * weight;
                }
            }
        }

        if (epoch % 100 == 0) { // Print loss every 100 epochs
            std::cout << "Epoch " << epoch << ", Loss: " << total_loss / training_data.size() << std::endl;

            // Calculate validation loss
            float validation_loss = 0.0;
            for (size_t i = 0; i < validation_data.size(); ++i) {
                std::vector<float> output_layer = predict(validation_data[i]);
                std::vector<float> target_output(3, 0.0);
                target_output[validation_labels[i]] = 1.0;
                for (int k = 0; k < 3; ++k) {
                    float error = target_output[k] - output_layer[k];
                    validation_loss += error * error;
                }
            }
            std::cout << "Validation Loss: " << validation_loss / validation_data.size() << std::endl;
        }
    }
}

std::vector<float> MLP::predict(const std::vector<float>& inputs) const { // Change return type to vector<float>
    std::vector<float> current_layer = inputs;
    for (int i = 0; i < hidden_layers.size(); ++i) {
        std::vector<float> next_layer(hidden_layers[i].size(), 0.0);
        for (int j = 0; j < hidden_layers[i].size(); ++j) {
            float sum = biases_hidden[i][j];
            for (int k = 0; k < current_layer.size(); ++k) {
                sum += current_layer[k] * weights_hidden[i][k][j];
            }
            next_layer[j] = relu(sum);
        }
        current_layer = next_layer;
    }

    std::vector<float> output_layer(3, 0.0); // 3 lớp đầu ra cho 3 hình dạng
    for (int i = 0; i < current_layer.size(); ++i) {
        for (int j = 0; j < 3; ++j) {
            output_layer[j] += current_layer[i] * weights_output[i][j];
        }
    }

    output_layer = softmax(output_layer);

    return output_layer; // Return the softmax output
}

#endif // MLP_H