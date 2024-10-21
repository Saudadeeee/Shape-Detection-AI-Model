#include "MLP.h"
#include "ActivationFunctions.h"

MLP::MLP(const std::vector<int>& layer_sizes, float lr) {
    int num_layers = layer_sizes.size();
    weights_hidden.resize(num_layers - 1);
    biases_hidden.resize(num_layers - 1);
    hidden_layers.resize(num_layers - 1);

    for (int i = 0; i < num_layers - 1; ++i) {
        weights_hidden[i].resize(layer_sizes[i], std::vector<float>(layer_sizes[i + 1], 0.1));
        biases_hidden[i].resize(layer_sizes[i + 1], 0.1);
        hidden_layers[i].resize(layer_sizes[i + 1], 0.0);
    }

    weights_output.resize(layer_sizes.back(), 0.1);
    bias_output = 0.1;
    learning_rate = lr;
}

float MLP::predict(const std::vector<float>& inputs) {
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

    float output_sum = bias_output;
    for (int i = 0; i < current_layer.size(); ++i) {
        output_sum += current_layer[i] * weights_output[i];
    }
    return sigmoid(output_sum);
}

void MLP::train(const std::vector<std::vector<float>>& training_data, const std::vector<float>& labels, int epochs) {
    for (int epoch = 0; epoch < epochs; ++epoch) {
        for (int i = 0; i < training_data.size(); ++i) {
            // Bước 1: Dự đoán
            std::vector<std::vector<float>> layer_outputs(hidden_layers.size() + 1);
            layer_outputs[0] = training_data[i];

            for (int j = 0; j < hidden_layers.size(); ++j) {
                std::vector<float> next_layer(hidden_layers[j].size(), 0.0);
                for (int k = 0; k < hidden_layers[j].size(); ++k) {
                    float sum = biases_hidden[j][k];
                    for (int l = 0; l < layer_outputs[j].size(); ++l) {
                        sum += layer_outputs[j][l] * weights_hidden[j][l][k];
                    }
                    next_layer[k] = relu(sum);
                }
                layer_outputs[j + 1] = next_layer;
            }

            float prediction = predict(training_data[i]);
            float error_output = labels[i] - prediction;

            // Bước 2: Điều chỉnh trọng số của lớp đầu ra
            for (int j = 0; j < weights_output.size(); ++j) {
                float delta_output = error_output * sigmoid_derivative(prediction);
                weights_output[j] += learning_rate * delta_output * layer_outputs.back()[j];
            }
            bias_output += learning_rate * error_output * sigmoid_derivative(prediction);

            // Bước 3: Điều chỉnh trọng số của các lớp ẩn
            std::vector<float> error_hidden_next(weights_output.size(), 0.0);
            for (int j = hidden_layers.size() - 1; j >= 0; --j) {
                std::vector<float> error_hidden(hidden_layers[j].size(), 0.0);
                for (int k = 0; k < hidden_layers[j].size(); ++k) {
                    float error = 0.0;
                    if (j == hidden_layers.size() - 1) {
                        error = error_output * sigmoid_derivative(prediction) * weights_output[k];
                    } else {
                        for (int l = 0; l < hidden_layers[j + 1].size(); ++l) {
                            error += error_hidden_next[l] * weights_hidden[j + 1][k][l];
                        }
                    }
                    error_hidden[k] = error * relu_derivative(layer_outputs[j + 1][k]);
                    for (int l = 0; l < layer_outputs[j].size(); ++l) {
                        weights_hidden[j][l][k] += learning_rate * error_hidden[k] * layer_outputs[j][l];
                    }
                    biases_hidden[j][k] += learning_rate * error_hidden[k];
                }
                error_hidden_next = error_hidden;
            }
        }
    }
}