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
    void train(const std::vector<std::vector<float>>& training_data, const std::vector<float>& labels, int epochs);
    float predict(const std::vector<float>& inputs) const;

private:
    std::vector<std::vector<std::vector<float>>> weights_hidden;
    std::vector<std::vector<float>> biases_hidden;
    std::vector<std::vector<float>> hidden_layers;
    std::vector<std::vector<float>> weights_output;
    std::vector<float> bias_output;
    float learning_rate;
};

// Implementation of MLP methods

MLP::MLP(const std::vector<int>& layer_sizes, float lr) {
    std::srand(std::time(0)); // Khởi tạo seed cho số ngẫu nhiên
    int num_layers = layer_sizes.size();
    weights_hidden.resize(num_layers - 1); //Thay đổi kích thước của vector weights_hidden để chứa trọng số của các lớp ẩn.
    biases_hidden.resize(num_layers - 1);//Thay đổi kích thước của vector biases_hidden để chứa bias của các lớp ẩn.
    hidden_layers.resize(num_layers - 1); //Thay đổi kích thước của vector hidden_layers để chứa giá trị của các nơ-ron ở các lớp ẩn.
    //Lặp qua các lớp ẩn để khởi tạo trọng số và bias
    for (int i = 0; i < num_layers - 1; ++i) {
        weights_hidden[i].resize(layer_sizes[i], std::vector<float>(layer_sizes[i + 1]));//Thay đổi kích thước của ma trận trọng số cho lớp ẩn thứ i
        biases_hidden[i].resize(layer_sizes[i + 1]); //Thay đổi kích thước của vector bias cho lớp ẩn thứ i
        hidden_layers[i].resize(layer_sizes[i + 1], 0.0); //Thay đổi kích thước của vector hidden_layers cho lớp ẩn thứ i
        //Vòng lặp qua các nơ-ron của lớp ẩn thứ i
        for (int j = 0; j < layer_sizes[i]; ++j) {//Vòng lặp qua các nơ-ron của lớp thứ i
            for (int k = 0; k < layer_sizes[i + 1]; ++k) {//Vòng lặp qua các nơ-ron của lớp tiếp theo.
                weights_hidden[i][j][k] = static_cast<float>(std::rand()) / RAND_MAX - 0.5f; // Khởi tạo ngẫu nhiên trong khoảng [-0.5, 0.5]
            //Mục đích của cái static_cast này là để chuyển đổi kiểu dữ liệu từ int sang float
            //Nói chung là để ép kiểu
            }
        }
        // Vòng lặp qua các nơ-ron của lớp tiếp theo.
        for (int j = 0; j < layer_sizes[i + 1]; ++j) {
            biases_hidden[i][j] = static_cast<float>(std::rand()) / RAND_MAX - 0.5f; // Khởi tạo ngẫu nhiên trong khoảng [-0.5, 0.5]
        }
    }
    //Khởi tạo trọng số và bias cho lớp đầu ra
    weights_output.resize(layer_sizes.back(), std::vector<float>(3)); // 3 lớp đầu ra cho 3 hình dạng
    for (int i = 0; i < layer_sizes.back(); ++i) {
        for (int j = 0; j < 3; ++j) {
            weights_output[i][j] = static_cast<float>(std::rand()) / RAND_MAX - 0.5f; // Khởi tạo ngẫu nhiên trong khoảng [-0.5, 0.5]
        }
    }
    bias_output.resize(3);
    for (int j = 0; j < 3; ++j) {
        bias_output[j] = static_cast<float>(std::rand()) / RAND_MAX - 0.5f; // Khởi tạo ngẫu nhiên trong khoảng [-0.5, 0.5]
    }
    learning_rate = lr;
}

void MLP::train(const std::vector<std::vector<float>>& training_data , const std::vector<float>& labels, int epochs) {
    std::vector<int> indices(training_data.size());
    std::iota(indices.begin(), indices.end(), 0); // Fill indices with 0, 1, ..., training_data.size()-1
    std::default_random_engine engine(std::random_device{}());

    for (int epoch = 0; epoch < epochs; ++epoch) {
        std::shuffle(indices.begin(), indices.end(), engine); // Shuffle the indices

        float total_loss = 0.0; // Initialize total loss for the epoch
 
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
        }

        if (epoch % 100 == 0) { // Print loss every 100 epochs
            std::cout << "Epoch " << epoch << ", Loss: " << total_loss / training_data.size() << std::endl;
        }
    }
}

float MLP::predict(const std::vector<float>& inputs) const {
    std::vector<float> current_layer = inputs;
    std::cout << "Input size: " << inputs.size() << std::endl;
    for (int i = 0; i < hidden_layers.size(); ++i) {
        std::cout << "Layer " << i << " size: " << hidden_layers[i].size() << std::endl;
        std::vector<float> next_layer(hidden_layers[i].size(), 0.0);
        for (int j = 0; j < hidden_layers[i].size(); ++j) {
            float sum = biases_hidden[i][j];
            for (int k = 0; k < current_layer.size(); ++k) {
                if (k >= weights_hidden[i].size() || j >= weights_hidden[i][k].size()) {
                    std::cerr << "Index out of bounds: weights_hidden[" << i << "][" << k << "][" << j << "]" << std::endl;
                    return -1; // Return an error value
                }
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

    return std::distance(output_layer.begin(), std::max_element(output_layer.begin(), output_layer.end()));
}

#endif // MLP_H