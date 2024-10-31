// Định nghĩa phương thức train của MLP
// Thực hiện quá trình train bằng sử dụng thuật toán lan truyền ngược
#include "MLPTrain.h"
#include "ActivationFunctions.h"//sử dụng các hàm kích hoạt và đạo hàm của chúng
#include <algorithm> // For std::shuffle
#include <random> // For std::default_random_engine
#include <iostream>
#include <numeric> // For std::iota

//Hàm train của MLP
//                                                    vector 2 chiều ,  vector chứa các nhãn tương ứng, Số lượng epochs
void MLP::train(const std::vector<std::vector<float>>& training_data , const std::vector<float>& labels, int epochs) {
    std::vector<int> indices(training_data.size());
    std::iota(indices.begin(), indices.end(), 0); // Fill indices with 0, 1, ..., training_data.size()-1
    std::default_random_engine engine(std::random_device{}());

    //Vòng lặp qua các epochs.
    for (int epoch = 0; epoch < epochs; ++epoch) {
        std::shuffle(indices.begin(), indices.end(), engine); // Shuffle the indices

        float total_loss = 0.0; // Initialize total loss for the epoch
 
        //Vòng lặp qua các mẫu dữ liệu huấn luyện.
        for (int idx : indices) {
            const auto& input = training_data[idx];
            float label = labels[idx];

            // Forward pass
            std::vector<std::vector<float>> layer_outputs(hidden_layers.size() + 1);//Khởi tạo một vector để lưu trữ đầu ra của các lớp. Kích thước của vector này bằng với số lớp của mô hình + 1.
            layer_outputs[0] = input;//Gán đầu vào của mẫu dữ liệu hiện tại cho lớp đầu tiên.
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