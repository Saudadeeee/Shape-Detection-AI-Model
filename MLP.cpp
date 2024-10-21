#include "MLP.h"
#include "ActivationFunctions.h"

MLP::MLP(int input_size, int hidden_size, float lr) {
    // Khởi tạo trọng số và bias
    weights_hidden.resize(input_size, std::vector<float>(hidden_size, 0.1));  // 0.1 là giá trị khởi tạo trọng số
    weights_output.resize(hidden_size, 0.1);
    hidden_layer.resize(hidden_size, 0.0);
    bias_hidden = 0.1;
    bias_output = 0.1;
    learning_rate = lr;
}

float MLP::predict(const std::vector<float>& inputs) {
    // Tính toán các nơ-ron ở lớp ẩn với ReLU
    for (int i = 0; i < hidden_layer.size(); i++) {
        float sum = bias_hidden;
        for (int j = 0; j < inputs.size(); j++) {
            sum += inputs[j] * weights_hidden[j][i];
        }
        hidden_layer[i] = relu(sum);
    }

    // Tính toán lớp đầu ra với Sigmoid
    float output_sum = bias_output;
    for (int i = 0; i < hidden_layer.size(); i++) {
        output_sum += hidden_layer[i] * weights_output[i];
    }
    return sigmoid(output_sum);
}

void MLP::train(const std::vector<std::vector<float>>& training_data, const std::vector<float>& labels, int epochs) {
    for (int epoch = 0; epoch < epochs; epoch++) {
        for (int i = 0; i < training_data.size(); i++) {
            // Bước 1: Dự đoán
            float prediction = predict(training_data[i]);

            // Bước 2: Tính toán lỗi
            float error_output = labels[i] - prediction;

            // Bước 3: Điều chỉnh trọng số của lớp đầu ra
            for (int j = 0; j < hidden_layer.size(); j++) {
                float delta_output = error_output * sigmoid_derivative(prediction);
                weights_output[j] += learning_rate * delta_output * hidden_layer[j];
            }
            bias_output += learning_rate * error_output * sigmoid_derivative(prediction);

            // Bước 4: Điều chỉnh trọng số của lớp ẩn
            for (int j = 0; j < hidden_layer.size(); j++) {
                float error_hidden = error_output * sigmoid_derivative(prediction) * weights_output[j];
                for (int k = 0; k < training_data[i].size(); k++) {
                    weights_hidden[k][j] += learning_rate * error_hidden * relu_derivative(hidden_layer[j]) * training_data[i][k];
                }
            }
            bias_hidden += learning_rate * error_output * sigmoid_derivative(prediction);
        }
    }
}