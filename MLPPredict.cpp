

#include "MLPPredict.h"
#include "ActivationFunctions.h"
#include <algorithm>
// Dự đoán đầu ra
float MLP::predict(const std::vector<float>& inputs) const {
    std::vector<float> current_layer = inputs;
    // Duyệt qua các lớp ẩn
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

    for (int j = 0; j < 3; ++j) {
        output_layer[j] = sigmoid(output_layer[j]);
    }

    // Tìm nhãn có giá trị lớn nhất
    return std::distance(output_layer.begin(), std::max_element(output_layer.begin(), output_layer.end()));
}