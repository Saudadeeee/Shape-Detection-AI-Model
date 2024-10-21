#ifndef MLP_H
#define MLP_H

#include <vector>

class MLP {
public:
    std::vector<std::vector<std::vector<float>>> weights_hidden; // Trọng số của các lớp ẩn
    std::vector<std::vector<float>> biases_hidden; // Bias của các lớp ẩn
    std::vector<std::vector<float>> hidden_layers; // Nơ-ron ở các lớp ẩn
    std::vector<float> weights_output; // Trọng số của lớp đầu ra
    float bias_output; // Bias của lớp đầu ra
    float learning_rate;

    MLP(const std::vector<int>& layer_sizes, float lr);

    // Dự đoán đầu ra
    float predict(const std::vector<float>& inputs);

    // Huấn luyện mô hình
    void train(const std::vector<std::vector<float>>& training_data, const std::vector<float>& labels, int epochs);
};

#endif // MLP_H