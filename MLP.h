#ifndef MLP_H
#define MLP_H

#include <vector>

class MLP {
public:
    std::vector<std::vector<float>> weights_hidden; // Trọng số của lớp ẩn
    std::vector<float> weights_output;         // Trọng số của lớp đầu ra
    std::vector<float> hidden_layer;           // Nơ-ron ở lớp ẩn
    float bias_hidden, bias_output;       // Bias của lớp ẩn và lớp đầu ra
    float learning_rate;

    MLP(int input_size, int hidden_size, float lr);

    // Dự đoán đầu ra
    float predict(const std::vector<float>& inputs);

    // Huấn luyện mô hình
    void train(const std::vector<std::vector<float>>& training_data, const std::vector<float>& labels, int epochs);
};

#endif // MLP_H