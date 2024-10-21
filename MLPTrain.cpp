// Định nghĩa phương thức train của MLP
// Thực hiện quá trình train bằng sử dụng thuật toán lan truyền ngược


#include "MLPTrain.h"
#include "ActivationFunctions.h"//sử dụng các hàm kích hoạt và đạo hàm của chúng


//Hàm train của MLP
//                                                    vector 2 chiều ,  vector chứa các nhãn tương ứng, Số lượng epochs
void MLP::train(const std::vector<std::vector<float>>& training_data , const std::vector<float>& labels, int epochs) {
    //Vòng lặp qua các epochs.
    for (int epoch = 0; epoch < epochs; ++epoch) {
        //Vòng lặp qua các mẫu dữ liệu huấn luyện.
        for (int i = 0; i < training_data.size(); ++i) {
            // Bước 1: Dự đoán
            std::vector<std::vector<float>> layer_outputs(hidden_layers.size() + 1);//Khởi tạo một vector để lưu trữ đầu ra của các lớp. Kích thước của vector này bằng với số lớp của mô hình + 1.
            layer_outputs[0] = training_data[i];//Gán đầu vào của mẫu dữ liệu hiện tại cho lớp đầu tiên.

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