//File này để định nghĩa các hàm kích hoạt mình sẽ sử dụng trong mô hình mạng nơ-ron

#ifndef ACTIVATION_FUNCTIONS_H
#define ACTIVATION_FUNCTIONS_H

#include <vector>
#include <algorithm>
#include <numeric>
#include <cmath>

float relu(float x) { //Hàm này trả về giá trị của x nếu x lớn hơn 0, ngược lại trả về 0
    return x > 0 ? x : 0;
}
                                //Hiểu đơn giản nó là đạo hàm của hàm relu
float relu_derivative(float x) { //sử dụng trong quá trình lan truyền ngược để cập nhật trọng số.
    return x > 0 ? 1 : 0;
}

float sigmoid(float x) { //Hiện chưa sử dụng đến hàm sigmoid nhưng nếu muốn vẫn có thể cho nó vào mô hình
    return 1 / (1 + exp(-x));
}

float sigmoid_derivative(float x) {
    return sigmoid(x) * (1 - sigmoid(x)); //Tương tự , là đạo hàm của sigmoid
}

std::vector<float> softmax(const std::vector<float>& x) {
    std::vector<float> result(x.size());
    float max_val = *std::max_element(x.begin(), x.end());
    float sum = 0.0;

    for (size_t i = 0; i < x.size(); ++i) {
        result[i] = std::exp(x[i] - max_val);
        sum += result[i];
    }

    for (size_t i = 0; i < x.size(); ++i) {
        result[i] /= sum;
    }

    return result;
}

#endif // ACTIVATION_FUNCTIONS_H