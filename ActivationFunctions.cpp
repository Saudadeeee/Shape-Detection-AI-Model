// định nghĩa các hàm kích hoạt và đạo hàm của chúng
// được sử dụng trong mô hình mạng nơ-ron theo toán

#include "ActivationFunctions.h"
#include <algorithm>
#include <numeric>
#include <vector>

float relu(float x) {
    return x > 0 ? x : 0;//Hàm này trả về giá trị của x nếu x lớn hơn 0, ngược lại trả về 0
}

float relu_derivative(float x) {
    return x > 0 ? 1 : 0;//sử dụng trong quá trình lan truyền ngược để cập nhật trọng số.
}

float sigmoid(float x) {
    return 1 / (1 + exp(-x));// trả về giá trị trong khoảng (0, 1) và được tính bằng công thức 1 / (1 + exp(-x))
}

float sigmoid_derivative(float x) {
    return sigmoid(x) * (1 - sigmoid(x));//Đạo hàm này cũng được sử dụng trong quá trình lan truyền ngược để cập nhật trọng số.
}

// Hàm softmax
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
