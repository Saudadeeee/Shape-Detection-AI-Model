// định nghĩa các hàm kích hoạt và đạo hàm của chúng
// được sử dụng trong mô hình mạng nơ-ron theo toán

#include "ActivationFunctions.h"

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

// float tanh(float x) {
//     return std::tanh(x);
// }

// float tanh_derivative(float x) {
//     return 1 - std::tanh(x) * std::tanh(x);
// }