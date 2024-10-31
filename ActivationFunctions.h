//khai báo các hàm kích hoạt và đạo hàm của chúng được sử dụng trong mô hình mạng nơ-ron theo toán
#ifndef ACTIVATION_FUNCTIONS_H
#define ACTIVATION_FUNCTIONS_H

#include <cmath>
#include <vector>
#include <algorithm> // Add this line to include std::max_element

// Hàm kích hoạt ReLU
float relu(float x);

// Đạo hàm của ReLU
float relu_derivative(float x);

// Hàm kích hoạt Sigmoid
float sigmoid(float x);

// Đạo hàm của Sigmoid
float sigmoid_derivative(float x);
//Tùy xem có muốn dùng thêm hàm kích hoạt không không thì 2 hàm dưới này có thể bỏ đi
// // Hàm kích hoạt Tanh
// float tanh(float x);

// // Đạo hàm của Tanh
// float tanh_derivative(float x);
//Hàm kích hoạt softmax
std::vector<float> softmax(const std::vector<float>& x);

#endif // ACTIVATION_FUNCTIONS_H