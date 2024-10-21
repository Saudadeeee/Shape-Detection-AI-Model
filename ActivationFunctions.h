#ifndef ACTIVATION_FUNCTIONS_H
#define ACTIVATION_FUNCTIONS_H

#include <cmath>

// Hàm kích hoạt ReLU
float relu(float x);

// Đạo hàm của ReLU
float relu_derivative(float x);

// Hàm kích hoạt Sigmoid
float sigmoid(float x);

// Đạo hàm của Sigmoid
float sigmoid_derivative(float x);

#endif // ACTIVATION_FUNCTIONS_H