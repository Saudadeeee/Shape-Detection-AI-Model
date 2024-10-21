#ifndef TRAIN_H
#define TRAIN_H

#include "MLP.h"
#include <cmath>  // Để dùng sigmoid_derivative

float relu_derivative(float x) {
    return x > 0 ? 1 : 0;
}

float sigmoid_derivative(float x) {
    return x * (1 - x);
}

void MLP::train(const vector<vector<float>>& training_data, const vector<float>& labels, int epochs) {
    for (int epoch = 0; epoch < epochs; epoch++) {
        for (int i = 0; i < training_data.size(); i++) {
            float prediction = predict(training_data[i]);

            float error_output = labels[i] - prediction;

            for (int j = 0; j < hidden_layer2.size(); j++) {
                float delta_output = error_output * sigmoid_derivative(prediction);
                weights_hidden2_output[j] += learning_rate * delta_output * hidden_layer2[j];
            }
            bias_output += learning_rate * error_output * sigmoid_derivative(prediction);

            for (int j = 0; j < hidden_layer2.size(); j++) {
                float error_hidden2 = error_output * sigmoid_derivative(prediction) * weights_hidden2_output[j];
                for (int k = 0; k < hidden_layer1.size(); k++) {
                    weights_hidden1_hidden2[k][j] += learning_rate * error_hidden2 * relu_derivative(hidden_layer2[j]) * hidden_layer1[k];
                }
            }
            bias_hidden2 += learning_rate * error_output * sigmoid_derivative(prediction);

            for (int j = 0; j < hidden_layer1.size(); j++) {
                float error_hidden1 = 0.0;
                for (int k = 0; k < hidden_layer2.size(); k++) {
                    error_hidden1 += weights_hidden1_hidden2[j][k] * relu_derivative(hidden_layer2[k]);
                }
                for (int k = 0; k < training_data[i].size(); k++) {
                    weights_input_hidden1[k][j] += learning_rate * error_hidden1 * relu_derivative(hidden_layer1[j]) * training_data[i][k];
                }
            }
            bias_hidden1 += learning_rate * error_output * sigmoid_derivative(prediction);
        }
    }
}

#endif
