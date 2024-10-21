#ifndef PREDICT_H
#define PREDICT_H

#include "MLP.h"
#include <cmath>  // Để dùng sigmoid

float relu(float x) {
    return x > 0 ? x : 0;
}

float sigmoid(float x) {
    return 1 / (1 + exp(-x));
}

float MLP::predict(const vector<float>& inputs) {
    for (int i = 0; i < hidden_layer1.size(); i++) {
        float sum = bias_hidden1;
        for (int j = 0; j < inputs.size(); j++) {
            sum += inputs[j] * weights_input_hidden1[j][i];
        }
        hidden_layer1[i] = relu(sum);
    }

    for (int i = 0; i < hidden_layer2.size(); i++) {
        float sum = bias_hidden2;
        for (int j = 0; j < hidden_layer1.size(); j++) {
            sum += hidden_layer1[j] * weights_hidden1_hidden2[j][i];
        }
        hidden_layer2[i] = relu(sum);
    }

    float output_sum = bias_output;
    for (int i = 0; i < hidden_layer2.size(); i++) {
        output_sum += hidden_layer2[i] * weights_hidden2_output[i];
    }
    return sigmoid(output_sum);
}

#endif
