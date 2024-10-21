//Khai báo phương thức train của MLP
#ifndef MLP_TRAIN_H
#define MLP_TRAIN_H

#include "MLP.h"

void MLP::train(const std::vector<std::vector<float>>& training_data, const std::vector<float>& labels, int epochs);

#endif // MLP_TRAIN_H