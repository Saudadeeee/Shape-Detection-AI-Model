#include <iostream>
#include "MLP.h"
#include "ActivationFunctions.h"

using namespace std;

int main() {
    // Dữ liệu huấn luyện (ví dụ đơn giản)
    vector<vector<float>> training_data = {
        {0, 0}, {0, 1}, {1, 0}, {1, 1}
    };

    // Nhãn đầu ra tương ứng
    vector<float> labels = {0, 1, 1, 0};  // XOR logic gate

    // Khởi tạo mô hình MLP với 2 input, 2 nơ-ron ẩn, tốc độ học 0.1
    MLP mlp(2, 2, 0.1);

    // Huấn luyện mô hình trong 10 epochs
    mlp.train(training_data, labels, 10000);

    // Kiểm tra dự đoán
    for (auto& data : training_data) {
        cout << "Du doan cho (" << data[0] << ", " << data[1] << "): " << mlp.predict(data) << endl;
    }

    return 0;
}