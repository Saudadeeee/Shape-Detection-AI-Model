// Biên dịch các tệp bằng g++ -o NewModel NewModel.cpp MLP.cpp MLPTrain.cpp MLPPredict.cpp ActivationFunctions.cpp ImageProcessor.cpp
// Sau đó chạy chương trình bằng g++ ./NewModel
//Tại vì t lười dùng makefile nên là phải làm thế này
//Nếu dùng makefile thì chỉ cần gõ make và ./NewModel là xong
//Cái này nó rất hay bị lỗi nên là nếu lỗi thì cứ biên dịch lại mấy lần là được
//Nếu không được thì hỏi t

#include <iostream>
#include <vector>
#include "MLP.h"
#include "ActivationFunctions.h"
#include "ImageProcessor.h"
#include "MLPTrain.h"
#include "MLPPredict.h"

using namespace std;

int main() {
    // Đọc ảnh và chuyển đổi thành dữ liệu đầu vào
    vector<vector<vector<int>>> images = ImageProcessor::readImages("image.txt");

    // Kiểm tra số lượng ảnh đọc được
    if (images.empty()) {
        cerr << "Không đọc được ảnh từ file!" << endl;
        return 1;
    }

    // Chuyển đổi ảnh thành vector 1 chiều
    vector<vector<float>> training_data;
    for (const auto& image : images) {
        vector<float> input;
        for (const auto& row : image) {
            for (int pixel : row) {
                input.push_back(static_cast<float>(pixel));
            }
        }
        training_data.push_back(input);
    }

    // Nhãn đầu ra tương ứng
    vector<float> labels = {
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, // 20 mẫu hình vuông
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, // 20 mẫu hình tròn
        2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2  // 20 mẫu hình tam giác
    };

    // Kiểm tra số lượng mẫu và nhãn
    if (training_data.size() != labels.size()) {
        cerr << "Số lượng mẫu và nhãn không khớp!" << endl;
        return 1;
    }

    // Khởi tạo mô hình MLP với số lượng đầu vào tương ứng với kích thước ảnh, 10 lớp ẩn (mỗi lớp 10 nơ-ron), tốc độ học 0.1
    vector<int> layer_sizes = {static_cast<int>(training_data[0].size()), 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 3}; // Kích thước đầu vào, 10 lớp ẩn, 3 output
    MLP mlp(layer_sizes, 0.1);

    // Huấn luyện mô hình trong 10000 epochs
    mlp.train(training_data, labels, 10000);

    // Kiểm tra dự đoán
    for (size_t i = 0; i < training_data.size(); ++i) {
        int prediction = mlp.predict(training_data[i]);
        cout << "Du doan cho anh " << i + 1 << ": " << prediction << " (";
        if (prediction == 0) {
            cout << "Hinh vuong";
        } else if (prediction == 1) {
            cout << "Hinh tron";
        } else {
            cout << "Hinh tam giac";
        }
        cout << ")" << endl;
    }

    return 0;
}