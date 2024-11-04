// Biên dịch các tệp bằng g++ -o NewModel NewModel.cpp MLP.cpp MLPTrain.cpp MLPPredict.cpp ActivationFunctions.cpp ImageProcessor.cpp
// Sau đó chạy chương trình bằng ./NewModel
//Tại vì t lười dùng makefile nên là phải làm thế này
//Nếu dùng makefile thì chỉ cần gõ make và ./NewModel là xong
//Cái này nó rất hay bị lỗi nên là nếu lỗi thì cứ biên dịch lại mấy lần là được
//Nếu không được thì hỏi t

#include <iostream>
#include <vector>
#include "MLP.h"
#include "ActivationFunctions.h"
#include "ImageProcessor.h"

using namespace std;

int main() {
    cout << "Starting program..." << endl;
    // Đọc ảnh và chuyển đổi thành dữ liệu đầu vào
    vector<vector<vector<int>>> images = ImageProcessor::readImages("image.txt");
    cout << "Images read successfully." << endl;

    // Kiểm tra số lượng ảnh đọc được
    if (images.empty()) {
        cerr << "Không đọc được ảnh từ file!" << endl;
        return 1;
    }
    cout << "Number of images: " << images.size() << endl;

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
    cout << "Training data prepared." << endl;

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
    cout << "Number of training samples: " << training_data.size() << endl;

    // Khởi tạo mô hình MLP với số lượng đầu vào tương ứng với kích thước ảnh, 5 lớp ẩn (mỗi lớp 5 nơ-ron), tốc độ học 0.01
    vector<int> layer_sizes = {static_cast<int>(training_data[0].size()), 5, 5, 5, 5, 3}; // Kích thước đầu vào, 5 lớp ẩn, 3 output

    // Kiểm tra kích thước đầu vào
    if (training_data[0].size() != layer_sizes[0]) {
        cerr << "Kích thước đầu vào không khớp với mô hình MLP!" << endl;
        return 1;
    }

    MLP mlp(layer_sizes, 0.01);
    cout << "MLP model initialized." << endl;

    // Huấn luyện mô hình trong 10000 epochs
    cout << "Training started." << endl;
    mlp.train(training_data, labels, 10000);
    cout << "Training completed." << endl;

    // Kiểm tra dự đoán
    cout << "Predictions:" << endl;
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