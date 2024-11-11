// Chạy tệp bình thường là được tại t đã rút gọn lại rồi



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

    // Ensure all images have the same size
    size_t image_size = images[0].size() * images[0][0].size();
    for (const auto& image : images) {
        if (image.size() * image[0].size() != image_size) {
            cerr << "Kích thước ảnh không đồng nhất!" << endl;
            return 1;
        }
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
    vector<int> layer_sizes = {static_cast<int>(training_data[0].size()), 10, 10, 10, 10, 10, 3}; // Increase neurons in hidden layers
    
    // Debug print to check dimensions
    cout << "First training sample size: " << training_data[0].size() << endl;
    cout << "Expected input size: " << layer_sizes[0] << endl;

    // Kiểm tra kích thước đầu vào
    if (training_data[0].size() != layer_sizes[0]) {
        cerr << "Kích thước đầu vào không khớp với mô hình MLP!" << endl;
        return 1;
    }

    MLP mlp(layer_sizes, 0.001); // Reduce learning rate
    cout << "MLP model initialized." << endl;

    // Split data into training and validation sets
    size_t validation_size = training_data.size() / 5;
    vector<vector<float>> validation_data(training_data.end() - validation_size, training_data.end());
    vector<float> validation_labels(labels.end() - validation_size, labels.end());
    training_data.resize(training_data.size() - validation_size);
    labels.resize(labels.size() - validation_size);

    // Huấn luyện mô hình trong 10000 epochs
    cout << "Training started." << endl;
    mlp.train(training_data, labels, 50000, validation_data, validation_labels); // Increase number of epochs and pass validation data to train method
    cout << "Training completed." << endl;

    // Kiểm tra dự đoán
    cout << "Predictions:" << endl;
    for (size_t i = 0; i < training_data.size(); ++i) {
        vector<float> prediction = mlp.predict(training_data[i]);
        int predicted_class = std::distance(prediction.begin(), std::max_element(prediction.begin(), prediction.end()));
        cout << "Du doan cho anh " << i + 1 << ": " << predicted_class << " (";
        if (predicted_class == 0) {
            cout << "Hinh vuong";
        } else if (predicted_class == 1) {
            cout << "Hinh tron";
        } else {
            cout << "Hinh tam giac";
        }
        cout << ")" << endl;
    }

    return 0;
}