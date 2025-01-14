# Project AI and Microprocessor

## Chủ đề: Nhận diện hình khối (vuông, tròn, tam giác, ngôi sao) bằng mạng CNN

### 1. Model AI

Dự án này sử dụng mạng nơ-ron tích chập (Convolutional Neural Network - CNN) để nhận diện các hình khối cơ bản như vuông, tròn, tam giác và ngôi sao. Mạng CNN được huấn luyện để phân loại các hình ảnh đầu vào thành một trong bốn loại hình khối này.

### 2. Giới thiệu về từng lớp trong model

Model CNN được định nghĩa trong file `cnn_model.py` với các lớp chính như sau:

- **Conv2d**: Lớp tích chập 2D, được sử dụng để trích xuất các đặc trưng từ hình ảnh đầu vào.
- **BatchNorm2d**: Lớp chuẩn hóa batch, giúp tăng tốc độ huấn luyện và ổn định mạng.
- **ReLU**: Hàm kích hoạt phi tuyến, giúp mạng học được các đặc trưng phức tạp.
- **MaxPool2d**: Lớp pooling, giảm kích thước của đặc trưng và giảm thiểu tính dư thừa.
- **Linear**: Lớp kết nối đầy đủ, được sử dụng để phân loại các đặc trưng đã trích xuất.

### 3. Hướng dẫn sử dụng

#### Workflow

1. **Chuẩn bị ảnh**: Ảnh đầu vào phải ở định dạng PNG.
2. **Gửi ảnh đến backend**: Ảnh sẽ được gửi đến backend thông qua một request POST.
3. **Xử lý ảnh**: Backend sẽ nhận ảnh, lưu tạm thời và chuẩn bị ảnh cho model CNN.
4. **Dự đoán**: Model CNN sẽ dự đoán loại hình khối từ ảnh đầu vào.
5. **Trả kết quả**: Backend sẽ trả về kết quả dự đoán dưới dạng JSON.

#### Cách sử dụng

1. **Chạy server backend**:
   ```bash
   python app.py
   ```

2. **Gửi ảnh đến server để dự đoán**:
   Sử dụng script `test_client.py` để gửi ảnh đến server:
   ```bash
   python test_client.py
   ```

   Script này sẽ gửi ảnh `test.png` đến server và in ra kết quả dự đoán.

#### Cấu trúc thư mục

- `cnn_model.py`: Định nghĩa model CNN.
- `Shape_Detection.py`: Các hàm xử lý ảnh.
- `app.py`: Server backend Flask.
- `test_client.py`: Script gửi ảnh đến server để dự đoán.

#### Yêu cầu

- Python 3.x
- Thư viện: `torch`, `torchvision`, `flask`, `requests`, `PIL`

Cài đặt các thư viện cần thiết:
```bash
pip install torch torchvision flask requests pillow
