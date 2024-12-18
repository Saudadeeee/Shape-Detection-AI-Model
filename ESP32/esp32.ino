#include "esp_camera.h"
#include "FS.h"
#include "SD_MMC.h"
#include <Preferences.h>

Preferences preferences;
int photoCount = 0;

void initCamera() {
    camera_config_t config;
    config.ledc_channel = LEDC_CHANNEL_0;
    config.ledc_timer = LEDC_TIMER_0;
    config.pin_d0 = 5;
    config.pin_d1 = 18;
    config.pin_d2 = 19;
    config.pin_d3 = 21;
    config.pin_d4 = 36;
    config.pin_d5 = 39;
    config.pin_d6 = 34;
    config.pin_d7 = 35;
    config.pin_xclk = 0;
    config.pin_pclk = 22;
    config.pin_vsync = 25;
    config.pin_href = 23;
    config.pin_sscb_sda = 26;
    config.pin_sscb_scl = 27;
    config.pin_pwdn = 32;
    config.pin_reset = -1;
    config.xclk_freq_hz = 20000000;
    config.pixel_format = PIXFORMAT_JPEG;
    config.frame_size = FRAMESIZE_UXGA;
    config.jpeg_quality = 10;
    config.fb_count = 2;

    if (esp_camera_init(&config) != ESP_OK) {
        Serial.println("Camera init failed");
        return;
    }
    Serial.println("Camera init success");

    // Cài đặt chế độ đen-trắng
    sensor_t *s = esp_camera_sensor_get();
    s->set_saturation(s, -2);
    s->set_special_effect(s, 2);
}

void takePhoto() {
    camera_fb_t *fb = esp_camera_fb_get();
    if (!fb) {
        Serial.println("Chụp ảnh không thành công");
        return;
    }

    // Tạo tên file ảnh
    String path = "/image_" + String(photoCount) + ".jpg";
    File file = SD_MMC.open(path.c_str(), FILE_WRITE);
    if (file) {
        file.write(fb->buf, fb->len);
        Serial.printf("Ảnh đã được lưu tại %s\n", path.c_str());
        file.close();
    } else {
        Serial.println("Lỗi khi ghi file");
    }

    // Giải phóng bộ nhớ và tăng biến đếm
    esp_camera_fb_return(fb);
    photoCount++;

    // Lưu lại photoCount vào flash
    preferences.putInt("photoCount", photoCount);
}

void setup() {
    Serial.begin(115200);
    initCamera();

    if (!SD_MMC.begin()) {
        Serial.println("SD Card không hoạt động");
        return;
    }
    Serial.println("SD Card khởi động thành công");

    // Đọc photoCount từ bộ nhớ flash
    preferences.begin("photo", false);
    photoCount = preferences.getInt("photoCount", 0);

    // Chụp ảnh đầu tiên khi reset
    takePhoto();
}

void loop() {
    // Không cần làm gì trong loop
}

