#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#define THRESHOLD 128 // Nguong nhi phan (co the dieu chinh)

// Ham doc anh BMP va chuyen sang grayscale
unsigned char **readBMP(const char *filename, int *width, int *height) {
    FILE *file = fopen(filename, "rb");
    if (file == NULL) {
        printf("Khong the mo file BMP: %s\n", filename);
        return NULL;
    }

    unsigned char header[54];
    fread(header, sizeof(unsigned char), 54, file); // Header BMP co kich thuoc 54 byte

    // Doc kich thuoc anh tu header
    *width = *(int*)&header[18];
    *height = *(int*)&header[22];
    printf("Kich thuoc anh: %dx%d\n", *width, *height);

    // Tinh so byte moi hang (duoc lam tron boi so cua 4)
    int rowPadded = (*width * 3 + 3) & (~3);

    // Cap phat bo nho cho anh grayscale
    unsigned char **grayscaleImage = (unsigned char **)malloc(*height * sizeof(unsigned char *));
    for (int i = 0; i < *height; i++) {
        grayscaleImage[i] = (unsigned char *)malloc(*width * sizeof(unsigned char));
    }

    // Doc du lieu pixel va chuyen doi sang grayscale
    unsigned char *rowData = (unsigned char *)malloc(rowPadded);
    for (int i = 0; i < *height; i++) {
        fread(rowData, sizeof(unsigned char), rowPadded, file);
        for (int j = 0; j < *width; j++) {
            int b = rowData[j * 3];
            int g = rowData[j * 3 + 1];
            int r = rowData[j * 3 + 2];
            grayscaleImage[*height - i - 1][j] = (r * 0.299 + g * 0.587 + b * 0.114); // Chuyen sang grayscale
        }
    }

    free(rowData);
    fclose(file);
    return grayscaleImage;
}

// Ham chuyen doi anh muc xam sang nhi phan
void convertToBinary(unsigned char **grayscaleImage, unsigned char **binaryImage, int width, int height) {
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            binaryImage[i][j] = (grayscaleImage[i][j] >= THRESHOLD) ? 1 : 0;
        }
    }
}

// Ham giai phong bo nho
void freeImage(unsigned char **image, int height) {
    for (int i = 0; i < height; i++) {
        free(image[i]);
    }
    free(image);
}

int main() {
    int width, height;

    // Doc anh BMP
    unsigned char **grayscaleImage = readBMP("input.bmp", &width, &height);
    if (grayscaleImage == NULL) {
        return 1;
    }

    // Cap phat bo nho cho anh nhi phan
    unsigned char **binaryImage = (unsigned char **)malloc(height * sizeof(unsigned char *));
    for (int i = 0; i < height; i++) {
        binaryImage[i] = (unsigned char *)malloc(width * sizeof(unsigned char));
    }

    // Chuyen doi sang nhi phan
    convertToBinary(grayscaleImage, binaryImage, width, height);

    // Ghi anh nhi phan ra file txt
    FILE *outputFile = fopen("output.txt", "w");
    if (outputFile == NULL) {
        printf("Khong the tao file txt.\n");
        freeImage(grayscaleImage, height);
        freeImage(binaryImage, height);
        return 1;
    }

    // Ghi ma tran 0 va 1 vao file txt
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            fprintf(outputFile, "%d ", binaryImage[i][j]); // Ghi 0 hoac 1
        }
        fprintf(outputFile, "\n"); // Xuong dong sau moi hang
    }
    fclose(outputFile);

    printf("Chuyen doi anh sang nhi phan va luu vao file txt thanh cong!\n");

    // Giai phong bo nho
    freeImage(grayscaleImage, height);
    freeImage(binaryImage, height);

    return 0;
}
