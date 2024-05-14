#include <stdio.h>
#include <stdlib.h>
#include <time.h>


#define MASK_WIDTH 3
#define O_TILE_WIDTH 12

float* readPGM(const char* filename, int* width, int* height) {
    FILE* f = fopen(filename, "rb");
    if (f == NULL) {
        perror("Error opening file");
        return NULL;
    }

    char type[3];
    fscanf(f, "%s", type);
    if (type[0] != 'P' || type[1] != '5') {
        fprintf(stderr, "Not a PGM file\n");
        fclose(f);
        return NULL;
    }

    fscanf(f, "%d %d", width, height);
    int maxVal;
    fscanf(f, "%d", &maxVal);

    int imageSize = (*width) * (*height);
    float* imageData = (float*)malloc(imageSize * sizeof(float));

    for (int i = 0; i < imageSize; i++) {
        unsigned char pixel;
        fread(&pixel, sizeof(unsigned char), 1, f);
        imageData[i] = (float)pixel;
    }

    fclose(f);
    return imageData;
}

void writePGM(const char* filename, float* imageData, int width, int height) {
    FILE* f = fopen(filename, "wb");
    if (f == NULL) {
        perror("Error opening file");
        return;
    }

    fprintf(f, "P5\n%d %d\n255\n", width, height);

    for (int i = 0; i < width * height; i++) {
        unsigned char pixel = (unsigned char)imageData[i];
        fwrite(&pixel, sizeof(unsigned char), 1, f);
    }

    fclose(f);
}

void convolution2D(float *N, float *M, float *P, int width, int height) {
    for (int row = 0; row < height; row++) {
        for (int col = 0; col < width; col++) {
            float output = 0.0f;
            for (int i = -MASK_WIDTH / 2; i <= MASK_WIDTH / 2; i++) {
                for (int j = -MASK_WIDTH / 2; j <= MASK_WIDTH / 2; j++) {
                    int curRow = row + i;
                    int curCol = col + j;
                    if (curRow >= 0 && curRow < height && curCol >= 0 && curCol < width) {
                        output += M[(i + MASK_WIDTH / 2) * MASK_WIDTH + (j + MASK_WIDTH / 2)] *
                                  N[curRow * width + curCol];
                    }
                }
            }
            P[row * width + col] = output;
        }
    }
}

int main() {
    const char* imagePath = "input_512_512.pgm";
    int imageWidth, imageHeight;

    float* image = readPGM(imagePath, &imageWidth, &imageHeight);
    if (image == NULL) {
        return -1;
    }

    int imageSize = imageWidth * imageHeight;
    float *mask, *output;

    mask = (float*)malloc(MASK_WIDTH * MASK_WIDTH * sizeof(float));
    output = (float*)malloc(imageSize * sizeof(float));

    for (int i = 0; i < MASK_WIDTH * MASK_WIDTH; i++) {
        mask[i] = 1.0f / (MASK_WIDTH * MASK_WIDTH);
    }
    clock_t start = clock();
    convolution2D(image, mask, output, imageWidth, imageHeight);
    clock_t end = clock();
    double executionTime = (double)(end - start) / CLOCKS_PER_SEC;

    printf("Time taken for erosion: %f seconds\n", executionTime);

    writePGM("output_image_CPU.pgm", output, imageWidth, imageHeight);

    free(image);
    free(mask);
    free(output);

    return 0;
}
