#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define MASK_WIDTH 3
#define O_TILE_WIDTH 12 
#define BLOCK_WIDTH (O_TILE_WIDTH + MASK_WIDTH - 1)

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

__global__ void convolution2D(float *N, float *M, float *P, int Width) {
    __shared__ float N_ds[BLOCK_WIDTH][BLOCK_WIDTH];
    __shared__ float M_ds[MASK_WIDTH][MASK_WIDTH];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row_o = blockIdx.y * O_TILE_WIDTH + ty;
    int col_o = blockIdx.x * O_TILE_WIDTH + tx;

    int row_i = row_o - MASK_WIDTH / 2;
    int col_i = col_o - MASK_WIDTH / 2;

    if ((row_i >= 0) && (row_i < Width) && (col_i >= 0) && (col_i < Width)) {
        N_ds[ty][tx] = N[row_i * Width + col_i];
    } else {
        N_ds[ty][tx] = 0.0f;
    }

    if (ty < MASK_WIDTH && tx < MASK_WIDTH) {
        M_ds[ty][tx] = M[ty * MASK_WIDTH + tx];
    }

    __syncthreads();

    float output = 0.0f;
    if (ty < O_TILE_WIDTH && tx < O_TILE_WIDTH) {
        for (int i = 0; i < MASK_WIDTH; i++) {
            for (int j = 0; j < MASK_WIDTH; j++) {
                output += M_ds[i][j] * N_ds[i + ty][j + tx];
            }
        }
        if (row_o < Width && col_o < Width) {
            P[row_o * Width + col_o] = output;
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
    float *d_image, *d_mask, *d_output;

    mask = (float*)malloc(MASK_WIDTH * MASK_WIDTH * sizeof(float));
    output = (float*)malloc(imageSize * sizeof(float));

    for (int i = 0; i < MASK_WIDTH * MASK_WIDTH; i++) {
        mask[i] = 1.0f / (MASK_WIDTH * MASK_WIDTH);
    }

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaMalloc(&d_image, imageSize * sizeof(float));
    cudaMalloc(&d_mask, MASK_WIDTH * MASK_WIDTH * sizeof(float));
    cudaMalloc(&d_output, imageSize * sizeof(float));

    cudaMemcpy(d_image, image, imageSize * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_mask, mask, MASK_WIDTH * MASK_WIDTH * sizeof(float), cudaMemcpyHostToDevice);

    dim3 dimBlock(BLOCK_WIDTH, BLOCK_WIDTH);
    dim3 dimGrid((imageWidth - 1) / O_TILE_WIDTH + 1, (imageHeight - 1) / O_TILE_WIDTH + 1);
    
    cudaEventRecord(start);
    convolution2D<<<dimGrid, dimBlock>>>(d_image, d_mask, d_output, imageWidth);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Execution time: %f seconds\n", milliseconds/1000);
    
    cudaMemcpy(output, d_output, imageSize * sizeof(float), cudaMemcpyDeviceToHost);

    writePGM("output_image_GPU.pgm", output, imageWidth, imageHeight);

    cudaFree(d_image);
    cudaFree(d_mask);
    cudaFree(d_output);

    free(image);
    free(mask);
    free(output);

    return 0;
}
