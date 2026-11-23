
#include "../include/image_kernels.cuh"
#include <cuda_runtime.h>
#include <cmath>
#include <stdexcept>

void checkCUDA(cudaError_t error) {
    if (error != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(error));
    }
}

__global__ void grayscaleKernel(unsigned char* input, unsigned char* output, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int index = y * width + x;
        int b = input[index * 3];
        int g = input[index * 3 + 1];
        int r = input[index * 3 + 2];

        output[index] = (unsigned char)(0.299f * r + 0.587f * g + 0.114f * b);
    }
}


__global__ void gaussianKernel(unsigned char* input, unsigned char* output, int width,int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int weights[3][3] = {{1, 2, 1}, {2, 4, 2},{1, 2, 1}};
        int sum = 0;

        for (int dy = -1; dy <= 1; dy++) {
            for (int dx = -1; dx <= 1; dx++) {
                int nx = min(max(x + dx, 0), width - 1);
                int ny = min(max(y + dy, 0), height - 1);
                int index = ny * width + nx;
                sum += input[index] * weights[dy + 1][dx + 1];
            }
        }

        output[y * width + x] = (unsigned char)(sum / 16);
    }
}

__global__ void sobelKernel(unsigned char* input, unsigned char* output, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int Gx[3][3] = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};
        int Gy[3][3] = {{-1, -2, -1}, { 0,  0,  0}, { 1,  2,  1}};
        int sumX = 0;
        int sumY = 0;

        for (int dy = -1; dy <= 1; dy++) {
            for (int dx = -1; dx <= 1; dx++) {
                int nx = min(max(x + dx, 0), width - 1);
                int ny = min(max(y + dy, 0), height - 1);
                int pixel = input[ny * width + nx];
                sumX += pixel * Gx[dy + 1][dx + 1];
                sumY += pixel * Gy[dy + 1][dx + 1];
            }
        }

        int magnitude = (int)sqrtf((float)(sumX * sumX + sumY * sumY));
        magnitude = min(magnitude, 255);
        output[y * width + x] = (unsigned char)magnitude;
    }
}

void grayscaleGPU(const cv::Mat& input, cv::Mat& output) {
    CV_Assert(input.type() == CV_8UC3);
    int width = input.cols;
    int height = input.rows;
    cv::Mat source = input.isContinuous() ? input : input.clone();
    output.create(height, width, CV_8UC1);
    unsigned char *d_input, *d_output;
    size_t inputSize = width * height * 3;
    size_t outputSize = width * height;
    checkCUDA(cudaMalloc((void**)&d_input, inputSize));
    checkCUDA(cudaMalloc((void**)&d_output, outputSize));
    checkCUDA(cudaMemcpy(d_input, source.data, inputSize, cudaMemcpyHostToDevice));

    dim3 block(16, 16);
    dim3 grid((width + 15) / 16, (height + 15) / 16);

    grayscaleKernel<<<grid, block>>>(d_input, d_output, width, height);

    checkCUDA(cudaGetLastError());
    checkCUDA(cudaDeviceSynchronize());
    checkCUDA(cudaMemcpy(output.data, d_output, outputSize, cudaMemcpyDeviceToHost));

    cudaFree(d_input);
    cudaFree(d_output);
}

void gaussianBlurGPU(const cv::Mat& input, cv::Mat& output) {

    CV_Assert(input.type() == CV_8UC1);
    int width = input.cols;
    int height = input.rows;

    cv::Mat source = input.isContinuous() ? input : input.clone();
    output.create(height, width, CV_8UC1);
    unsigned char *d_input, *d_output;
    size_t size = width * height;

    checkCUDA(cudaMalloc((void**)&d_input, size));
    checkCUDA(cudaMalloc((void**)&d_output, size));

    checkCUDA(cudaMemcpy(d_input, source.data, size, cudaMemcpyHostToDevice));

    dim3 block(16, 16);
    dim3 grid((width + 15) / 16, (height + 15) / 16);

    gaussianKernel<<<grid, block>>>(d_input, d_output, width, height);

    checkCUDA(cudaGetLastError());
    checkCUDA(cudaDeviceSynchronize());
    checkCUDA(cudaMemcpy(output.data, d_output, size, cudaMemcpyDeviceToHost));

    cudaFree(d_input);
    cudaFree(d_output);
}

void sobelGPU(const cv::Mat& input, cv::Mat& output) {

    CV_Assert(input.type() == CV_8UC1);
    int width = input.cols;
    int height = input.rows;
    cv::Mat source = input.isContinuous() ? input : input.clone();
    output.create(height, width, CV_8UC1);
    unsigned char *d_input, *d_output;
    size_t size = width * height;

    checkCUDA(cudaMalloc((void**)&d_input, size));
    checkCUDA(cudaMalloc((void**)&d_output, size));
    checkCUDA(cudaMemcpy(d_input, source.data, size, cudaMemcpyHostToDevice));

    dim3 block(16, 16);
    dim3 grid((width + 15) / 16, (height + 15) / 16);

    sobelKernel<<<grid, block>>>(d_input, d_output, width, height);
    checkCUDA(cudaGetLastError());
    checkCUDA(cudaDeviceSynchronize());
    checkCUDA(cudaMemcpy(output.data, d_output, size, cudaMemcpyDeviceToHost));

    cudaFree(d_input);
    cudaFree(d_output);
}