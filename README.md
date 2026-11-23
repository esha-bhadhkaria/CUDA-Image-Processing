# CUDA Batch Image Processing

## 1. Project Overview

This project implements GPU-accelerated batch image processing using custom CUDA kernels and OpenCV.

The program processes a batch of 100 images from the CIFAR-10 dataset. Each input image is processed using three image-processing operations: grayscale conversion, Gaussian blur, and Sobel edge detection.

The image-processing computations are performed on the GPU using CUDA, while OpenCV is used for reading and saving images.

## 2. Features

The project implements three custom CUDA kernels:

### 1. Grayscale Conversion
- Converts colour images into grayscale images.
- Uses the weighted RGB formula:

  `Gray = 0.299R + 0.587G + 0.114B`

### 2. Gaussian Blur
- Applies a 3 × 3 Gaussian filter to smooth images.
- Uses the following kernel:

  ```
  1  2  1
  2  4  2
  1  2  1
  ```
- Divides the weighted sum by 16 to obtain the output pixel value.

### 3. Sobel Edge Detection
- Detects edges by calculating horizontal and vertical intensity gradients.
- Uses the Sobel X and Y operators.
- Computes edge magnitude using:

  `Magnitude = sqrt(Gx² + Gy²)`

All three operations use CUDA threads to process image pixels in parallel.

## 3. Technologies Used

- CUDA C++
- NVIDIA CUDA Toolkit
- OpenCV
- Python
- CIFAR-10 Dataset
- Makefile

## 4. Project Structure

```text
CUDA-Image-Processing/
│
├── src/
│   ├── main.cpp
│   └── image_kernels.cu
│
├── include/
│   └── image_kernels.cuh
│
├── Makefile
├── download_dataset.py
└── README.md
```

## 5. Dataset

The project uses the CIFAR-10 dataset.

The `download_dataset.py` script downloads the CIFAR-10 training dataset and extracts the first 100 images.

The images are converted into PNG format and saved in:

```text
dataset/input/
```

Install the required Python packages:

```bash
pip install torchvision opencv-python numpy
```

Download the dataset:

```bash
python download_dataset.py
```

## 6. Requirements

The following software and libraries are required:

- NVIDIA GPU with CUDA support
- NVIDIA CUDA Toolkit (`nvcc`)
- OpenCV development libraries
- C++17-compatible compiler
- pkg-config

On Ubuntu, OpenCV development libraries can be installed using:

```bash
sudo apt-get update
sudo apt-get install libopencv-dev pkg-config
```

Verify CUDA installation:

```bash
nvcc --version
```

Verify OpenCV installation:

```bash
pkg-config --modversion opencv4
```

## 7. Compilation

The project includes a Makefile to simplify compilation.

Navigate to the project directory and execute:

```bash
make
```

This compiles `main.cpp` and `image_kernels.cu` using NVCC and links the required OpenCV libraries.

The compiled executable is named:

```text
image_processor
```

## 8. Execution

Run the program using:

```bash
make run
```

The program also accepts command-line arguments to specify custom input and output directories:

```bash
./image_processor --input dataset/input --output output
```

The command-line arguments are:

- `--input`: Specifies the directory containing input images.
- `--output`: Specifies the directory where processed images will be saved.

If no arguments are provided, the program uses `dataset/input` as the input directory and `output` as the output directory.

## 9. Output

For every input image, the program generates three processed images.

The output directory is organised as follows:

```text
output/
│
├── grayscale/
├── blurred/
└── edges/
```

- `grayscale/` contains grayscale images.
- `blurred/` contains Gaussian-blurred images.
- `edges/` contains Sobel edge-detected images.

The program also displays a summary containing the total number of images found, successfully processed images, and failed images.

## 10. Execution Results

The project was successfully executed on an NVIDIA Tesla T4 GPU using CUDA 13.0.

The first 100 CIFAR-10 images were processed in a single execution.

| Metric | Result |
|---|---:|
| Total input images | 100 |
| Successfully processed | 100 |
| Failed images | 0 |
| Grayscale outputs | 100 |
| Gaussian blur outputs | 100 |
| Sobel edge outputs | 100 |
| Total output images | 300 |

## 11. Challenges and Lessons Learned

During the implementation, the project involved understanding how image-processing operations can be mapped to parallel GPU execution.

Some of the key concepts explored were:

- Mapping individual image pixels to CUDA threads.
- Organising threads into two-dimensional blocks and grids.
- Transferring image data between CPU and GPU memory.
- Implementing convolution-based operations such as Gaussian blur and Sobel edge detection.
- Handling image boundaries while applying neighbourhood-based filters.
- Integrating custom CUDA kernels with OpenCV for image input and output.

The project demonstrates how GPU parallelism can be applied to image-processing workloads involving multiple images.

## 12. Cleaning Generated Files

To remove the compiled executable and generated output images, run:

```bash
make clean
```

---

**Project:** CUDA Batch Image Processing  
**Dataset:** CIFAR-10 (100 images)  
**GPU Used:** NVIDIA Tesla T4