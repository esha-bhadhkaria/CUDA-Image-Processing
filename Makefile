NVCC = nvcc
TARGET = image_processor
SRC = src/main.cpp src/image_kernels.cu
INCLUDE = -Iinclude

OPENCV = $(shell pkg-config --cflags --libs opencv4)

all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) -std=c++17 $(INCLUDE) $(SRC) -o $(TARGET) $(OPENCV)

run: $(TARGET)
	./$(TARGET) --input dataset/input --output output

clean:
	rm -f $(TARGET)
	rm -f output/grayscale/* output/blurred/* output/edges/*

.PHONY: all run clean
