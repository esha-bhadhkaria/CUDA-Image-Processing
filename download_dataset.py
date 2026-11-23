import torchvision
import cv2
import numpy as np
import os

dataset = torchvision.datasets.CIFAR10(root="./data", train=True, download=True)
os.makedirs("dataset/input", exist_ok=True)

for i in range(100):
    image, label = dataset[i]
    image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    cv2.imwrite(f"dataset/input/image_{i}.png",image)

print("Dataset downloaded successfully!")
print("Total images:", len(os.listdir("dataset/input")))