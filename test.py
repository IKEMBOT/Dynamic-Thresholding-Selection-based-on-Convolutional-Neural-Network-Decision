import torch
import torch.nn as nn
import cv2
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from PIL import Image
from filter import HLS_filter
from utils import load_checkpoints

LOAD_FILE = "save_models/model_950.pth"
IMAGE_FILE = "Datasets/test_images/00000784.jpg"
SCALE_FACTOR = [180, 255, 255, 180, 255, 255]
PREPROCESS = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])
model = load_checkpoints(LOAD_FILE)
model.test(IMAGE_FILE,model,PREPROCESS,SCALE_FACTOR)

