import torch
import torch.nn as nn
import albumentations as A
# import torchvision.transforms as transforms
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

#Validation Function
save_file = "/home/ntnuerc/ilham/Machine_Learning/FINAL_EXAM_CNN+NN/save_models"
def validate(loader, model, criterion, device):
    model.eval()
    val_loss = 0

    with torch.no_grad():
        for images, labels in loader:
            
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            # print(outputs.shape,labels.shape)
            
            loss = criterion(outputs, labels)
            val_loss += loss.item()

    avg_val_loss = val_loss / len(loader)
    return avg_val_loss


def save_checkpoints(model,epoch):
    # Save the trained model
    print("filed saved")
    path = save_file + "model_{}.pth".format(epoch)
    torch.save(model, path)

def load_checkpoints(load_file):
    # Load the trained model
    model = torch.load(load_file)
    model.eval()
    return model
