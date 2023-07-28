import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
# from torch.utils.tensorboard import SummeryWritter
import matplotlib.pyplot as plt
import numpy as np
from filter import HLS_filter
from datasets import CustomDataset
from models import ConvMLP
from utils import save_checkpoints, validate

# Define the hyperparameters
EPOCHS = 400
BATCH_SIZE = 16
LEARNING_RATE = 1e-7
WEIGHT_DECAY = 1e-5
SCALE_FACTORS = [180, 255, 255, 180, 255, 255]
FREQ_PARAMS = 50

# Set the device for training
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Define the transformation
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

# Create the custom dataset and data loader
image_directory = 'Datasets/train_images'
validation_directory = 'Datasets/validation_images'

dataset = CustomDataset(image_directory, transform=transform)
validation_dataset = CustomDataset(validation_directory, transform=transform)

data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
validation_loader = DataLoader(validation_dataset, batch_size=BATCH_SIZE, shuffle=True)

# Create the model, loss function, and optimizer
model = ConvMLP().to(device)
#Matrice Validation
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), 
                             lr=LEARNING_RATE, 
                             weight_decay=WEIGHT_DECAY)

# Training loop
loss_values_list = []
val_loss_values_list = []

for epoch in range(EPOCHS):
    train_loss = 0.0
    model.train()

    for images, labels in data_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        # print(outputs)
        
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    avg_train_loss = train_loss / len(data_loader)
    avg_val_loss = validate(validation_loader, model, criterion, device)
    loss_values_list.append(avg_train_loss)
    val_loss_values_list.append(avg_val_loss)

    print(f"Epoch [{epoch + 1}/{EPOCHS}], Loss: {avg_train_loss:.8f}")

    if epoch % FREQ_PARAMS == 0:
        save_checkpoints(model, epoch)

# Plotting the training and validation loss
np.save("plot_loss.npy", np.array(loss_values_list))
np.save("plot_loss.npy", np.array(loss_values_list))

plt.plot(loss_values_list, label=f'Training Loss{avg_train_loss:.6f}')
plt.plot(val_loss_values_list, label=f'Validation Loss {avg_val_loss:.6f}')

plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation (MSE)Loss')

plt.legend()
plt.show()

# Validation images and color slider
model.eval()

with torch.no_grad():
    fig, axes = plt.subplots(1, 2)

    for images, labels in validation_loader:
        images = images.to(device)
        labels = labels.to(device)

        images_cpu = images[0].permute(1, 2, 0).mul(255).byte().cpu().numpy()
        outputs = model(images)
        outputs = outputs[0].cpu().numpy() * SCALE_FACTORS

        axes[0].imshow(images_cpu)
        axes[0].set_title('Original Image')
        axes[0].axis('off')

        filtered = HLS_filter(images_cpu, outputs)
        axes[1].imshow(filtered,cmap = 'gray')
        axes[1].set_title('Filtered Image')
        axes[1].axis('off')

        plt.show()
