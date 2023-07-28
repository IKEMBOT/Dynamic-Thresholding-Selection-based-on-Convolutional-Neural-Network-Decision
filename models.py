import os
import torch
import torch.nn as nn
import torch
import torch.nn as nn
import cv2
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from PIL import Image
from filter import HLS_filter

# Define the model architecture
class ConvMLP(nn.Module):
    def __init__(self):
        super(ConvMLP, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            # nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            # nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            # nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            # nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        self.fc = nn.Linear(512 * 16 * 16, 256)
        # self.fc2 = nn.Linear(256,6)
        # self.activation = nn.Sigmoid()
        

    def forward(self, x):
        x = self.encoder(x)
        x = torch.flatten(x, start_dim=1)  # Flatten the tensor
        x = self.fc(x)
        # x = self.dropout(x)
        # x = self.activation(x)
        # x = self.fc2(x)
        # x = self.activation(x)
        return x

    def test(self,image,model,preprocess,scaling_factors):
        # Load and preprocess the test image
        test_image = cv2.imread(image)
        test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
        test_image_pil = Image.fromarray(test_image)  # Convert NumPy array to PIL image
        preprocessed_image = preprocess(test_image_pil)
        input_tensor = preprocessed_image.unsqueeze(0)  # Add batch dimension

        # If available, move the input tensor to the GPU for faster inference
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        input_tensor = input_tensor.to(device)
        model.to(device)

        # Perform the forward pass
        with torch.no_grad():
            output = model(input_tensor)
            hls_values = output.squeeze().cpu().numpy() * scaling_factors

        # Apply HLS filtering to the test image
        filtered = HLS_filter(test_image, hls_values)

        # Display the original image and the filtered image
        fig, axes = plt.subplots(1, 2)
        axes[0].imshow(test_image)
        axes[0].set_title('Original Image')
        axes[0].axis('off')

        axes[1].imshow(filtered, cmap='gray')
        axes[1].set_title('Filtered Image')
        axes[1].axis('off')

        plt.show()
