import os
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
from comet_ml import Experiment

# 1. Initialize Comet Experiment
import os
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
from comet_ml import Experiment
import torchvision.transforms as transforms
from PIL import Image
import torch

# 1. Initialize Comet Experiment
experiment = Experiment(
    api_key="isAZKqoDUbKLHrFCn1ojc7ckM",
    project_name="Next image sequence -GAN-Pytorc",
    workspace="edramos-lab"
)

model_name="custom-model"
experiment.log_parameter("model_name", model_name)

# 2. Image Preprocessing and Dataset Definition

class AssemblyDataset(Dataset):
    def __init__(self, data_dir, target_size=(224, 224)):
        self.images = []
        self.labels = []
        self.target_size = target_size
        
        for step_num in range(1, 39):  # steps 01 to 38
            step_folder = os.path.join(data_dir, f"step{step_num:02d}")
            
            if not os.path.exists(step_folder):
                print(f"Directory {step_folder} not found.")
                continue
            
            for img_name in os.listdir(step_folder):
                img_path = os.path.join(step_folder, img_name)
                
                img = cv2.imread(img_path)
                if img is None:
                    print(f"Failed to load {img_path}")
                    continue
                
                img_resized = cv2.resize(img, self.target_size)
                img_normalized = (img_resized / 127.5) - 1.0
                
                self.images.append(img_normalized)
                self.labels.append(step_num - 1)
        
        self.images = np.array(self.images, dtype=np.float32)
        self.labels = np.array(self.labels, dtype=np.int32)
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img = self.images[idx].transpose(2, 0, 1)  # Convert to (C, H, W) for PyTorch
        label = self.labels[idx]
        return torch.tensor(img, dtype=torch.float32), torch.tensor(label, dtype=torch.int64)
        
'''
class AssemblyDataset(Dataset):
    def __init__(self, data_dir, target_size=(224, 224)):
        self.data_dir = data_dir
        self.target_size = target_size

        # Lists to store image paths and labels
        self.images = []
        self.labels = []
        
        # Iterate over the steps (folders) and collect images
        for step_num in range(1, 39):  # Assuming folders like step01, step02, etc.
            step_folder = os.path.join(data_dir, f"step{step_num:02d}")
            if not os.path.exists(step_folder):
                print(f"Warning: Directory {step_folder} not found.")
                continue
            
            for img_name in os.listdir(step_folder):
                img_path = os.path.join(step_folder, img_name)
                
                # Only consider valid image files (e.g., .jpg, .png)
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    try:
                        # Open the image to ensure it's valid
                        img = Image.open(img_path)
                        self.images.append(img_path)  # Store the image path
                        self.labels.append(step_num - 1)  # Use step number as label
                    except Exception as e:
                        print(f"Error loading image {img_path}: {e}")
        
        # Define the augmentation pipeline
        self.transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(degrees=10),  # Rotate by up to 10 degrees
            #transforms.ColorJitter(brightness=0.01, contrast=0.01, saturation=0.01, hue=0.01),
            transforms.Resize(target_size),  # Resize images to the target size
            transforms.ToTensor(),  # Convert images to tensors (0-1 range)
        ])
    
    def __len__(self):
        return len(self.images)  # Return the number of images in the dataset

    def __getitem__(self, idx):
        # Load the image file at the given index
        img_path = self.images[idx]
        img = Image.open(img_path)

        # Apply augmentation transforms to the image
        img = self.transform(img)

        # Get the label associated with the image
        label = self.labels[idx]

        return img, torch.tensor(label, dtype=torch.int64)
'''
# Initialize the dataset
data_dir = 'C:/Users/edgar/OneDrive/Documentos/PhD/MLOPS/SMAP-Recognition/CustomDataset/train'#'/content/SMAP-Recognition/CustomDataset/train'
dataset = AssemblyDataset(data_dir, target_size=(224, 224))
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

# 3. Define the Generator

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        
        # Encoder
        self.conv1 = nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(128)
        
        # Decoder
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.upconv2 = nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1)
    
    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.bn1(self.conv2(x)))
        
        x = torch.relu(self.upconv1(x))
        x = torch.tanh(self.upconv2(x))
        return x

# 4. Define the Discriminator

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        
        self.conv1 = nn.Conv2d(6, 64, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(128)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(128*56*56, 1)
    
    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.bn1(self.conv2(x)))
        
        x = self.flatten(x)
        x = torch.sigmoid(self.fc1(x))
        return x

# 5. Loss and Optimizer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

generator = Generator().to(device)
discriminator = Discriminator().to(device)

criterion = nn.BCELoss()
optimizer_g = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_d = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

# 6. Training Loop with Comet Logging
def train_gan(epochs, dataloader):
    for epoch in range(epochs):
        for i, (imgs, labels) in enumerate(dataloader):
            imgs = imgs.to(device)
            labels = labels.to(device)
            
            # Shift labels to get next-step images
            next_labels = labels + 1
            next_labels[next_labels > 37] = 0
            
            next_imgs = torch.zeros_like(imgs)
            for j, label in enumerate(next_labels):
                # Get only the image data from the dataset
                next_img, _ = dataset[label.item()]  # Get image and discard label 
                next_imgs[j] = next_img # Assign only the image to next_imgs[j]
            
            next_imgs = next_imgs.to(device)
            
            batch_size = imgs.size(0)
            valid = torch.ones(batch_size, 1).to(device)
            fake = torch.zeros(batch_size, 1).to(device)
            
            # Train Discriminator
            optimizer_d.zero_grad()
            real_imgs = torch.cat((imgs, next_imgs), 1)
            real_loss = criterion(discriminator(real_imgs), valid)
            
            generated_imgs = generator(imgs)
            fake_imgs = torch.cat((imgs, generated_imgs), 1)
            fake_loss = criterion(discriminator(fake_imgs), fake)
            
            d_loss = (real_loss + fake_loss) / 2
            d_loss.backward()
            optimizer_d.step()
            
            # Train Generator
            optimizer_g.zero_grad()
            generated_imgs = generator(imgs)
            fake_imgs = torch.cat((imgs, generated_imgs), 1)
            g_loss = criterion(discriminator(fake_imgs), valid)
            g_loss.backward()
            optimizer_g.step()
        
        print(f"Epoch {epoch}/{epochs} | D Loss: {d_loss.item()} | G Loss: {g_loss.item()}")
        
        # Log metrics to Comet after each epoch
        experiment.log_metric("d_loss", d_loss.item(), step=epoch)
        experiment.log_metric("g_loss", g_loss.item(), step=epoch)
        experiment.log_epoch_end(epoch)


# Train the GAN and log the important metrics to Comet
train_gan(epochs=4, dataloader=dataloader)

# 7. Inference

def infer_next_image(input_img_path):
    input_img = cv2.imread(input_img_path)
    input_img_resized = cv2.resize(input_img, (224, 224))
    input_img_normalized = (input_img_resized / 127.5) - 1.0
    input_img_normalized = np.transpose(input_img_normalized, (2, 0, 1))  # Convert to (C, H, W)
    input_tensor = torch.tensor(input_img_normalized, dtype=torch.float32).unsqueeze(0).to(device)
    
    with torch.no_grad():
        predicted_img = generator(input_tensor).cpu().numpy()[0]
        predicted_img = np.transpose(predicted_img, (1, 2, 0))  # Convert back to (H, W, C)
        predicted_img = 0.5 * predicted_img + 0.5  # Rescale to [0, 1] range
    
    return predicted_img

# Example inference
predicted_img = infer_next_image('/content/SMAP-Recognition/CustomDataset/train/step07/20240310_174433_100.jpg')

# Visualize the input and output
plt.subplot(1, 2, 1)
plt.title("Input Image (step07)")
plt.imshow(cv2.cvtColor(cv2.imread('/content/SMAP-Recognition/CustomDataset/train/step07/20240310_174433_100.jpg'), cv2.COLOR_BGR2RGBA))

plt.subplot(1, 2, 2)
plt.title("Predicted Image (step08)")
plt.imshow(predicted_img)

plt.show()


