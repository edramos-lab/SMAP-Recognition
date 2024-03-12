import os
import shutil
from datetime import datetime
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import torch
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

def copyFiles(source_folder,destination_folder):
    if not os.path.exists(destination_folder):
        source_folder = "/home/edramos/Documents/MLOPS/SmartAssemblyProcessRecognition/dataset"
        destination_folder = "/home/edramos/Documents/MLOPS/SmartAssemblyProcessRecognition/CustomDataset"
        prefix = "SpaceShuttle-"

        # Get a list of all files in the source folder
        files = os.listdir(source_folder)

        # Iterate over each file
        for file in files:
            # Get the full path of the file
            file_path = os.path.join(source_folder, file)
            
            # Get the timestamp from the original file
            timestamp = os.path.getmtime(file_path)
            timestamp = datetime.fromtimestamp(timestamp).strftime("%Y%m%d%H%M%S")
            
            # Create the new file name with the prefix and timestamp
            new_file_name = prefix + timestamp + "-" + file
            
            # Get the full path of the destination file
            destination_file_path = os.path.join(destination_folder, new_file_name)
            
            # Copy the file to the destination folder with the new name
            shutil.copy2(file_path, destination_file_path)


def createFolders(destination_folder):
    for i in range(1, 51):
        folder_name = f"step{i}"
        folder_path = os.path.join(destination_folder, folder_name)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
            print(f"Created folder: {folder_path}")
        else:
            print(f"Folder already exists: {folder_path}")


if __name__ == '__main__':



        # Define the augmentation transformations
    transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomShadow(p=0.5),
        A.RandomBrightnessContrast(p=0.5),
        A.GaussNoise(p=0.5, var_limit=(0, 2)),
        A.ZoomBlur(p=0.5),
        A.RandomSnow(p=0.5),
        A.RandomSunFlare(p=0.5),
        A.Defocus(p=0.5),
        ToTensorV2()
    ])

    dataset_path="/home/edramos/Documents/MLOPS/SmartAssemblyProcessRecognition/CustomDataset"
        # Create the dataset
    dataset = ImageFolder(dataset_path, transform=transform)
    # Duplicate the dataset 5 times
    expanded_dataset = torch.utils.data.ConcatDataset([dataset] * 5)
    # Calculate the sizes for training, validation, and testing
    total_samples = len(expanded_dataset)
    train_size = int(0.8 * total_samples)
    val_size = int(0.1 * total_samples)
    test_size = total_samples - train_size - val_size

    print("Total Samples:", total_samples)
    print("Train Size:", train_size)
    print("Validation Size:", val_size)
    print("Test Size:", test_size)

    # Split the dataset into training, validation, and testing
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(expanded_dataset, [train_size, val_size, test_size])

    # Create the data loaders
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=True)

    print("Train Dataloader Size:", len(train_dataloader.dataset))
