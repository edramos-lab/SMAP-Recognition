from torchvision.transforms import Compose, Resize, RandomHorizontalFlip, RandomVerticalFlip, RandomRotation, ColorJitter, ToTensor, Normalize

image_size=(224,224)
data_transforms = Compose([
    Resize(image_size),
    RandomHorizontalFlip(),
    RandomVerticalFlip(),
    RandomRotation(45),
    ColorJitter(brightness=0.01, contrast=0.01, saturation=0.01),  # Reduced effect
    ToTensor(),
    # Updated Normalize values (example only; calculate based on your dataset)
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# prompt: plot one image per subfolder, showing the label on the top of each image, in a two row, six column layout

# prompt: plot one image per subfolder, showing the label on the top of each image, in a two row layout, there are 11 subfolders, ensure plot all the images (11)

import matplotlib.pyplot as plt
import os

def plot_images_with_labels(use_transforms=False):
    # Get a list of all the subfolders in the train directory
    subfolders = [f for f in os.listdir('/content/things-10/train') if os.path.isdir(os.path.join('/content/things-10/train', f))]

    # Plot one image per subfolder
    fig, axes = plt.subplots(2, 6, figsize=(15, 8))
    for i, subfolder in enumerate(subfolders):
        # Get the path to the first image in the subfolder
        image_path = os.path.join('/content/things-10/train', subfolder, os.listdir(os.path.join('/content/things-10/train', subfolder))[0])

        # Load the image
        image = plt.imread(image_path)

        # Apply data_transforms if use_transforms is True
        if use_transforms:
            image = data_transforms(image)

        # Plot the image
        axes[i // 6, i % 6].imshow(image)
        axes[i // 6, i % 6].set_title(subfolder)
        axes[i // 6, i % 6].axis('off')

    # Show the plot
    
    axes[1,5].imshow(image)
    axes[1, 5].set_title(subfolder)
    axes[1,5].axis('off')
    plt.show()
    

# Call the function without using data_transforms
plot_images_with_labels(use_transforms=False)

# Call the function using data_transforms
plot_images_with_labels(use_transforms=True)
