import os
import shutil
import random
import time
source_path = '/home/edramos/Documents/MLOPS/SMAP-Recognition/CustomDataset/train'
destination_path = '/home/edramos/Documents/MLOPS/SMAP-Recognition/CustomDataset/dataset'

# Get a list of all subfolders in the source path
subfolders = [f.path for f in os.scandir(source_path) if f.is_dir()]

# Iterate through each subfolder
for subfolder in subfolders:
    # Create the corresponding subfolder in the destination path
    destination_subfolder = os.path.join(destination_path, os.path.basename(subfolder))
    os.makedirs(destination_subfolder, exist_ok=True)
    
    # Get a list of all image files in the subfolder
    image_files = [f.path for f in os.scandir(subfolder) if f.is_file() and f.name.endswith('.jpg')]
    
    # Randomly select 5 image files
    selected_images = random.sample(image_files, 5)
    
    # Copy the selected images to the destination subfolder and rename them
    for image in selected_images:
        timestamp = str(int(time.time()))
        random_number = random.randint(0, 100)
        new_filename = f'spaceshuttle-{timestamp}-{random_number}.png'
        new_filepath = os.path.join(destination_subfolder, new_filename)
        shutil.copy(image, new_filepath)
