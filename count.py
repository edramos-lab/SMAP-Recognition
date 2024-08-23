import os

def count_images_per_folder(directory):
    image_count = {}
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(('.jpg', '.jpeg', '.png', '.gif')):
                folder = os.path.basename(root)
                image_count[folder] = image_count.get(folder, 0) + 1
    return image_count

directory = 'C:/Users/edgar/OneDrive/Documentos/PhD/MLOPS/SMAP-Recognition/CustomDataset/resized_train (copy)'  # Replace with the actual directory path
image_count = count_images_per_folder(directory)
for folder, count in image_count.items():
    print(f"Folder '{folder}' contains {count} image(s).")

