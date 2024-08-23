import os
import shutil

def count_images_per_folder(directory):
    image_count = {}
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(('.jpg', '.jpeg', '.png', '.gif')):
                folder = os.path.basename(root)
                image_count[folder] = image_count.get(folder, 0) + 1
    return image_count

def move_images_to_test(directory):
    test_directory = os.path.join(directory, 'test')
    os.makedirs(test_directory, exist_ok=True)

    for root, dirs, files in os.walk(directory):
        for folder in dirs:
            train_folder = os.path.join(root, folder)
            test_folder = os.path.join(test_directory, folder)
            os.makedirs(test_folder, exist_ok=True)

            image_files = [file for file in os.listdir(train_folder) if file.endswith(('.jpg', '.jpeg', '.png', '.gif'))]
            for i in range(min(20, len(image_files))):
                image_file = image_files[i]
                source_path = os.path.join(train_folder, image_file)
                destination_path = os.path.join(test_folder, image_file)
                shutil.move(source_path, destination_path)

directory = 'C:/Users/edgar/OneDrive/Documentos/PhD/MLOPS/SMAP-Recognition/CustomDataset/train'  # Replace with the actual directory path
image_count = count_images_per_folder(directory)
for folder, count in image_count.items():
    print(f"Folder '{folder}' contains {count} image(s).")

move_images_to_test(directory)
