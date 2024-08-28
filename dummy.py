import os

directory = r'C:/Users/edgar/OneDrive/Documentos/PhD/MLOPS/SMAP-Recognition/CustomDataset/AssemblySequence'
new_name = 'img'

# Get a list of all files in the directory
files = os.listdir(directory)

# Iterate over the files and rename them
for i, file in enumerate(files):
    # Check if the file is an image (you can modify this condition as per your requirement)
    if file.endswith('.png') or file.endswith('.jpg'):
        # Generate the new file name
        new_file_name = f'{new_name}{i+1:03d}.png'
        
        # Construct the full paths of the old and new file names
        old_file_path = os.path.join(directory, file)
        new_file_path = os.path.join(directory, new_file_name)
        
        # Rename the file
        os.rename(old_file_path, new_file_path)