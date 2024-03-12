import os
import cv2

output_dir = "/home/edramos/Documents/MLOPS/SmartAssemblyProcessRecognition/CustomDataset/resized_train"
background_path = "/home/edramos/Documents/MLOPS/SmartAssemblyProcessRecognition/CustomDataset/table.png"

# Load the background image once
background = cv2.imread(background_path)

for step in range(1, 39):  # Assuming step1 to step38
    subfolder_name = f"step{step}"
    subfolder_path = os.path.join(output_dir, subfolder_name)

    if os.path.isdir(subfolder_path):
        for filename in os.listdir(subfolder_path):
            if filename.endswith((".jpg", ".png")):
                image_path = os.path.join(subfolder_path, filename)
                image = cv2.imread(image_path)

                # Resize the background image to match the size of the image
                resized_background = cv2.resize(background, (image.shape[1], image.shape[0]))

                # Here you should overlay the image on the background
                # For simplicity, this example just replaces the background entirely
                # with the original image, which may not be your final intention
                cv2.imwrite(image_path, image)  # This line was likely intended to save the processed image

# Note: This snippet directly saves the original images without modification,
# as the correct approach to overlay or merge images with a new background
# depends on specific requirements, such as segmentation masks.
