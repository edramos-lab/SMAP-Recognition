import cv2
import torch
import torchvision.transforms as transforms
import numpy as np
import argparse



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-model_path','--model_path', help='Path to the dataset folder', required=False,default="/home/edramos/Documents/MLOPS/model_20240312030133/content/model_20240312030133/data/model.pth")

    args = parser.parse_args()
    model_path = args.model_path

    print(f"model_path: {model_path}, type: {type(model_path)}")
    # Check if GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the model
    model_path = model_path#'/home/edramos/Documents/MLOPS/model_20240312030133/content/model_20240312030133/data/model.pth'
    model = torch.load(model_path)
    model = model.to(device)
    model.eval()

    # Set up the webcam
    cap = cv2.VideoCapture(0)


    font                   = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (10,450)
    fontScale              = 1
    fontColor              = (255,255,255)
    thickness              = 1
    lineType               = 2

    class_names = [f"step{i}" for i in range(1, 38)]
    while True:
        # Capture frame from the webcam
        ret, frame = cap.read()

        # Preprocess the frame
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        frame = transform(frame)
        frame = frame.unsqueeze(0)
        frame = frame.to(device)

        # Run inference
        with torch.no_grad():
            prediction = model(frame)
            # Assuming the model outputs a single class probability
            class_probability = torch.softmax(prediction, dim=1)[0]

        # Transfer the results back to CPU if necessary
        class_probability = class_probability.cpu()

        frame = frame.squeeze().permute(1, 2, 0).cpu().numpy()

        # Convert from float (0.0 to 1.0) to uint8 (0 to 255)
        frame = (frame * 255).astype(np.uint8)

        # Since the normalization might have altered colors, direct conversion without denormalization
        # might lead to unusual color representations, but it will be suitable for display.


        
        # Display the result
        class_label = torch.argmax(class_probability).item()
        
        print(class_label)         # Display the class label
        label = f"Class: {class_label}"
        class_name_predicted=class_names[class_label]
        print(class_name_predicted)
        frame = cv2.cvtColor(src=frame, code=cv2.COLOR_RGB2BGR)
        cv2.putText(frame,class_name_predicted, 
        bottomLeftCornerOfText, 
        font, 
        fontScale,
        fontColor,
        thickness,
        lineType)
        print(class_name_predicted)
        cv2.imshow('Webcam', frame)
        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the webcam and close the window
    cap.release()
    cv2.destroyAllWindows()