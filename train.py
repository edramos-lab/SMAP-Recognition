
from itertools import cycle
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle
import torch.nn.functional as F
from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose, Resize, RandomHorizontalFlip, RandomVerticalFlip, RandomRotation, ColorJitter, ToTensor, Normalize
from torch.utils.data import DataLoader, Subset
import numpy as np
import os
from collections import Counter
import torch
import torch.nn as nn
import timm
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose, Resize, RandomHorizontalFlip, RandomVerticalFlip, RandomRotation, ColorJitter, ToTensor, Normalize
from collections import Counter
from sklearn.model_selection import KFold
import wandb
from sklearn.metrics import accuracy_score, auc, classification_report, confusion_matrix, precision_score, recall_score, f1_score, matthews_corrcoef, roc_curve
import seaborn as sns
import argparse
import torch
import mlflow
import mlflow
from mlflow.tracking import MlflowClient
import mlflow.pytorch
import mlflow.sklearn
import numpy as np
from torch.utils.data import DataLoader, Subset
from collections import Counter
import datetime
import argparse
import random
import os
import glob as glob
import csv
import cv2

class_names = [f'step{i}' for i in range(1, 39)]

def plot_scatter_dataset(image_list):

    with open('image_statistics.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Image', 'Width', 'Height'])  # Write header row
        for image_path in image_list:
            img = cv2.imread(image_path)
            height, width, _ = img.shape
            writer.writerow([image_path, width, height])  # Write image statistics to CSV

    # Read the image statistics from the CSV file
    image_stats = pd.read_csv('image_statistics.csv')

    # Plot the scatter plot
    plt.scatter(image_stats['Width'], image_stats['Height'])
    plt.xlabel('Width')
    plt.ylabel('Height')
    plt.title('dataset Statistics')
    plt.savefig("scatter_plot.png")
    return plt


def show_images(images, labels, ncols=4):
    nrows = (len(images) + ncols - 1) // ncols
    fig, axs = plt.subplots(nrows, ncols, figsize=(12, 12))
    axs = axs.flatten()
    for i, (image, label) in enumerate(zip(images, labels)):
        img = image.permute(1, 2, 0).numpy()
        axs[i].imshow(img)
        axs[i].set_title(class_names[label])
        axs[i].axis('off')
    for ax in axs[i+1:]:
        ax.axis('off')
    plt.tight_layout()
    plt.savefig("random_images.png")

# Load random samples from the test dataset
def get_random_samples(dataloader, num_samples=16):
    dataset = dataloader.dataset
    indices = list(range(len(dataset)))
    random_indices = random.sample(indices, num_samples)
    images, labels = [], []
    for idx in random_indices:
        image, label = dataset[idx]
        images.append(image)
        labels.append(label)
    return images, labels

def preprocess_and_load_data(dataset_multiplier,dataset_folder, image_size, batch_size, subset_ratio):
    """
    Preprocesses the dataset, loads it into DataLoader, and creates a balanced subset of the training dataset.

    Args:
    - dataset_folder: Path to the dataset folder.
    - image_size: Tuple of ints for the size of the images (height, width).
    - batch_size: Batch size for loading the data.
    - subset_ratio: Fraction of data to use in the subset for each class.

    Returns:
    - A dictionary containing 'train', 'val', and 'test' DataLoaders.
    - subset_dataset: A balanced subset of the training dataset.
    - balancing_efficiency: The efficiency of balancing the dataset.
    - num_classes: The number of classes in the dataset.
    """
    data_transforms = Compose([
        Resize(image_size),
        RandomHorizontalFlip(),
        RandomVerticalFlip(),
        RandomRotation(45),
        ColorJitter(brightness=0.05, contrast=0.05, saturation=0.05),  # Reduced effect
        ToTensor(),
        # Updated Normalize values (example only; calculate based on your dataset)
        Normalize(mean=[0.7741, 0.8521, 0.9056], std=[0.1562, 0.1865, 0.1919]),
    ])

    # Load datasets
    dataset = ImageFolder(os.path.join(dataset_folder, 'train'), transform=data_transforms)
    dataclasses = ImageFolder(dataset_folder+"/train")
    image_list = glob.glob(dataset_folder+"/train"+'/*/*.jpg')
    plot_scatter_dataset(image_list)
    mlflow.log_artifact("scatter_plot.png")

    num_classes = len(dataclasses.classes)
    print("Num classes: ",num_classes)
    expanded_dataset = torch.utils.data.ConcatDataset([dataset] * dataset_multiplier)
    
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

    # Assuming the original dataset `expanded_dataset` has a 'targets' attribute
    original_targets = dataset.targets  # Access targets from the original dataset

    # Modify your dataloaders as before
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    print("Train Dataloader Size:", len(train_dataloader.dataset))

    import matplotlib.pyplot as plt
    class_names = [f'step{i}' for i in range(1, 39)]


    # Get random images and their labels
    random_images, random_labels = get_random_samples(test_dataloader, 16)

    # Show the images
    show_images(random_images, random_labels)
    mlflow.log_artifact("random_images.png")


    indices = []
    subset_indices = []  # This will hold the indices of the original dataset to form the subset

    # Update: Adjusting for the `Subset` indices mapping
    for class_index in range(num_classes):
        # Find indices in the original dataset matching each class
        class_indices = np.where(np.array(original_targets) == class_index)[0]
        # Find intersection with train_dataset indices to ensure correctness after split
        class_indices = [i for i in class_indices if i in train_dataset.indices]
        np.random.shuffle(class_indices)
        subset_size = int(len(class_indices) * subset_ratio)
        subset_indices.extend(class_indices[:subset_size])

    # Now we need to map subset_indices to the indices in the train_dataset subset
    mapped_indices = [train_dataset.indices.index(i) for i in subset_indices]

    subset_dataset = Subset(train_dataset, mapped_indices)
    subset_loader = DataLoader(subset_dataset, batch_size=batch_size, shuffle=True)

    # Calculate balancing efficiency using the subset's indices in the context of the original dataset
    class_counts = Counter([original_targets[i] for i in subset_indices])
    max_samples = max(class_counts.values())
    balancing_efficiency = len(subset_indices) / (num_classes * max_samples)
    print(f"Balancing efficiency: {balancing_efficiency}")
    return {
        'train': train_dataloader,
        'val': val_dataloader,
        'test': test_dataloader,
        'subset': subset_loader
    }, subset_dataset, balancing_efficiency, num_classes,total_samples,train_size,test_size

def preprocess_and_load_data2(dataset_folder, image_size, batch_size, subset_ratio):
    """
    Preprocesses the dataset, loads it into DataLoader, and creates a balanced subset of the training dataset.

    Args:
    - dataset_folder: Path to the dataset folder.
    - image_size: Tuple of ints for the size of the images (height, width).
    - batch_size: Batch size for loading the data.
    - subset_ratio: Fraction of data to use in the subset for each class.

    Returns:
    - A dictionary containing 'train', 'val', and 'test' DataLoaders.
    - subset_dataset: A balanced subset of the training dataset.
    - balancing_efficiency: The efficiency of balancing the dataset.
    - num_classes: The number of classes in the dataset.
    """
    data_transforms = Compose([
        Resize(image_size),
        RandomHorizontalFlip(),
        RandomVerticalFlip(),
        RandomRotation(45),
        ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Load datasets
    train_dataset = ImageFolder(os.path.join(dataset_folder, 'train'), transform=data_transforms)
    val_dataset = ImageFolder(os.path.join(dataset_folder, 'valid'), transform=data_transforms)
    test_dataset = ImageFolder(os.path.join(dataset_folder, 'test'), transform=data_transforms)

    num_classes = len(train_dataset.classes)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # Creating a balanced subset
    indices = []
    for class_index in range(num_classes):
        class_indices = np.where(np.array(train_dataset.targets) == class_index)[0]
        np.random.shuffle(class_indices)
        subset_size = int(len(class_indices) * subset_ratio)
        indices.extend(class_indices[:subset_size])

    subset_dataset = Subset(train_dataset, indices)
    subset_loader = DataLoader(subset_dataset, batch_size=batch_size, shuffle=True)

    # Calculate balancing efficiency
    class_counts = Counter([train_dataset.targets[i] for i in indices])
    max_samples = max(class_counts.values())
    balancing_efficiency = len(indices) / (num_classes * max_samples)
    print(f"Balancing efficiency: {balancing_efficiency}")
    return {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader,
        'subset': subset_loader
    }, subset_dataset, balancing_efficiency, num_classes

def train_model_kfold_wandb(subset_dataset, project_name,architecture,lr, n_splits,epochs, num_classes, batch_size):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    wandb.init(project=project_name, config={"architecture": architecture, "epochs": epochs, "batch_size": batch_size})

    for fold, (train_idx, val_idx) in enumerate(kf.split(subset_dataset)):
        print(f"Training fold {fold+1} for {architecture}")
        
        # Model initialization
        model = timm.create_model(architecture, pretrained=True, num_classes=num_classes).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
        criterion = nn.CrossEntropyLoss()

        # Subset training and validation loaders
        train_sampler = SubsetRandomSampler(train_idx)
        val_sampler = SubsetRandomSampler(val_idx)
        train_loader = DataLoader(subset_dataset, batch_size=batch_size, sampler=train_sampler)
        val_loader = DataLoader(subset_dataset, batch_size=batch_size, sampler=val_sampler)

        for epoch in range(epochs):
            model.train()
            train_loss, train_correct, train_total = 0, 0, 0
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                train_correct += torch.sum(preds == labels).item()
                train_total += labels.size(0)
            
            # Validation accuracy
            model.eval()
            val_correct, val_total = 0, 0
            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    _, preds = torch.max(outputs, 1)
                    val_correct += torch.sum(preds == labels).item()
                    val_total += labels.size(0)
            
            # Logging metrics
            wandb.log({
                "fold": fold+1,
                "epoch": epoch+1,
                "train_loss": train_loss / len(train_loader),
                "train_accuracy": 100.0 * train_correct / train_total,
                "val_accuracy": 100.0 * val_correct / val_total,
            })
        scheduler.step()
    return model, optimizer, scheduler

def train_model_kfold_mlflow(subset_dataset, project_name, architecture, lr, n_splits, epochs, num_classes, batch_size):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    #mlflow.set_experiment(project_name)

    for fold, (train_idx, val_idx) in enumerate(kf.split(subset_dataset)):
        print(f"Training fold {fold+1} for {architecture}")
        
      
        # Model initialization
        model = timm.create_model(architecture, pretrained=True, num_classes=num_classes).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
        criterion = nn.CrossEntropyLoss()

        # Subset training and validation loaders
        train_sampler = SubsetRandomSampler(train_idx)
        val_sampler = SubsetRandomSampler(val_idx)
        train_loader = DataLoader(subset_dataset, batch_size=batch_size, sampler=train_sampler)
        val_loader = DataLoader(subset_dataset, batch_size=batch_size, sampler=val_sampler)
        with mlflow.start_run(nested=True):
            for epoch in range(epochs):
                model.train()
                train_loss, train_correct, train_total = 0, 0, 0
                for images, labels in train_loader:
                    images, labels = images.to(device), labels.to(device)
                    optimizer.zero_grad()
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    train_loss += loss.item()
                    _, preds = torch.max(outputs, 1)
                    train_correct += torch.sum(preds == labels).item()
                    train_total += labels.size(0)
                
                # Validation accuracy
                model.eval()
                val_correct, val_total = 0, 0
                with torch.no_grad():
                    for images, labels in val_loader:
                        images, labels = images.to(device), labels.to(device)
                        outputs = model(images)
                        _, preds = torch.max(outputs, 1)
                        val_correct += torch.sum(preds == labels).item()
                        val_total += labels.size(0)
                
                # Logging metrics
                mlflow.log_metric("train_loss", train_loss / len(train_loader), step=epoch+1)
                mlflow.log_metric("train_accuracy", 100.0 * train_correct / train_total, step=epoch+1)
                mlflow.log_metric("val_accuracy", 100.0 * val_correct / val_total, step=epoch+1)
                mlflow.log_metric("fold", fold+1)

            scheduler.step()

    return model, optimizer, scheduler

def test_model_wandb(model, test_loader, architecture, optimizer, scheduler, batch_size, image_size):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    test_accuracy = 0  # Placeholder for accuracy calculation

    # Initialize lists to store true and predicted labels
    true_labels = []
    predicted_labels = []

    # Evaluate the model on the test split
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            correct += preds.eq(labels).sum().item()
            total += len(labels)

            # For detailed metrics
            true_labels.extend(labels.cpu().numpy())
            predicted_labels.extend(preds.cpu().numpy())

    # Convert lists to NumPy arrays for sklearn metrics
    true_labels = np.array(true_labels)
    predicted_labels = np.array(predicted_labels)

    # Calculate metrics
    confusion = confusion_matrix(true_labels, predicted_labels)
    test_accuracy = 100 * accuracy_score(true_labels, predicted_labels)
    test_precision = 100 * precision_score(true_labels, predicted_labels, average='weighted')
    test_recall = 100 * recall_score(true_labels, predicted_labels, average='weighted')
    test_f1_score = 100 * f1_score(true_labels, predicted_labels, average='weighted')
    matthews_corr = 100 * matthews_corrcoef(true_labels, predicted_labels)

    # Log metrics

    # Calculate the confusion matrix
    confusion = confusion_matrix(true_labels, predicted_labels)

    # Convert the confusion matrix to a DataFrame
    confusion_df = pd.DataFrame(confusion)

    # Save the DataFrame as a CSV file
    confusion_df.to_csv("confusion_matrix.csv", index=False)

    # Plot and save the confusion matrix as a .png file
    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.savefig("confusion_matrix.png")

    # Generate the confusion report
    confusion_report = classification_report(true_labels, predicted_labels)

    # Save the classification report to a text file
    report_filename = f"classification_report_{architecture}.txt"
    with open(report_filename, "w") as report_file:
        report_file.write(confusion_report)

    wandb.log({
        "Test Accuracy": test_accuracy,
        "Test Precision": test_precision,
        "Test Recall": test_recall,
        "Test F1 Score": test_f1_score,
        "Matthews Correlation Coefficient": matthews_corr,
        "Confusion Matrix": wandb.Image("confusion_matrix.png"),
        "Confusion Report": confusion_report
    })

    print("Test accuracy: %.3f" % test_accuracy)
    print("Confusion Matrix:\n", confusion)
    print("Test Precision: {:.6f}".format(test_precision))
    print("Test Recall: {:.6f}".format(test_recall))
    print("Test F1 Score: {:.6f}".format(test_f1_score))
    print("Matthews Correlation Coefficient: {:.6f}".format(matthews_corr))

    # Save the model to WandB
    model_path = "model_{}.pth".format(architecture)
    torch.save(model.state_dict(), model_path)
    artifact = wandb.Artifact(architecture, type="model")
    artifact.add_file(model_path)
    wandb.log_artifact(artifact)

    # Save the model locally
    local_model_path = "local_model_{}.pth".format(architecture)
    torch.save(model.state_dict(), local_model_path)

    roc_fig = auroc(model, test_loader, num_classes)
    wandb.log({"ROC Curve": wandb.Image(roc_fig)})

    # Clean up CUDA memory
    torch.cuda.reset_max_memory_allocated()
    torch.cuda.empty_cache()

    wandb.finish()


def test_model_mlflow(model, test_loader,architecture, optimizer, scheduler, batch_size, image_size):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    test_accuracy = 0  # Placeholder for accuracy calculation

    # Initialize lists to store true and predicted labels
    true_labels = []
    predicted_labels = []

    # Evaluate the model on the test split
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            correct += preds.eq(labels).sum().item()
            total += len(labels)

            # For detailed metrics
            true_labels.extend(labels.cpu().numpy())
            predicted_labels.extend(preds.cpu().numpy())

    # Convert lists to NumPy arrays for sklearn metrics
    true_labels = np.array(true_labels)
    predicted_labels = np.array(predicted_labels)

    # Calculate metrics
    confusion = confusion_matrix(true_labels, predicted_labels)
    test_accuracy = 100 * accuracy_score(true_labels, predicted_labels)
    test_precision = 100 * precision_score(true_labels, predicted_labels, average='weighted')
    test_recall = 100 * recall_score(true_labels, predicted_labels, average='weighted')
    test_f1_score = 100 * f1_score(true_labels, predicted_labels, average='weighted')
    matthews_corr = 100 * matthews_corrcoef(true_labels, predicted_labels)

    # Log metrics

    # Calculate the confusion matrix
    confusion = confusion_matrix(true_labels, predicted_labels)

    # Convert the confusion matrix to a DataFrame
    confusion_df = pd.DataFrame(confusion)

    # Save the DataFrame as a CSV file
    confusion_df.to_csv("confusion_matrix.csv", index=False)

    # Plot and save the confusion matrix as a .png file
    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.savefig("confusion_matrix.png")

    # Generate the confusion report
    confusion_report = classification_report(true_labels, predicted_labels)

    # Save the classification report to a text file
    report_filename = f"classification_report_{architecture}.txt"
    with open(report_filename, "w") as report_file:
        report_file.write(confusion_report)

    # Log metrics using MLflow
    mlflow.log_metric("Test Accuracy", test_accuracy)
    mlflow.log_metric("Test Precision", test_precision)
    mlflow.log_metric("Test Recall", test_recall)
    mlflow.log_metric("Test F1 Score", test_f1_score)
    mlflow.log_metric("Matthews Correlation Coefficient", matthews_corr)
    mlflow.log_artifact("confusion_matrix.png")
    
    mlflow.log_artifact(report_filename)


    print("Test accuracy: %.3f" % test_accuracy)
    print("Confusion Matrix:\n", confusion)
    print("Test Precision: {:.6f}".format(test_precision))
    print("Test Recall: {:.6f}".format(test_recall))
    print("Test F1 Score: {:.6f}".format(test_f1_score))
    print("Matthews Correlation Coefficient: {:.6f}".format(matthews_corr))

    # Save the model using MLflow with a timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    model_name = f"model_{timestamp}"
    mlflow.pytorch.log_model(model, model_name)
    mlflow.pytorch.save_model(model, model_name)
    roc_fig = auroc2(model, test_loader, num_classes)

    mlflow.log_artifact("roc-auc.png")

    # Clean up CUDA memory
    torch.cuda.reset_max_memory_allocated()
    torch.cuda.empty_cache()

    mlflow.end_run()

def auroc(model, test_loader, num_classes):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.eval()
    y_test = []
    y_score = []
    with torch.no_grad():
        for i, (inputs, classes) in enumerate(test_loader):
            inputs = inputs.to(device)
            y_test.append(F.one_hot(classes, num_classes).numpy())

            try:
                bs, ncrops, c, h, w = inputs.size()
            except:
                bs, c, h, w = inputs.size()
                ncrops = 1
            if ncrops > 1:
                outputs = model(inputs.view(-1, c, h, w))
                outputs = outputs.view(bs, ncrops, -1).mean(1)
            else:
                outputs = model(inputs)
            y_score.append(outputs.cpu().numpy())
    y_test = np.array([t.ravel() for t in y_test])
    y_score = np.array([t.ravel() for t in y_score])

    # Compute ROC curve and ROC area for each class in each fold
    fpr = dict()
    tpr = dict()
    local_roc_auc = dict()
    for i in range(num_classes):
        fpr[i], tpr[i], _ = roc_curve(np.array(y_test[:, i]), np.array(y_score[:, i]))
        local_roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
    local_roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # Compute macro-average ROC curve and ROC area
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(num_classes)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(num_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= num_classes
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    local_roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # Plot ROC curve
    plt.figure(figsize=(12, 10))
    plt.plot(fpr["micro"], tpr["micro"], label='micro-average ROC curve (area = {0:0.2f})'.format(local_roc_auc["micro"]), color='deeppink', linestyle=':', linewidth=2)
    plt.plot(fpr["macro"], tpr["macro"], label='macro-average ROC curve (area = {0:0.2f})'.format(local_roc_auc["macro"]), color='navy', linestyle=':', linewidth=2)
    for i in range(num_classes):
        plt.plot(fpr[i], tpr[i], label='ROC curve of class {0} (area = {1:0.2f})'.format(i, local_roc_auc[i]))
    plt.plot([0, 1], [0, 1], 'k--', linewidth=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")

    # Save the plot as "roc_curve.png"

    plt.savefig('roc-auc.png')
    return plt

def auroc2(model, test_loader, num_classes):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    y_test = []
    y_score = []
    
    with torch.no_grad():
        for inputs, classes in test_loader:
            inputs = inputs.to(device)
            classes = classes.to(device)
            
            outputs = model(inputs)
            
            # Convert classes to one-hot encoding
            y_test.append(F.one_hot(classes, num_classes=num_classes).cpu().numpy())
            
            # Assume outputs are raw scores that need softmax
            y_score.append(F.softmax(outputs, dim=1).cpu().numpy())
    
    # Stack all batch-wise arrays to create a single array for true labels and scores
    y_test = np.vstack(y_test)
    y_score = np.vstack(y_score)

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(num_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # Aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(num_classes)]))

    # Then interpolate all ROC curves at these points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(num_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= num_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # Plot all ROC curves
    plt.figure(figsize=(10, 8))
    plt.plot(fpr["micro"], tpr["micro"],
             label='Micro-average ROC curve (area = {0:0.2f})'.format(roc_auc["micro"]),
             color='deeppink', linestyle=':', linewidth=4)

    plt.plot(fpr["macro"], tpr["macro"],
             label='Macro-average ROC curve (area = {0:0.2f})'.format(roc_auc["macro"]),
             color='navy', linestyle=':', linewidth=4)

    colors = iter(plt.cm.rainbow(np.linspace(0, 1, num_classes)))
    for i, color in zip(range(num_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label='ROC curve of class {0} (area = {1:0.2f})'.format(i, roc_auc[i]))

    plt.figure(figsize=(12, 10))
    plt.plot(fpr["micro"], tpr["micro"], label='micro-average ROC curve (area = {0:0.2f})'.format(roc_auc["micro"]), color='deeppink', linestyle=':', linewidth=2)
    plt.plot(fpr["macro"], tpr["macro"], label='macro-average ROC curve (area = {0:0.2f})'.format(roc_auc["macro"]), color='navy', linestyle=':', linewidth=2)
    for i in range(num_classes):
        plt.plot(fpr[i], tpr[i], label='ROC curve of class {0} (area = {1:0.2f})'.format(i, roc_auc[i]))
    plt.plot([0, 1], [0, 1], 'k--', linewidth=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.savefig('roc-auc.png')
    plt.close()
    return plt

    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-dataset','--dataset_folder', help='Path to the dataset folder', required=False,default="/home/edramos/Documents/MLOPS/ImageClassification-MFG/things-8")
    parser.add_argument('-lr','--lr', help='Learning rate for the optimizer', required=False, default=0.0001,type=float)
    parser.add_argument('-batchsize','--batch_size', help='Batch size for training', required=False, default=8, type=int)
    parser.add_argument('-epochs','--epochs', help='Number of epochs for training', required=False, default=10, type=int)
    args = parser.parse_args()
    dataset_folder = args.dataset_folder
    lr = args.lr
    batch_size = args.batch_size
    print(f"lr: {lr}, type: {type(lr)}")
    print(f"batch_size: {batch_size}, type: {type(batch_size)}")
    print(f"epochs: {args.epochs}, type: {type(args.epochs)}")



    
    print(f"dataset_folder: {dataset_folder}, type: {type(dataset_folder)}")  
    # Set up default values
    n_splits = 5
    #epochs = 10
    model = "convnextv2_tiny"#"efficientnet_b0"
    #lr = 0.0001
    #batch_size = 8
    subset_ratio = 0.95
    project_name = "SmartAssemblyProcess"
    #dataset_folder = "/home/edramos/Documents/MLOPS/SmartAssemblyProcessRecognition/CustomDataset/"
    mlflowuri = "http://127.0.0.1:5000"
    dataset_multiplier = 1

    # Connect to MLflow server
    mlflow.set_tracking_uri("http://127.0.0.1:5000")

    # Set the active MLflow project
    mlflow.set_experiment("SmartAssemblyProcess")
    with mlflow.start_run():


        #dataset_folder = '/home/edramos/Documents/MLOPS/ImageClassification-MFG/nigel-chassises-1'
        if dataset_folder == None:
            dataset_folder = '/home/edramos/Documents/MLOPS/SmartAssemblyProcessRecognition/CustomDataset/'
        image_size = (225, 224)  # Example image size
        data_loaders, subset_dataset, balancing_efficiency, num_classes,total_samples,train_size,test_size = preprocess_and_load_data(dataset_multiplier,dataset_folder, image_size, batch_size, subset_ratio)

        # Example of how to use the data_loaders and subset_dataset
        print(f"Number of classes: {num_classes}")
        print(f"Balancing Efficiency: {balancing_efficiency}")
        for images, labels in data_loaders['train']:
            print(f'Train Batch size: {len(images)}')
            break  # Just to show the first batch, you can remove this break to iterate through the dataset

        for images, labels in data_loaders['subset']:
            print(f'Subset Batch size: {len(images)}')
            break  # Just to show the first batch from the subset

        #architectures = ["efficientnet_b0", "inception_v4", "swin_tiny_patch4_window7_224", "convnextv2_tiny", "xception41", "deit3_base_patch16_224"]
        architecture= model
           
        
        mlflow.log_param("architecture", architecture)
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("lr", lr)
        mlflow.log_param("total_samples", total_samples)
        mlflow.log_param("subset_ratio", subset_ratio)
        mlflow.log_param("train_size", train_size)
        mlflow.log_param("test_size", test_size)
        mlflow.log_param("dataset_multiplier", dataset_multiplier)
        mlflow.log_param("n_splits", n_splits)
        mlflow.log_param("balancing_efficiency", balancing_efficiency)
        mlflow.log_param("image_size", image_size)


        
        model, optimizer, scheduler =train_model_kfold_mlflow(subset_dataset, project_name,architecture, lr,n_splits,epochs, num_classes, batch_size)
        
        test_loader = data_loaders['test']

        test_model_mlflow(model, test_loader,architecture, optimizer, scheduler, batch_size, image_size)
    mlflow.end_run()
    
    