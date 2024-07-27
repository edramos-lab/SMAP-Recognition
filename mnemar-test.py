import sys
import numpy as np
from statsmodels.stats.contingency_tables import mcnemar
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import matplotlib.pyplot as plt

def load_confusion_matrices(file_path, has_header=True):
    # Load the confusion matrices from the CSV file
    matrices = np.loadtxt(file_path, delimiter=',', skiprows=1 if has_header else 0)
    
    # Count the number of elements in the first row
    num_elements = matrices.shape[1]
    
    # Split the concatenated matrices vertically
    num_matrices = matrices.shape[0] // num_elements
    conf_matrices = np.split(matrices, num_matrices)
    
    return conf_matrices

def count_matrices(conf_matrices):
    # Count the number of classes in the matrices
    num_classes = conf_matrices[0].shape[0]
    
    return num_classes

def count_classes(conf_matrices):
    # Count the number of matrices
    num_matrices = len(conf_matrices)
    
    return num_matrices

def calculate_manhattan_distance(conf_matrices):
    # Calculate the Manhattan distance based on the best accuracy and the accuracy of each of the others
    best_accuracy = np.max(np.diagonal(conf_matrices[0]))
    accuracies = np.diagonal(conf_matrices[1:])
    distance = np.sum(np.abs(accuracies - best_accuracy))
    
    return distance

def calculate_metrics(conf_matrix):
    true_positives = np.diag(conf_matrix)
    print("Shape of true_positives:", true_positives.shape)
    false_positives = np.sum(conf_matrix, axis=0) - true_positives
    print("Shape of false_positives:", false_positives.shape)
    false_negatives = np.sum(conf_matrix, axis=1) - true_positives
    print("Shape of false_negatives:", false_negatives.shape)
    true_negatives = np.sum(conf_matrix) - (true_positives + false_positives + false_negatives)
    print("Shape of true_negatives:", true_negatives.shape)

    accuracy = np.sum(true_positives) / np.sum(conf_matrix)
    precision = np.sum(true_positives) / (np.sum(true_positives) + np.sum(false_positives))
    recall = np.sum(true_positives) / (np.sum(true_positives) + np.sum(false_negatives))
    f1 = 2 * (precision * recall) / (precision + recall)

    return accuracy, precision, recall, f1

def plot_manhattan_distance(conf_matrices):
    # Calculate the Manhattan distance based on the best accuracy and the accuracy of each of the others
    best_accuracy = np.max(np.diagonal(conf_matrices[0]))
    distances = [np.sum(np.abs(np.diagonal(conf_matrix) - best_accuracy)) for conf_matrix in conf_matrices[1:]]
    
    # Plot the distances for each matrix
    for i, distance in enumerate(distances):
        plt.plot([i+1], [distance], marker='o')
    
    plt.xlabel('Confusion Matrix Pair')
    plt.ylabel('Manhattan Distance')
    plt.title('Manhattan Distance between Confusion Matrices')
    plt.show()

if __name__ == "__main__":
    # Check if the file path and distance option are provided as command line arguments
    if len(sys.argv) != 3:
        print("Usage: python mnemar-test.py <file_path> <distance_option>")
        sys.exit(1)
    
    # Get the file path and distance option from command line arguments
    file_path = sys.argv[1]
    distance_option = sys.argv[2]
    
    # Load the confusion matrices
    conf_matrices = load_confusion_matrices(file_path, has_header=True)
    
    # Count the number of classes and matrices
    num_classes = count_classes(conf_matrices)
    print("Number of classes:", num_classes)
    num_matrices = count_matrices(conf_matrices)
    print("Number of matrices:", num_matrices)
    
    # Calculate the number of instances where classifiers A and B disagree
    # Instances where A is correct but B is incorrect
    n_01 = np.sum((conf_matrices[0] - conf_matrices[1]) > 0)

    # Instances where A is incorrect but B is correct
    n_10 = np.sum((conf_matrices[1] - conf_matrices[0]) > 0)

    # Construct the contingency table
    contingency_table = np.array([[0, n_01],
                                  [n_10, conf_matrices[0].shape[0]]])

    # Perform McNemar's test using statsmodels
    result = mcnemar(contingency_table, exact=True)

    # Print the result
    print('Statistic:', result.statistic)
    print('P-value:', result.pvalue)
    
    # Calculate the distance based on the selected option
    if distance_option == "euclidean":
        distance = np.linalg.norm(conf_matrices[0] - conf_matrices[1])
        print("Euclidean Distance:", distance)
    elif distance_option == "manhattan":
        distance = calculate_manhattan_distance(conf_matrices)
        print("Manhattan Distance:", distance)
        plot_manhattan_distance(conf_matrices)
    elif distance_option == "frobenius_norm":
        distance = np.linalg.norm(conf_matrices[0] - conf_matrices[1], 'fro')
        print("Frobenius Norm:", distance)
    else:
        print("Invalid distance option. Please choose 'euclidean', 'manhattan', or 'frobenius_norm'.")
    
    # Calculate and print metrics for each matrix
    for i, conf_matrix in enumerate(conf_matrices):
        accuracy, precision, recall, f1 = calculate_metrics(conf_matrix)
        print(f"Metrics for Matrix {i+1}:")
        print("Accuracy:", accuracy)
        print("Precision:", precision)
        print("Recall:", recall)
        print("F1-score:", f1)
