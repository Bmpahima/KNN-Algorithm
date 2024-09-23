import pandas as pd
import math


def euclidian_distance(record1, record2):
    """
    Calculate the Euclidean distance between two records.

    Parameters:
    record1 (list): The first record (list of numerical features).
    record2 (list): The second record (list of numerical features).

    Returns:
    float: The Euclidean distance between the two records, rounded to 7 decimal places.
    """
    sub_record = [(record1[i] - record2[i]) ** 2 for i in range(len(record1))]
    return round(abs(math.sqrt(sum(sub_record))), 7)


def data_scaling(train_df, test_df):
    """
    Normalize the data in a DataFrame using min-max scaling.

    Parameters:
    df (pandas.DataFrame): The DataFrame to be scaled.

    Returns:
    pandas.DataFrame: The normalized DataFrame.
    """
    train_df = train_df.astype('float64')
    test_df = test_df.astype('float64')
    for i in range(train_df.shape[1]):
        column = train_df.iloc[:, i]
        max_in_column = max(column)
        min_in_column = min(column)
        for j in range(train_df.shape[0]):
            train_df.iloc[j, i] = round((train_df.iloc[j, i] - min_in_column) / (max_in_column - min_in_column), 7)
        for j in range(test_df.shape[0]):
            test_df.iloc[j, i] = round((test_df.iloc[j, i] - min_in_column) / (max_in_column - min_in_column), 7)
    return train_df, test_df



def k_nearest_neighbors(distances, k):
    """
    Identify the indices of the k smallest distances.

    Parameters:
    distances (list): List of distances.
    k (int): The number of nearest neighbors to find.

    Returns:
    list: Indices of the k nearest neighbors.
    """
    nearest = []
    for i in range(k):
        minimum = min(distances)
        nearest.append(distances.index(minimum))
        distances[distances.index(minimum)] = float('inf')
    return nearest


def create_probability(classes):
    """
    Calculate the probability of each class in a list.

    Parameters:
    classes (list): List of class labels (0 or 1).

    Returns:
    tuple: Probability of class 0 and class 1.
    """
    number_of_one = sum(classes)
    number_of_zero = len(classes) - number_of_one
    return number_of_zero / len(classes), number_of_one / len(classes)


def knn_predict(train_path, test_path, k):
    """
    Perform k-nearest neighbors prediction.

    Parameters:
    train_path (str): Path to the training dataset CSV file.
    test_path (str): Path to the testing dataset CSV file.
    k (int): The number of nearest neighbors to consider.

    Returns:
    list: List of tuples containing the probabilities of class 0 and class 1 for each test record.
    """
    test_df = pd.read_csv(test_path)
    train_df = pd.read_csv(train_path)

    # Normalize the data frames
    normal_train_df, normal_test_df = data_scaling(train_df.iloc[:, :-1], test_df)
    # Calculate distances
    test_set_distances = {}
    for i in range(normal_test_df.shape[0]):
        test_set_distances[i] = []
        current_test_features = normal_test_df.iloc[i, :].to_list()
        for j in range(normal_train_df.shape[0]):
            current_train_features = normal_train_df.iloc[j, :].to_list()
            test_set_distances[i].append(euclidian_distance(record1=current_train_features, record2=current_test_features))
    # Find the neighbors
    nearest_points = {}
    for i in range(len(test_set_distances)):
        nearest_points[i] = k_nearest_neighbors(test_set_distances[i], k)
    # Classification
    train_classes = {}
    for i in range(len(nearest_points)):
        train_classes[i] = []
        for j in nearest_points[i]:
            train_classes[i].append(int(train_df.iloc[j, -1]))
    list_of_probabilities = [create_probability(train_classes[i]) for i in range(len(train_classes))]
    return list_of_probabilities


def main():
    """
    Main function to perform k-nearest neighbors classification and print the results.
    """
    k = 5
    probs = knn_predict("train_set.csv", "test_set.csv", k)
    for i in range(len(probs)):
        print("{0} classified as {1}".format(probs[i], "1" if probs[i][1] > probs[i][0] else "0"))


main()
