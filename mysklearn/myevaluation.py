import numpy as np
from mysklearn import myutils

# =============== RANDOM FOREST CLASSIFIER SECTION ======================

def bootstrap_sample(X, y=None, n_samples=None, random_state=None):
    """
    Purpose: Split dataset into bootstrapped training set and out of bag test set.

    Args:
        X (list of list of obj): The list of samples
        y (list of obj): - The target y values (labels corresponding to X)
                         - Default is None (in this case, the calling code only wants to sample X)
        n_samples (int): - Number of samples to generate. If left to None (default) this is automatically set to the first dimension of X.
        random_state (int): integer used for seeding a random number generator for reproducible results

    Returns:
        X_sample (list of list of obj): The list of samples
        X_out_of_bag (list of list of obj): The list of "out of bag" samples (e.g. left-over samples that form the test set)
        y_sample (list of obj): - The list of target y values sampled (labels corresponding to X_sample)
                                - None if y is None
        y_out_of_bag(list of obj): - The list of target y values "out of bag" (labels corresponding to X_out_of_bag)
                                   - None if y is None
    Notes:
        Loosely based on sklearn's resample(): https://scikit-learn.org/stable/modules/generated/sklearn.utils.resample.html
        Sample indexes of X with replacement, then build X_sample and X_out_of_bag as lists of instances using sampled indexes (use same indexes to build y_sample and y_out_of_bag)
    """
    # set random seed.
    if random_state is not None:
        np.random.seed(random_state)

    # dataset has D rows.
    # select D rows with replacement (you might select the same row) ==> train set (typically 63.2% of the original rows).
    # the rows not selected (out of bag samples / the samples not chosen) ==> test set (typically 36.8% of the original rows).

    # determine the number of samples to choose for the train set (D = number of samples in the dataset, for a dataset with D rows)
    if n_samples is None:
        n_samples = len(X)

    # create a list of indices (with replacement). 
    indices = np.arange(len(X))
    random_indices = np.random.choice(indices, size=n_samples, replace=True) # indices for train set.
    out_of_bag_indices = np.setdiff1d(indices, random_indices) # indices for test set.

    # create train and test sets with indices.
    X_sample = [X[idx] for idx in random_indices]
    X_out_of_bag = [X[idx] for idx in out_of_bag_indices]

    if y is not None:
        y_sample = [y[idx] for idx in random_indices]
        y_out_of_bag = [y[idx] for idx in out_of_bag_indices]
    else:
        y_sample = None
        y_out_of_bag = None


    return X_sample, X_out_of_bag, y_sample, y_out_of_bag

def accuracy_score(y_true, y_pred, normalize=True):
    """
        Purpose: Compute the classification prediction accuracy score.

        Args:
            y_true (list of obj): - The ground_truth target y values
                                  - has shape: n_samples
            y_pred (list of obj): - The predicted target y values (corresponding to y_true)
                                 - has shape: n_samples
            normalize(bool): - If False, return the number of correctly classified samples.
                             - Otherwise, return the fraction of correctly classified samples.

        Returns:
            score (float): If normalize == True, return the fraction of correctly classified samples (float),
                        otherwise, return the number of correctly classified samples (int).

        Notes: - Loosely based on sklearn's accuracy_score(): https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html#sklearn.metrics.accuracy_score
               - acc = (TP + TN) / (P + N) = (TP + TN) / (TP + FP + TN + FN)
    """

    score = 0.0 # to store the number of correctly classified samples as a float.
    correct = 0 # to keep track of the total number of correctly predicted samples. This = (TP + TN)

    for idx in range(len(y_true)):
        if y_true[idx] == y_pred[idx]:
            correct += 1

    if normalize is True:
        score = correct / len(y_true) # ensure score is the fraction / proportion of correctly classified samples (float).
    else:
        score = correct # ensure score is the number of correctly classified samples (int).

    return score

# =============== GENERAL EVALUATION SECTION =====================

def confusion_matrix(y_true, y_pred, labels):
    """Compute confusion matrix to evaluate the accuracy of a classification.

    Args:
        y_true(list of obj): The ground_truth target y values
            The shape of y is n_samples
        y_pred(list of obj): The predicted target y values (parallel to y_true)
            The shape of y is n_samples
        labels(list of str): The list of all possible target y labels used to index the matrix

    Returns:
        matrix(list of list of int): Confusion matrix whose i-th row and j-th column entry
            indicates the number of samples with true label being i-th class
            and predicted label being j-th class

    Notes:
        Loosely based on sklearn's confusion_matrix():
            https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html
    """
    
    # creates a blank matrix so it can be added to later, instead of having to create/append when adding to matrix
    matrix = [[0 for _ in range(len(labels))] for _ in range(len(labels))]
    
    for loc in range(len(y_true)):
        matrix[labels.index(y_true[loc])][labels.index(y_pred[loc])] += 1 # adds to the specific column's (predicted class) specific row (actual class)

    return matrix

def binary_precision_score(y_true, y_pred, labels=None, pos_label=None):
    """Compute the precision (for binary classification). The precision is the ratio tp / (tp + fp)
        where tp is the number of true positives and fp the number of false positives.
        The precision is intuitively the ability of the classifier not to label as
        positive a sample that is negative. The best value is 1 and the worst value is 0.

    Args:
        y_true(list of obj): The ground_truth target y values
            The shape of y is n_samples
        y_pred(list of obj): The predicted target y values (parallel to y_true)
            The shape of y is n_samples
        labels(list of obj): The list of possible class labels. If None, defaults to
            the unique values in y_true
        pos_label(obj): The class label to report as the "positive" class. If None, defaults
            to the first label in labels

    Returns:
        precision(float): Precision of the positive class

    Notes:
        Loosely based on sklearn's precision_score():
            https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_score.html
    """

    if labels is None:
        labels = []
        unique_keys = myutils.get_frequency(y_true)
        for key in unique_keys:
            # since get_frequency returns the frequency of each unique value
            # appending the key only puts in the unique values of y
            labels.append(key) 
    
    if pos_label is None:
        pos_label = labels[0] # sets positive value to be the first label

    tp = 0
    for loc in range(len(y_true)):
        if (y_true[loc] == y_pred[loc]) and (y_true[loc] == pos_label): 
            # captures matching true and pred, AND labels that are only the positive label
            tp += 1

    tp_fp = tp
    for loc in range(len(y_true)):
        if (y_true[loc] != pos_label) and (y_pred[loc] == pos_label): 
            # captures only the true values that are NOT the positive label AND the predictions that incorrectly predict posiitve label (aka false positive)
            tp_fp += 1 # only adds false positives since true positives is already calculated and added earlier 

    if tp_fp == 0: # prevents division by zero
        return 0
    else:
        return tp/tp_fp
    
def binary_recall_score(y_true, y_pred, labels=None, pos_label=None):
    """Compute the recall (for binary classification). The recall is the ratio tp / (tp + fn) where tp is
        the number of true positives and fn the number of false negatives.
        The recall is intuitively the ability of the classifier to find all the positive samples.
        The best value is 1 and the worst value is 0.

    Args:
        y_true(list of obj): The ground_truth target y values
            The shape of y is n_samples
        y_pred(list of obj): The predicted target y values (parallel to y_true)
            The shape of y is n_samples
        labels(list of obj): The list of possible class labels. If None, defaults to
            the unique values in y_true
        pos_label(obj): The class label to report as the "positive" class. If None, defaults
            to the first label in labels

    Returns:
        recall(float): Recall of the positive class

    Notes:
        Loosely based on sklearn's recall_score():
            https://scikit-learn.org/stable/modules/generated/sklearn.metrics.recall_score.html
    """
    
    if labels is None:
        labels = []
        unique_keys = myutils.get_frequency(y_true)
        for key in unique_keys:
            # since get_frequency returns the frequency of each unique value
            # appending the key only puts in the unique values of y
            labels.append(key)
    
    if pos_label is None:
        pos_label = labels[0] # sets positive value to be the first label

    tp = 0
    for loc in range(len(y_true)):
        if (y_true[loc] == y_pred[loc]) and (y_true[loc] == pos_label): 
            # captures pred that correctly classified the TRUE label
            tp += 1

    tp_fn = tp
    for loc in range(len(y_true)):
        if (y_true[loc] == pos_label) and (y_pred[loc] != pos_label):
            # captures only the true values that are actually positive AND the pred that is incorrectly classified as negative (aka the false negatives)
            tp_fn += 1 # only adds false negatives since true positives is already calculated and added earlier

    if tp_fn == 0: # prevents division by zero
        return 0
    else:
        return tp/tp_fn
    
def binary_f1_score(y_true, y_pred, labels=None, pos_label=None):
    """Compute the F1 score (for binary classification), also known as balanced F-score or F-measure.
        The F1 score can be interpreted as a harmonic mean of the precision and recall,
        where an F1 score reaches its best value at 1 and worst score at 0.
        The relative contribution of precision and recall to the F1 score are equal.
        The formula for the F1 score is: F1 = 2 * (precision * recall) / (precision + recall)

    Args:
        y_true(list of obj): The ground_truth target y values
            The shape of y is n_samples
        y_pred(list of obj): The predicted target y values (parallel to y_true)
            The shape of y is n_samples
        labels(list of obj): The list of possible class labels. If None, defaults to
            the unique values in y_true
        pos_label(obj): The class label to report as the "positive" class. If None, defaults
            to the first label in labels

    Returns:
        f1(float): F1 score of the positive class

    Notes:
        Loosely based on sklearn's f1_score():
            https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html
    """

    if labels is None:
        labels = []
        unique_keys = myutils.get_frequency(y_true)
        for key in unique_keys:
            # since get_frequency returns the frequency of each unique value
            # appending the key only puts in the unique values of y
            labels.append(key)
    
    if pos_label is None:
        pos_label = labels[0] # sets positive value to be the first label

    # since F1 requries precision and recall, call the functions that calculate them to avoid redundant code
    precision = binary_precision_score(y_true, y_pred, labels, pos_label)
    recall = binary_recall_score(y_true, y_pred, labels, pos_label)

    if (precision + recall) == 0: # prevents division by zero
        return 0
    else:
        return (2 * (precision * recall / (precision + recall)))