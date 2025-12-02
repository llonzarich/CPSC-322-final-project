import numpy as np


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

# =============== NAIVE BAYES CLASSIFIER SECTION =====================





