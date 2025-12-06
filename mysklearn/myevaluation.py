import numpy as np
from mysklearn import myutils
import math


# ======================= GENERAL EVALUATION SECTION ==========================
def train_test_split(X, y, test_size=0.33, random_state=None, shuffle=True):
    """
        Purpose: Split dataset into train and test sets based on a test set size.

        Args:
            X (list of list of obj): - The list of samples. 
                                    - has shape: (n_samples, n_features)
            y (list of obj): - The target y values (labels corresponding to X)
                            - has shape:  n_samples
            test_size (float or int): - float for proportion of dataset to be in test set (e.g. 0.33 for a 2:1 split)
                                     - int for absolute number of instances to be in test set (e.g. 5 for 5 instances in test set)
            random_state (int): - integer used for seeding a random number generator for reproducible results
                               - Use random_state to seed your random number generator (you can use the math module or use numpy for your generator) (choose one and consistently use that generator throughout your code)
            shuffle (bool): whether or not to randomize the order of the instances before splitting
                Shuffle the rows in X and y before splitting and be sure to maintain the parallel order of X and y!!

        Returns:
            X_train(list of list of obj): The list of training samples
            X_test(list of list of obj): The list of testing samples
            y_train(list of obj): The list of target y values for training (parallel to X_train)
            y_test(list of obj): The list of target y values for testing (parallel to X_test)

        Note: - Loosely based on sklearn's train_test_split(): https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
    """
    # convert X and y to numpy arrays.
    X = np.array(X)
    y = np.array(y)

    # set random seed.
    if random_state is not None:
        np.random.seed(random_state)

    # create a list to store all the indices in the dataset. 
    indices = np.arange(len(X))

    # compute the number of instances for the test set, based on the information we're given.
    if isinstance(test_size, float):
        num_test = math.ceil(len(X) * test_size)
    else:
        num_test = test_size

    # shuffle indices if True
    # "find" the instances for the train and test sets.
    # note: splitting based off indices ensures the train and test sets will have corresponding instances.
    if shuffle == True:
        np.random.shuffle(indices)
        test_indices = indices[:num_test] # grab instances from index   0 --> train size index.
        train_indices = indices[num_test:] # grab instances from train size index --> last index in the indices list. 
    else:
        test_indices = indices[len(X) - num_test:]
        train_indices = indices[:len(X) - num_test]

    # create train set.
    X_train = X[train_indices]
    y_train = y[train_indices]

    # create the test set.
    X_test = X[test_indices]
    y_test = y[test_indices]

    # convert numpy arrays back to lists
    X_train = X_train.tolist()
    y_train = y_train.tolist()
    X_test = X_test.tolist()
    y_test = y_test.tolist()

    return X_train, X_test, y_train, y_test




def kfold_split(X, n_splits=5, random_state=None, shuffle=False):
    """
        Purpose: Split dataset into cross validation folds.

        Args:
            X (list of list of obj): - The list of samples
                                     - has shape:  (n_samples, n_features)
            n_splits (int): Number of folds.
            random_state (int): integer used for seeding a random number generator for reproducible results
            shuffle (bool): whether or not to randomize the order of the instances before creating folds

        Returns:
            folds (list of 2-item tuples): - The list of folds where each fold is defined as a 2-item tuple.
                                            (The first item in the tuple is the list of training set indices for the fold)
                                            (The second item in the tuple is the list of testing set indices for the fold)

        Notes:
            The first n_samples % n_splits folds have size n_samples // n_splits + 1. Other folds have size n_samples // n_splits, where n_samples is the number of samples (e.g. 11 samples and 4 splits, the sizes of the 4 folds are 3, 3, 3, 2 samples)
            Loosely based on sklearn's KFold split(): https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html
    """
    # convert X to numpy arrays.
    X = np.array(X)

    # find the number of instances in the dataset.
    num_instances = len(X) 

    # set random seed.
    if random_state is not None:
        np.random.seed(random_state)

    # create a list to store all the indices in the dataset. 
    indices = np.arange(len(X))

    # shuffle indices if shuffle=True.
    if shuffle == True:
        np.random.shuffle(indices)

    # compute the 5 split sizes.
    split_sizes = [num_instances // n_splits] * n_splits 
    
    # distribute the remainders.
    for i in range(num_instances % n_splits):
        split_sizes[i] += 1 

    # split X (initial dataset) into 5 equal splits, or "folds".
    folds = []
    counter = 0
    
    for split_size in split_sizes:
        test_indices = indices[counter:counter + split_size]
        train_indices = np.setdiff1d(indices, test_indices)

        # convert train and test indices back to lists. 
        train_indices = train_indices.tolist()
        test_indices = test_indices.tolist()
        
        # append splits to the given fold idx in the tuple.
        folds.append((train_indices, test_indices))

        # increment the counter (so we can get the indices of the next split).
        counter += split_size

    return folds


def stratified_kfold_split(X, y, n_splits=5, random_state=None, shuffle=False):
    """
        Purpose: Split dataset into stratified cross validation folds.

        Args:
            X (list of list of obj): - the list of instances (samples).
                                     - has shape: (n_samples, n_features)
            y(list of obj): - the  target y values (labels corresponding to X).
                            - has shape: n_samples
            n_splits (int): Number of folds.
            random_state (int): integer used for seeding a random number generator for reproducible results
            shuffle (bool): whether or not to randomize the order of the instances before creating folds

        Returns:
            folds (list of 2-item tuples): The list of folds where each fold is defined as a 2-item tuple
                                           - The first item in the tuple is the list of training set indices for the fold
                                           - The second item in the tuple is the list of testing set indices for the fold

        Notes:
            Loosely based on sklearn's StratifiedKFold split(): https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html#sklearn.model_selection.StratifiedKFold

        General Steps: 
            (1) randomize dataset
            (2) group by class: partition the dataset such that each partition contains all rows for a given class. The number of partitions must therefore = num_distinct_classes in the dataset.
            (3) generate folds: iterate through each partition and distribute the partition roughly by class label.
    """
    # convert X to numpy arrays.
    X = np.array(X)

    # set random seed.
    if random_state is not None:
        np.random.seed(random_state)

    # create a list to store all the indices in the dataset. 
    all_indices = np.arange(len(X))

    # ----------- (STEP 1) RANDOMIZE DATASET --------------
    # shuffle indices if shuffle=True.
    if shuffle == True:
        np.random.shuffle(all_indices)


    # ----------- (STEP 2) GROUP BY CLASS LABEL -----------
    # create a dictionary where each class label is a key and the value of that key is the list of all indices of the instances with that label.
    group_by_class = dict()

    # iterate through each instance in X to put it into a list corresponding to one of the keys in the dict.
    for idx, val in enumerate(X):
        class_label = y[idx] # get the label of the current X val (using the index of the current X val)
        
        if class_label not in group_by_class:
            group_by_class[class_label] = [] # create a key-val pair for a new key (class label) in the dict.
        
        group_by_class[class_label].append(idx) # add the index of the instance with the class label to the list of its corresponding key.

    
    # iterate through each key-value pair in the dictionary and shuffle the indices in that group.
    for label, indices in group_by_class.items():
        if shuffle == True:
            np.random.shuffle(indices)


    # ----------- (STEP 3) GENERATE FOLDS -----------
    # note: we want to generate 5 folds where the distribution of each class within each fold imitates the original distribution of the each class in the entire dataset.
    # first, we need to split each class group into k=5 equal folds, or "partitions".
    
    # create a dictionary where each class label's value (for each label / key in group_by_class dictionary) is an empty list. 
    class_folds = {label: [] for label in group_by_class}

    # iterate through each key-value pair in the dictionary to split each class group into k=5 subgroups, or "splits". 
    for label, indices in group_by_class.items():
        split_size = len(indices) // n_splits # compute the size of each split.

        # iterate n_splits=k=5 times to generate the 5 subgroups, or "splits".
        for i in range(n_splits):
            start_idx = i * split_size 
            
            if i < (n_splits - 1):
                end_idx = start_idx + split_size
            else: # if the split is the last split, we can just set the last index we'll grab from to be the length of the list of indices.
                end_idx = len(indices)
            
            split_indices = indices[start_idx:end_idx]
            class_folds[label].append(split_indices) # append the list of indices for the i-th split to the dictionary of its corresponding key, label.


    # now that each of the class groups are split into k=5 subgroups, we can combine subgroups together to make the full split, or fold. 
    folds = []

    # iterate n_splits=k=5 times to combine class group splits. 
    for i in range(n_splits):
        test_indices = []

        for label in class_folds:
            test_indices.extend(class_folds[label][i]) 
        
        # create the train set indices using the indices that were not included in the test set.
        train_indices = [idx for idx in all_indices if idx not in test_indices]
        
        # append splits to the given fold idx in the tuple.
        folds.append((train_indices, test_indices))

    return folds


def confusion_matrix(y_true, y_pred, labels):
    """
        Purpose: Compute confusion matrix to evaluate the accuracy of a classification.

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
        Loosely based on sklearn's confusion_matrix(): https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html
    """
    
    # creates a blank matrix so it can be added to later, instead of having to create/append when adding to matrix
    matrix = [[0 for _ in range(len(labels))] for _ in range(len(labels))]
    
    for loc in range(len(y_true)):
        matrix[labels.index(y_true[loc])][labels.index(y_pred[loc])] += 1 # adds to the specific column's (predicted class) specific row (actual class)

    return matrix


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


def binary_precision_score(y_true, y_pred, labels=None, pos_label=None):
    """
        Purpose: - Compute the precision (for binary classification)
                 - The precision is the ratio TP / (TP + FP) where tp is the number of true positives and fp the number of false positives.
                 - The precision is intuitively the ability of the classifier not to label as positive a sample that is negative
                 - The best value is 1 and the worst value is 0.

    Args:
        y_true(list of obj): - The ground_truth target y values
                             - has shape: n_samples
        y_pred (list of obj): - The predicted target y values (parallel to y_true)
                              - has shape: n_samples
        labels(list of obj): - The list of possible class labels
                             - If None, defaults to the unique values in y_true
        pos_label(obj): - The class label to report as the "positive" class.
                        - If None, defaults to the first label in labels

    Returns:
        precision (float): Precision of the positive class

    Notes:
        Loosely based on sklearn's precision_score(): https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_score.html
    """
    if labels is None:
        labels = list(set(y_true)) # labels are the unique values in y_true. 
        
    if pos_label is None:
        pos_label = labels[0]

    TP = 0 # the number of instances correctly predicted as positive label.
    FP = 0 # the number of instances predicted to be positive label, but not actually positive label.

    # iterate through each TRUE class label and PREDICTED class label.
    # use the zip() function to combine list iterables (y_true and y_pred). This way we can iterate through each side by side.
    for y1, y2 in zip(y_true, y_pred):
        if y2 not in labels:
            continue

        # if the predicted class label is positive, then the true class label should be positive as well. 
        if y2 == pos_label:
            if y1 == pos_label:
                TP += 1
            else:
                FP += 1

    # if denominator is 0, we want to ensure we return a float still: 0.0
    if TP + FP == 0:
        return 0.0
    
    precision = TP / (TP + FP) # compute precision.

    return precision


def multiclass_precision_score(y_true, y_pred, labels=None):
    '''
        Purpose: compute the precision (for multiclass classification). 
                 precision is the ratio of TP / (TP + FP)
                 to compute multiclass precision, we will compute precision per class, then take an avg (either macro or weighted).

        Args:
            y_true(list of obj): - The ground_truth target y values
                                - has shape: n_samples
            y_pred(list of obj): - The predicted target y values (parallel to y_true)
                                - has shape: n_samples
            labels(list of obj): - The list of possible class labels.
                                - If None, defaults to the unique values in y_true

        Returns:
            precision(float): 1 multiclass precision score
    '''
    # find all class labels.
    labels = list(set(y_true))

    precisions = []

    for label in labels:
        precision = binary_precision_score(y_true, y_pred, labels=labels, pos_label=label) # treat current class as the "positive class" and the other 2 as the "negative class".
        precisions.append(precision)

    # find the avg precision (use weighted avg because the confident label is weighted much heavier)
    class_counts = [y_true.count(label) for label in labels]
    total_count = sum(class_counts)
    precision = sum(p * count for p, count in zip(precisions, class_counts)) / total_count

    return precision


def binary_recall_score(y_true, y_pred, labels=None, pos_label=None):
    """
        Purpose: - Compute the recall (for binary classification)
                 - The recall is the ratio TP / (TP + FN) where tp is the number of true positives and fn the number of false negatives.
                 - The recall is intuitively the ability of the classifier to find all the positive samples.
                 - The best value is 1 and the worst value is 0.

    Args:
        y_true(list of obj): - The ground_truth target y values
                             - has shape: n_samples
        y_pred(list of obj): - The predicted target y values (parallel to y_true)
                             - has shape: n_samples
        labels(list of obj): - The list of possible class labels.
                             - If None, defaults to the unique values in y_true
        pos_label(obj): - The class label to report as the "positive" class.
                        - If None, defaults to the first label in labels

    Returns:
        recall(float): Recall of the positive class

    Notes:
        Loosely based on sklearn's recall_score(): https://scikit-learn.org/stable/modules/generated/sklearn.metrics.recall_score.html
    """
    if labels is None:
        labels = list(dict.fromkeys(y_true))

    if pos_label is None:
        pos_label = labels[0]

    TP = 0 # the number of instances correctly predicted as positive label.
    FN = 0 # the number of instances predicted to be negative label, but not actually negative label.

    # iterate through each TRUE class label and PREDICTED class label.
    # use the zip() function to combine list iterables (y_true and y_pred). This way we can iterate through each side by side.
    for y1, y2 in zip(y_true, y_pred):
        
        # if the predicted class label is positive, then the true class label should be positive as well. 
        if y2 == pos_label:
            if y1 == pos_label:
                TP += 1
        
        else:
            if y1 == pos_label:
                FN += 1 

    # if denominator is 0, we want to ensure we return a float still: 0.0
    if (TP + FN) == 0:
        return 0.0
    
    recall = TP / (TP + FN) # compute recall

    return recall


def multiclass_recall_score(y_true, y_pred, labels=None):
    '''
        Purpose: compute the recall for multiclass classification. 
                 recall is the ratio of TP / (TP + FN)
                 to compute multiclass recall, we will compute recall per class, then take an avg (either macro or weighted).

        Args:
            y_true(list of obj): - The ground_truth target y values
                                - has shape: n_samples
            y_pred(list of obj): - The predicted target y values (parallel to y_true)
                                - has shape: n_samples
            labels(list of obj): - The list of possible class labels.
                                - If None, defaults to the unique values in y_true

        Returns:
            precision(float): 1 multiclass recall score
    '''
    # find all class labels.
    labels = list(set(y_true))

    recalls = []

    for label in labels:
        recall = binary_recall_score(y_true, y_pred, labels=labels, pos_label=label) # treat current class as the "positive class" and the other 2 as the "negative class".
        recalls.append(recall)

    # find the avg recall (use weighted avg because the confident label is weighted much heavier)
    class_counts = [y_true.count(label) for label in labels]
    total_count = sum(class_counts)
    recall = sum(r * count for r, count in zip(recalls, class_counts)) / total_count

    return recall


def binary_f1_score(y_true, y_pred, labels=None, pos_label=None):
    """
        Purpose: - Compute the F1 score (for binary classification), also known as balanced F-score or F-measure.
                 - The F1 score can be interpreted as a harmonic mean of the precision and recall, where an F1 score reaches its best value at 1 and worst score at 0.
                 - The relative contribution of precision and recall to the F1 score are equal.
                 - The formula for the F1 score is: F1 = 2 * (precision * recall) / (precision + recall)

    Args:
        y_true(list of obj): - The ground_truth target y values
                             - has shape: n_samples
        y_pred(list of obj): - The predicted target y values (labels corresponding to y_true)
                             - has shape: n_samples
        labels(list of obj): - The list of possible class labels.
                             - If None, defaults to the unique values in y_true
        pos_label(obj): - The class label to report as the "positive" class.
                        - If None, defaults to the first label in labels

    Returns:
        f1(float): F1 score of the positive class

    Notes:
        Loosely based on sklearn's f1_score(): https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html
    """
    if labels is None:
        labels = list(dict.fromkeys(y_true))

    if pos_label is None:
        pos_label = labels[0]

    # compute precision and recall using functions defined above.
    precision = binary_precision_score(y_true, y_pred, labels=labels, pos_label=pos_label)
    recall = binary_recall_score(y_true, y_pred, labels=labels, pos_label=pos_label)

    # if denominator is 0, we want to ensure we return a float still: 0.0
    if precision + recall == 0:
        return 0.0
    
    f1 = ( 2 * (precision * recall) ) / (precision + recall) # compute f1.

    return f1 


def multiclass_f1_score(y_true, y_pred, labels=None):
    '''
        Purpose: compute the f1 score for multiclass classification. 
                 The formula for the F1 score is: F1 = 2 * (precision * recall) / (precision + recall)
                 to compute multiclass f1, we will compute f1 per class, then take an avg (either macro or weighted).

        Args:
            y_true(list of obj): - The ground_truth target y values
                                - has shape: n_samples
            y_pred(list of obj): - The predicted target y values (parallel to y_true)
                                - has shape: n_samples
            labels(list of obj): - The list of possible class labels.
                                - If None, defaults to the unique values in y_true

        Returns:
            precision(float): 1 multiclass recall score
    '''
    # find all class labels.
    labels = list(set(y_true))

    f1s = []

    for label in labels:
        f1 = binary_f1_score(y_true, y_pred, labels=labels, pos_label=label) # treat current class as the "positive class" and the other 2 as the "negative class".
        f1s.append(f1)

    # find the avg f1 (use weighted avg because the confident label is weighted much heavier)
    class_counts = [y_true.count(label) for label in labels]
    total_count = sum(class_counts)
    f1 = sum(f * count for f, count in zip(f1s, class_counts)) / total_count

    return f1


def binary_precision_score(y_true, y_pred, labels=None, pos_label=None):
    """
        Purpose: compute the precision (for binary classification). The precision is the ratio tp / (tp + fp) where tp is the number of true positives and fp the number of false positives.
                 The precision is intuitively the ability of the classifier not to label as positive a sample that is negative. The best value is 1 and the worst value is 0.

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
        Loosely based on sklearn's precision_score(): https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_score.html
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
    """
        Purpose: Compute the recall (for binary classification). The recall is the ratio tp / (tp + fn) where tp is the number of true positives and fn the number of false negatives.
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