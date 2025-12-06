import numpy as np
from mysklearn import myevaluation as myeval

# =============== RANDOM FOREST CLASSIFIER SECTION =====================

def get_frequency(column):
    """creates a dictionary for the frequency of each unique value in a list
    
    Parameters:
        column (list): a list that contains any values, so this method can find the frequency  of each unique value in the list

    Returns:
        freq (dictionary): a dictionary that has each unique value in the column parameter as a key, and the number of times the 
                    unique value appears in the column paramater list

    """    
    freq = {}

    for value in column:
        if value in freq:
            freq[value] += 1
        else:
            freq[value] = 1

    return freq


def all_same_values(y_data):
    """checks to see if all the class labels are the same (so a leaf node can be created)
    
    Parameters:
        y_data (list of objects): the class labels of a partition

    Returns:
        bool: returns True if all class labels are the same
        
    """
    first_val = y_data[0] # checks to see if all values are the same as the first value
    for y in y_data:
        if y != first_val:
            return False
    return True

def attribute_to_split(X_data, y_data, curr_attr):
    """finds the attribute with the lowest entropy to split on 
    
    Parameters:
        X_data(list of list of object): the X_data (of a partition) to find the attribute with the lowest entropy
        y_data (list of object): the class labels (of a partition)
        curr_attr (list of int): the indices of the remaining atributes being considered for partitioning 

    Returns:
        all_entropy.index(min(all_entropy)) (int): the index of the attribute with the smallest entropy
        
    """
    all_entropy = []
    
    for col in range(len(X_data[0])): # calculates the entropies of all the attributes
        curr_col = []
        for row in X_data:
            curr_col.append(row[col])
        all_entropy.append(calculate_entropy(curr_col, y_data))
    
    for index in range(len(all_entropy)):
        if index not in curr_attr: 
            all_entropy[index] = 10 # sets the entropy of an already picked attribute to 10, so it does not get picked 
    
    return all_entropy.index(min(all_entropy))


def calculate_entropy(x_attribute, y_data):
    """calculates the entropy of an attribute
    
    Parameters:
        x_attribute(list of objects): the values of a column/partition of an attribute
        y_data(list of objects): the values of the class label for the attribute/partition, used to calculate the entropy

    Returns:
        e_new (float): the entropy of a specific column/partition
        
    """
    freq = get_frequency(y_data) # finds the number of times an attribute appears in dataset
    
    e_new = 0
    x_freq = get_frequency(x_attribute)
    for value in x_freq.keys():
        e_curr = 0
        for key in freq.keys():
            count = 0
            for index, val in enumerate(x_attribute):
                if (val == value and y_data[index] == key):
                    count += 1 # finds the numerator when calculating entropy

            if count == 0:
                e_curr += 0 # in case there are no instances, e_curr needs to be set to 0 or else there is an error with log2
            else:
                e_curr += ((-count/x_freq[value]) * (np.log2(count/x_freq[value])))
            
        e_new += (x_freq[value]/len(y_data)) * e_curr # multiplies entropy of each value of an attribute to a value across the dataset
    
    return float(e_new)


def partition_data(X_data, y_data, att_to_split):
    """splits data based on attribute values
    
    Parameters:
        X_data (list of list of objects): the X-values of the data to split up based on an attribute value
        y_data (list of objects): the y-values/class labels of the data to split up, to parallel the split of X_data
        att_to_split (int): the index of the attribute with the lowest entropy to partition the data

    Returns:
        subset (dict contining lists of objects): the partitioned X and y data, with the 0 index of the list in the
            dictionary being the X values and the 1 index being the y values
            shape: {attribute_val1: [[x, x, x], [y]]}
        
    """
    subset = {}

    for index, row in enumerate(X_data):
        value = row[att_to_split] # splits data up based on the attribute with the lowest entropy
        if row[att_to_split] not in subset:
            subset[row[att_to_split]] = ([],[]) # adds a blank when there is no instance corresponding to an attribute value 
        subset[value][0].append(row)
        subset[value][1].append(y_data[index])
    
    subset_items = sorted(subset.items()) # sorts alphabetically
    subset = dict(subset_items)

    return subset


def most_freq_class(y_data):
    """returns most frequent class label
    
    Parameters:
        y_data (list of obj): all class labels for a dataset/partition

    Returns:
        (str): the most frequent (or if there is a tie, the first alphabetically) class in the dataset
        
    """

    freq = get_frequency(y_data)
    indices = []
    
    for key in freq:
        if freq[key] == max(freq.values()):
            indices.append(key)

    indices.sort() # if there is a tie in most frequent class label, takes the first alphabetically

    return indices[0]


# ================== GENERAL SECTION ===========================

def cross_val_predict(X, y, k, classifier_class, stratify=None):
    '''
        Purpose: - compute the k-fold cross-validaton for k = 10 and evaluate model performance for each split.
                 - aka, partition the data into 10 equal folds, and use 1 to be the test set for each iteration (NO repeated test sets).

        Arguments: 
            X (list of lists of obj's): - the list of samples
                                        - has shape: (n_samples, n_features)
            y (list of obj): - The target y values (labels corresponding to X)
                             - Default is None (in this case, the calling code only wants to sample X)
            k (int): the number of folds. aka, the number of times we'll generate train and test splits.
            classifier_class (class obj): - the classifier class we'll use to fit the model and predict.

        Returns:
            avg_acc (int): the avg accuracy of the fitted model over all k splits.
            avg_err_rate (int): the avg error rate of the fitted model over all k splits. 
            y_trues (list of strings): a list of all true mpg values in the dataset. 
            y_preds (list of strings): a list of all predicted mpg values.  
    '''
    # from mysklearn.myevaluation import kfold_split, stratified_kfold_split, accuracy_score, multiclass_precision_score, multiclass_recall_score, multiclass_f1_score
    
    # get all unique class labels. 
    labels = list(set(y)) 

    # initialize lists to store the acc, error rate, precision, recall, and f1-score of the model on each of the 10 splits of the data. 
    accuracies = []
    err_rates = []
    precisions = []
    recalls = []
    f1s = []

    # initalize lists to store all the predicted class labels and all the true class labels (for the confusion matrix)
    y_trues = []
    y_preds = []

    # split the dataset into cross-validation folds. 
    # note: kfold_split returns a list of tuples where each tuple has the train and test indicies for a given fold.
    if stratify == False:
        folds = myeval.kfold_split(X, n_splits=k, shuffle=True)
    if stratify == True:
        folds = myeval.stratified_kfold_split(X, y, n_splits=k, shuffle=True)

    # iterate over each train/test split so we can evaluate model performance on the different subsets of data.
    for train_indices, test_indices in folds:
        # convert X and y to numpy arrays ONLY for slicing
        X_array = np.array(X, dtype=object)
        y_array = np.array(y, dtype=object)

        # create train and test sets for the current fold.
        X_train, y_train = X_array[train_indices].tolist(), y_array[train_indices].tolist()
        X_test, y_test = X_array[test_indices].tolist(), y_array[test_indices].tolist()

        # create a classifier object (because we want a fresh classifier for each new split of data).
        classifier = classifier_class()

        # train the classifier on the training data (samples and corresponding labels). 
        classifier.fit(X_train, y_train)

        # predict MPG for the test instances.
        y_pred = classifier.predict(X_test)

        pred_ratings = y_pred
        actual_ratings = y_test

        # compute the accuracy and error rate of the model by comparing true and predicted mpg.
        acc = myeval.accuracy_score(actual_ratings, pred_ratings)
        err = 1 - acc
        precision = myeval.multiclass_precision_score(actual_ratings, pred_ratings, labels=labels)
        recall = myeval.multiclass_recall_score(actual_ratings, pred_ratings, labels=labels)
        f1 = myeval.multiclass_f1_score(actual_ratings, pred_ratings, labels=labels)

        accuracies.append(acc)
        err_rates.append(err)
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)

        y_trues.extend(actual_ratings)
        y_preds.extend(pred_ratings)

    # find the avg accuracy and avg error rate of the model across all 10 splits of data.  
    avg_acc = sum(accuracies) / k
    avg_err_rate = sum(err_rates) / k
    avg_precision = sum(precisions) / k
    avg_recall = sum(recalls) / k
    avg_f1 = sum(f1s) / k

    # convert y_trues and y_pred as strings
    y_trues = [str(y) for y in y_trues]
    y_preds = [str(y) for y in y_preds]

    return avg_acc, avg_err_rate, avg_precision, avg_recall, avg_f1, y_trues, y_preds


def my_discretizer(value):
    """returns discrete values for already normalized values
    
    Parameters:
        value (float): a normalized value (between 0 and 1) 

    Returns:
        i (int): a categorical discrete value, ranging between 1 - 10
    
    Notes:
        Discrete values based on every tenth of a normalized value
            For example: 0 - 0.1 ==> 1
                         0.1 (excluded) - 0.2 ==> 2
                         Etc.
        
    """
    for i in list(range(1, 11)):
        if float(round(value, 1)) <= (i/10):
            return i


def get_col(X_data, col_num):
    """returns a column of a dataset
    
    Parameters:
        X_data (a list of a list of objects): the list of training samples
        col_num (integer): the column index of the column being returned

    Returns:
        return_col (list of objects): all values in a specific column
        
    """
    
    return_col = []
    for row in X_data:
        return_col.append(row[col_num])
    
    return return_col


def list_deep_copy(old_list):
    """returns a deep copy of a 2D list
    
    Parameters:
        old_list (list of list of obj): a 2D list that needs to be copied
            - This function is mainly used for the confusion matrix

    Returns:
        new_list (list of list of obj): A deep copied list of the old_list, with
            the exact same values/length of lists
        
    """
    new_list = []

    for row in old_list:
        curr_row = []
        for value in row:
            curr_row.append(value)
        
        new_list.append(curr_row)
    
    return new_list
