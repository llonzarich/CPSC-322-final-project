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


def my_discretizer(value):
    for i in list(range(1, 11)):
        if float(round(value, 1)) <= (i/10):
            return i









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


def rf_kfold_split(X, n_splits=5, random_state=None, shuffle=False):
    """Split dataset into cross validation folds.

    Args:
        X(list of list of obj): The list of samples
            The shape of X is (n_samples, n_features)
        n_splits(int): Number of folds.
        random_state(int): integer used for seeding a random number generator for reproducible results
        shuffle(bool): whether or not to randomize the order of the instances before creating folds

    Returns:
        folds(list of 2-item tuples): The list of folds where each fold is defined as a 2-item tuple
            The first item in the tuple is the list of training set indices for the fold
            The second item in the tuple is the list of testing set indices for the fold

    Notes:
        The first n_samples % n_splits folds have size n_samples // n_splits + 1,
            other folds have size n_samples // n_splits, where n_samples is the number of samples
            (e.g. 11 samples and 4 splits, the sizes of the 4 folds are 3, 3, 3, 2 samples)
        Loosely based on sklearn's KFold split():
            https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html
    """
    np.random.seed(random_state)

    # to divide up the folds, to see which indexes are used for training or testing
    x_length = len(X)
    x_unshuffled_indices = list(range(x_length))
    
    X_copy = X[:]
    X_train = []
    X_indices = []

    for i in range(len(X_copy)): # saves each row into the training set
        if shuffle: # if shuffle is True, a random row from the list of samples is saved repeatedly until there are no more rows left
            rand_int = np.random.randint(0, len(X_copy))
            X_train.append(X_copy.pop(rand_int)) # need to pop so the same row isn't randomly chosen twice (why a copy of X was created)
            X_indices.append(x_unshuffled_indices.pop(rand_int))
        else:
            X_train.append(X_copy[i])
            X_indices.append(x_unshuffled_indices[i])

    # divides the indexes into folds, in case the folds do not partition dataset evenly
    # ensures the last few folds have a lower number of indices, if dataset not divided evenly
    fold_sizes = [x_length // n_splits + (1 if i < x_length % n_splits else 0)
                  for i in range(n_splits)]

    # saves indexes for the folds, divided into tuples depending on training and testing
    folds = []
    curr = 0
    for size in fold_sizes:
        start, stop = curr, curr + size # saves the start and stop indices for the X_indices for testing
        test_index = X_indices[start:stop]
        train_index = X_indices[0:start] + X_indices[stop:] # saves the data before and after the testing data, used as training data
        folds.append((train_index, test_index))
        curr = stop # moves up the dataset, so only the start and stop indicies for testing data is looked at

    return folds

def fold_values(folds, table, attributes, class_name):
    column_indices = []
    for col in attributes:
        column_indices.append(table.column_names.index(col))

    X_train_indices = []
    X_test_indices = []
    for items in folds:
        [X_train_indices.append(items[0])] # X values needs to be a 2d list
        [X_test_indices.append(items[1])]
    
    X_train = []
    X_test = []
    y_train = []
    y_test = []

    for row in X_train_indices:
        temp_x_training_row = [] # makes sure to divide each row
        temp_y_training_row = []
        for index in row:
            # finds the y values for each x (as this approach doesn't return y-train)
            temp_y_training_row.append(table.get_column(class_name)[index])
            temp_row = [] # ensures each row is a 2d list
            for col_index in column_indices:
                # saves the values of the specific columns for each row, not just the row itself
                temp_row.append(table.data[index][col_index])
            temp_x_training_row.append(temp_row)
        y_train.append(temp_y_training_row)
        X_train.append(temp_x_training_row)

    # finds the actual values for each testing indices set
    for row in X_test_indices:
        temp_x_testing_row = [] # makes sure to divide each row
        temp_y_testing_row = []
        for index in row:
            # finds the y values for each x (as this approach doesn't return y-test/actual y values)
            temp_y_testing_row.append(table.get_column(class_name)[index])
            temp_row = [] # ensures each row is a 2d list
            for col_index in column_indices:
                # saves the values of the specific columns for each row, not just the row itself
                temp_row.append(table.data[index][col_index])
            temp_x_testing_row.append(temp_row) 
        y_test.append(temp_y_testing_row)
        X_test.append(temp_x_testing_row)

    return X_train, X_test, y_train, y_test
