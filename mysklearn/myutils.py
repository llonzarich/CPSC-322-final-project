import numpy as np
import math

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
    from mysklearn.myevaluation import kfold_split, stratified_kfold_split, accuracy_score, multiclass_precision_score, multiclass_recall_score, multiclass_f1_score
    
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
        folds = kfold_split(X, n_splits=k, shuffle=True)
    if stratify == True:
        folds = stratified_kfold_split(X, y, n_splits=k, shuffle=True)

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
        acc = accuracy_score(actual_ratings, pred_ratings)
        err = 1 - acc
        precision = multiclass_precision_score(actual_ratings, pred_ratings, labels=labels)
        recall = multiclass_recall_score(actual_ratings, pred_ratings, labels=labels)
        f1 = multiclass_f1_score(actual_ratings, pred_ratings, labels=labels)

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


def rf_cross_val_predict(x_data, n_splits = 5, state = None, shuffle = False):
    """ calls the kfold_split function with a certain number of splits, meant for a Random Forest classifier
    
    Parameters:
        x_data (list of list of obj): The list of samples
            The shape of X is (n_samples, n_features)
        n_splits (int): number of subsets the x_data is partitioned into
        state (int): integer used for seeding a random number generator for reproducible results
        shuffle (bool): whether or not to randomize the order of the instances before splitting

    Returns:
        List of 2 item tuples: list of folds where each fold is defined as a 2-item tuple
            The first item in the tuple is the list of training set indices for the fold
            The second item in the tuple is the list of testing set indices for the fold
        
    """

    # since kfold_split is function that already iterates depending on the number of splits, no need to iterate in this function
    return kfold_split(x_data, n_splits, state, shuffle) 


def kfold_split(X, n_splits=5, random_state=None, shuffle=False):
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




# =============== DECISION TREE SECTION =====================

def select_attribute_dt(instances, attributes, header, attribute_domains):
    '''
        Purpose: - implement the general E_new algorithm to select an attribute to split on.
                 - whichever attribute has the lowest E_new val is the chosen attribute for splitting.

        Args:
            instances (list of lists): - the instances in X_train that need to be assigned to a partition still
                                       - (the class label is at the end of each inner list).
                                       - example: [ 
                                                    ["Senior", "Java", "no", "no", "False],
                                                    ["Junior", "Java", "yes", "no", "True"] 
                                                   ]
            attributes (list): - attribute names that are still availible to split on 
                               - example: ["att0", "att1"]
            header (list): a fixed list of headers
            attribute_domains (dict): a dictionary that stores the unique values that each attribute can take on.

        Returns:
            selected_att (obj): - the name of the attribute to split on.
                                - example: "att0"

    '''
    best_attribute = []
    best_entropy = float("inf")
    zeros_E_new = 0 # to count how many attributes have entropy = 0

    num_instances = len(instances)

    # iterate over each attribute (aka, header) (that hasn't been used as a partition yet)
    # (we iterate over each attribute to compute its E_new because the attribute we split on depends on which attribute has the lowest E_new value).
    for att in attributes:
        # print("CURR ATTRIBUTE: ", att)
        att_index = header.index(att) # find the index of the col corresponding to the given attribute to split on.

        # att_domain = attribute_domains[att] # find all possible values that the given attribute can take on.
    
        # partition the instances based on their attribute value.
        partitions = partition_instances_dt(instances, att, header, attribute_domains)
        # print("PARTITIONS: ", partitions)

        E_partitions = [] # intialize an empty list to store each partition's weighted entropy (= each partition's entropy MULTIPLIED by the total partition probability).

        # iterate over each partition (aka, each attribute value in the given attribute). 
        # btw, each partition is made up of instances (lists)
        # att_value = the dictionary "key", partition = the dictionary "value" (a list of instances)
        for partition in partitions.values():
            num_instances_in_partition = len(partition) # find the number of instances in the current partition.

            # if the partition is empty, the entropy of the current partition = 0
            if num_instances_in_partition == 0:
                continue

            # compute the probability of his partition.
            weight = num_instances_in_partition / num_instances

            # find all unique labels of the instances in the current partition (i.e., "yes" vs "no", "True" vs. "False").
            labels = [row[-1] for row in partition] 
            unique_labels = set(labels)

            # iterate over each unique label (i.e., "yes" and "no")
            entropy = 0
            for label in unique_labels:
                # count the number of times the current label appears in the list of all labels.
                num_curr_label = labels.count(label)
                
                # compute class label probabilities for the current partition.
                prob = num_curr_label / num_instances_in_partition 

                # compute entropy for the current partition ONLY if the current label has at least 1 instance in it.
                if prob > 0:
                    entropy += ( (-prob) * math.log2(prob) )
            
            # compute the weighted entropy for the current partition.
            entropy *= weight

            E_partitions.append(entropy) # append the entropy of the current partition to the list.

        # after we've found the entropy of each partition for the current attribute, we need to find E_new (the TOTAL weighted entropy for the current attribute).
        E_new = sum(E_partitions) 
                
        # if the E_new for the attribute is better (aka, lower) than the current best_entropy, replace it.
        if E_new < best_entropy:
            best_entropy = E_new
            best_attribute = [att]
            # print("BEST ATTRIBUTE: ", best_attribute[0])
        # if the E_new for this attribute is equal to the current best_entropy, append it to the list.
        elif E_new == best_entropy:
            best_attribute.append(att)
            # print("BEST ATTRIBUTE LIST: ", best_attribute)

    # sort the attributes in alphabetical order.
    best_attribute = sorted(best_attribute)

    # extract the best attribute. (because if there are entropy ties, we want to choose the one that comes first in the alphabet).
    best_attribute = best_attribute[0]

    # print("best attribute to split on: ", best_attribute)

    return best_attribute


def partition_instances_dt(instances, attribute, header, attribute_domains):
    '''
        Purpose: partition the dataset into separate groups based on the values the attribute can take on. 

        Args:
            instances (list of lists): - the instances in X_train that need to be assigned to a partition still
                                       - (the class label is at the end of each inner list).
                                       - example: [ 
                                                    ["Senior", "Java", "no", "no", "False], 
                                                    ["Junior", "Java", "yes", "no", "True"]
                                                   ]
            attribute (str): the attribute (aka, col) that we are creating a partition on.
            header (list): a fixed list of headers
            attribute_domains (dict): a dictionary that stores the unique values that each attribute can take on.

        Returns: 
            partitions (dict): a dictionary mapping each attribute value to the list of instances that have that value: {attribute_value: [instances]}
                               - example: { 
                                           "Senior": [ [instance], [instance], ... ],
                                           "Mid": [ [instance], [instance], ... ],
                                           "Junior": [ [instance], [instance], ... ]
                                           }

        Notes: - this is group by attribute domain (NOT group by values of attributes in instances).
               - example: if the chosen attribute is "att0" which can take on values 1,2,and 3, then this function will create 3 separate groups - each separate group will have all instances of an attribute value
    '''
    att_index = header.index(attribute) # find index of the col corresponding to the given attribute to split on.

    att_domain = attribute_domains[attribute] # find all possible values that the given attribute can take on.
    
    partitions = {} # initialize an empty dict to assign instances to each separate partition.
    
    # iterate over each possible value that the given attribute can take on (i.e., "Junior", "Mid", "Senior" for att = "level")
    for att_value in att_domain:
        partitions[att_value] = [] # create an empty nested dict for each attribute value with dict key = the current attribute value.

        # iterate over each instance in the given list of instances to assign it to the key if applicable.
        for instance in instances:
            # if the current instance (aka row) has the current attribute value in its 'att_index' column ==> append instance to the list for the current attribute value.
            if instance[att_index] == att_value:
                partitions[att_value].append(instance)

    return partitions


def all_same_class_dt(instances):
    '''
        Purpose: check if instances in a given subset belong to the same class label.

        Args: 
            instances (list of lists): - the instances in X_train that need to be assigned to a partition still
                                       - (the class label is at the end of each inner list).
                                       - example: [ 
                                                    ["Senior", "Java", "no", "no", "False], 
                                                    ["Junior", "Java", "yes", "no", "True"]
                                                   ]

        Returns:
            - True if all instances in the given subset belong to the same class label.
            - False otherwise.
    '''
    # get the class label of the first instance.
    first_class = instances[0][-1]
    
    # iterate over each instance in the given list of instances.
    for instance in instances:
        # if any label differs, return False immediately.
        if instance[-1] != first_class:
            return False
        
    # return True if the loop completes without finding differences.
    return True 



def tdidt_dt(current_instances, available_attributes, header, attribute_domains, parent_instances=None):
    '''
        Purpose: recursively building a decision tree using the TDIDT algorithm

        Args:
            current_instances: the instances we have to assign to a split.
            available_attributes: attributes that have not been chosen to be split on yet.
            header (list): a fixed list of headers
            attribute_domains (dict): a dictionary that stores the unique values that each attribute can take on.
            parent_instances: the instances in the parent node of the node/leaf we're trying to make.
            
        Returns:
            tree (list of lists): the nested structure that contains all decision rules and attribute splits.

        Steps
            1. Select the best attribute to split on and create an "Attribute" node - you can do this randomly or with entropy
            2. For each value of the selected attribute:
                a. Create a "Value" subtree.
                b. If all instances in this partition have the same class: Append a "Leaf" node
                c. If there are no more attributes to select: Append a "Leaf" node (handle clash w/majority vote leaf node)
                d. If the partition is empty: Append a "Leaf" node (backtrack and replace attribute node with majority vote leaf node)
                e. Otherwise: Recursively build another "Attribute" subtree for this partition and append it to the "Value" subtree.
            3. Append each "Value" subtree to the current "Attribute" node.
            4. Return the current tree (nested list structure).
    '''
    if parent_instances is None:
        parent_instances = current_instances

    # display the availible attributes that we can split on (aka, the attributes that haven't been used as a partition yet).
    # print("available attributes:", available_attributes)
    
    # select an attribute to split on.
    split_attribute = select_attribute_dt(current_instances, available_attributes, header, attribute_domains)
    # print("splitting on:", split_attribute)
    
    # remove the attribute we just split on from the list of availible attributes (because we can't split on this attribute again).
    available_attributes.remove(split_attribute)

    # add the attribute that we're going to split on to the tree as the start of a new nested list.
    tree = ["Attribute", split_attribute]

    # group data by attribute domains (creates pairwise disjoint partitions)
    partitions = partition_instances_dt(current_instances, split_attribute, header, attribute_domains)
    # print("partitions:", partitions)
    # ==> 'partitions' has every current instance (aka row from the dataset) separated into different "bins" according to its attribute value.
    
    # iterate over each partition (aka, attribute value) in 'partitions' unless one of the following occurs (base case)
    # note: the partitions are the different attribute values for a given attribute (i.e., 1, 2, and 3 for attribute = "job_status")
    # note: each partition is made up of instances (aka, rows from the dataset).
    # note: process partitions in alphabetical order. 
    for att_value in sorted(partitions.keys()):
        att_partition = partitions[att_value] # get the list of instances (aka rows) in the current partition.
        value_subtree = ["Value", att_value]  # start a subtree for the current attribute, branching off of the main node.


        # CASE 1: all class labels of the partition are the same
        # => make a leaf node
        if len(att_partition) > 0 and all_same_class_dt(att_partition):
            # print("CASE 1")

            # find the CLASS LABEL that all instances belong to by looking at the first partition, last element (because the last element in all instance lists is the class label).
            curr_instances_class = att_partition[0][-1]

            # find NUMBER OF INSTANCES of instances the current partition
            num_in_partition = len(att_partition)
            
            # find the NUMBER OF INSTANCES in the parent node of the leaf node we're making (aka, the number of instances in the current node).
            num_in_parent = len(current_instances)

            # make a leaf node.
            value_subtree.append( ["Leaf", curr_instances_class, num_in_partition, num_in_parent] )
            # print("LEAF NODE: majority label =", curr_instances_class, ", probability=", num_in_partition, "/", num_in_parent)

            # append the completed subtree to the tree.
            tree.append(value_subtree)

            continue


        # CASE 3: no more instances to partition (empty partition)
        # => backtrack and replace the ENTIRE attribute node with a majority vote leaf.
        elif len(att_partition) == 0:
            # print("CASE 3")

            # find all unique labels among all the parent instances. (i.e., unique labels in )
            labels = [row[-1] for row in current_instances]
            unique_labels = set(labels)

            # find the majority CLASS LABEL of the parent's node by comparing the number of instances per class.
            majority_labels = []
            majority_count = 0
            for label in unique_labels:
                # count the number of times the current label appears in the list of all labels.
                num_curr_label = labels.count(label)

                if num_curr_label > majority_count:
                    majority_labels = [label]
                    majority_count = num_curr_label
                elif num_curr_label == majority_count:
                    majority_labels.append(label)
                
            # sort labels alphabetically (because if there are ties, we want to choose the label based on which one comes first alphabetically).
            majority_labels = sorted(majority_labels)
            majority_label = majority_labels[0]

            # find the NUMBER OF INSTANCES in the node ABOVE this new leaf's node partition
            num_in_curr = len(current_instances)

            # find the NUMBER OF INSTANCES in the node above this new leaf node.
            num_in_parent = len(parent_instances)

            return ["Leaf", majority_label, num_in_curr, num_in_parent]
        

        # CASE 2: no more attributes to select (clash)
        # => handle clash w/ majority vote leaf node
        elif len(att_partition) > 0 and len(available_attributes) == 0:
            # print("CASE 2")
            
            # find all unique labels of the instances in the current partition.
            labels = [row[-1] for row in att_partition] 
            unique_labels = set(labels)

            # determine the majority CLASS LABEL of the current partition's instances by comparing the number of instances per class.
            majority_labels = []
            majority_count = 0
            for label in unique_labels:
                # count the number of times the current label appears in the list of all labels.
                num_curr_label = labels.count(label)
                
                if num_curr_label > majority_count:
                    majority_labels = [label]
                    majority_count = num_curr_label
                elif num_curr_label == majority_count:
                    majority_labels.append(label)
                
            # sort labels alphabetically (because if there are ties, we want to choose the label based on which one comes first alphabetically).
            majority_labels = sorted(majority_labels)
            majority_label = majority_labels[0]
            
            # find the NUMBER OF INSTANCES in the current partition. (numerator)
            num_in_partition = len(att_partition)

            # find the NUMBER OF INSTANCES in the parent node of the leaf. (denominator)
            num_in_parent = len(current_instances)

            # make a leaf node.
            # value_subtree.append( ["Leaf", majority_label, num_in_majority_class, num_in_partition] )
            value_subtree.append(["Leaf", majority_label, num_in_partition, num_in_parent])
            # print("LEAF NODE: majority label =", majority_label, ", probability=", num_in_partition, "/", num_in_parent)

            # append the completed subtree to the tree.
            tree.append(value_subtree)

            continue

        else: # if none of the base cases apply, we recurse!!
            # print("ATTRIBUTE VALUE (NEST FOR TREE DICTIONARY): ", value_subtree)

            subtree = tdidt_dt(att_partition, available_attributes.copy(), header, attribute_domains, current_instances)
            
            value_subtree.append(subtree)
            tree.append(value_subtree)
            
    return tree



def tdidt_predict_dt(tree, instance, header):
    '''
        Purpose: travel the tree to generate predictions for ONE test instance.

        Args: 
            tree (list of lists): nested list that we traverse in order to generate our prediction.
            instance (list): an instance we want to classify using the decision tree classifer.
            header (list or str): a list of header names (as strings).

        Returns: the prediction for 1 unseen test instance.
    '''
    data_type = tree[0]

    # Base case: if this is a leaf, just return its class label. stored as [leaf, class]. 
    if data_type == "Leaf":
        label = tree[1]
        return label
    
    # Recursive case:if we are here, this is an Attribute node
    attribute_name = tree[1]
    attribute_index = header.index(attribute_name)
    instance_value = instance[attribute_index]

    # Look for the matching value node
    for values in tree[2:]:
        value = values[1]
        subtree = values[2]
        
        if instance_value == value:
            return tdidt_predict_dt(subtree, instance, header)
    
    return None


    