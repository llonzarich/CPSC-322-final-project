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
    