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


