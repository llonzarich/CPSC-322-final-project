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


def select_attribute(instances, attributes, header, attribute_domains, F=None):
    '''
        Purpose: - implement the genereal E_new algorithm to select an attribute to split on.
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
            F (int): - the number of randomly selected attributes (from the remaining attributes) to be used as candidates for the split.
                     - if None, let F be the number of attributes (the remaining attributes to be split on).

        Returns:
            selected_att (obj): - the name of the attribute to split on.
                                - example: "att0"

    '''
    # set the number of attribute to be used as candidates for the next split.
    if F is not None and F < len(attributes):
        num_attribute_candidates = F 
    else:
        F = len(attributes)
        num_attribute_candidates = F

    # define the attribute candidates.
    attribute_candidates = np.random.choice(attributes, size=F, replace=False)

    best_attribute = []
    best_entropy = float("inf")

    num_instances = len(instances)

    # iterate over each attribute (aka, header) (that hasn't been used as a partition yet)
    # (we iterate over each attribute to compute its E_new because the attribute we split on depends on which attribute has the lowest E_new value).
    for att in attribute_candidates:
        # partition the instances based on their attribute value.
        partitions = partition_instances(instances, att, header, attribute_domains)

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

    # sort the attributes in alphabetical order.
    best_attribute = sorted(best_attribute)

    # extract the best attribute. (because if there are entropy ties, we want to choose the one that comes first in the alphabet).
    best_attribute = best_attribute[0]

    # print("best attribute to split on: ", best_attribute)

    return best_attribute


def partition_instances(instances, attribute, header, attribute_domains):
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


def all_same_class(instances):
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



def tdidt(current_instances, available_attributes, header, attribute_domains, parent_instances=None):
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

    # set the number of attributes to consider as candidates for selecting an attribute to split on.
    F = math.floor(math.sqrt(len(available_attributes)))
    
    # select an attribute to split on.
    split_attribute = select_attribute(current_instances, available_attributes, header, attribute_domains, F)
    # print("splitting on:", split_attribute)
    
    # remove the attribute we just split on from the list of availible attributes (because we can't split on this attribute again).
    available_attributes.remove(split_attribute)

    # add the attribute that we're going to split on to the tree as the start of a new nested list.
    tree = ["Attribute", split_attribute]

    # group data by attribute domains (creates pairwise disjoint partitions)
    partitions = partition_instances(current_instances, split_attribute, header, attribute_domains)
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
        if len(att_partition) > 0 and all_same_class(att_partition):
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

            subtree = tdidt(att_partition, available_attributes.copy(), header, attribute_domains, current_instances)
            
            value_subtree.append(subtree)
            tree.append(value_subtree)
            
    return tree



def tdidt_predict(tree, instance, header):
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
            return tdidt_predict(subtree, instance, header)
    
    return None