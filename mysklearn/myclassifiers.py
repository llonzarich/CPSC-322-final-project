import numpy as np
from mysklearn import myutils
from mysklearn import myevaluation
from graphviz import Graph


class MyDecisionTreeClassifier:
    """Represents a decision tree classifier.

    Attributes:
        X_train(list of list of obj): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
        y_train(list of obj): The target y values (parallel to X_train).
            The shape of y_train is n_samples
        tree(nested list): The extracted tree model.
        F (int): The number of randomly selected attributes to consider partitioning data on
            - Only set to a value if this class is used for the Random Forest Classifier 

    Notes:
        Loosely based on sklearn's DecisionTreeClassifier:
            https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
        Terminology: instance = sample = row and attribute = feature = column
    """
    def __init__(self, F = None):
        """Initializer for MyDecisionTreeClassifier.

            Args: 
                F (int): - the number of randomly chosen attributes to be used as candidates for the split
                         - used for when the Random Forest Classifier class creates a Decision Tree object
                         -- default is None (for when the Decision Tree is not used for a Random Forest)
        """
        self.X_train = None
        self.y_train = None
        self.tree = None
        self.F = F

    def fit(self, X_train, y_train, random_state = None):
        """Fits a decision tree classifier to X_train and y_train using the TDIDT
        (top down induction of decision tree) algorithm.

        Args:
            X_train(list of list of obj): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples

        Notes:
            Since TDIDT is an eager learning algorithm, this method builds a decision tree model
                from the training data.
            Build a decision tree using the nested list representation described in class.
            On a majority vote tie, choose first attribute value based on attribute domain ordering.
            Store the tree in the tree attribute.
            Use attribute indexes to construct default attribute names (e.g. "att0", "att1", ...).

            Recursively calls build_tree function
        """

        self.X_train = X_train
        self.y_train = y_train
        self.tree = []

        if random_state is not None:
            np.random.seed(random_state)

        def build_tree(X_train, y_train, current_att, prev_length):
            """Recursively builds the decision tree using X_train and y_train using the TDIDT

            Args:
                X_train(list of list of obj): The list of training instances (samples).
                    The shape of X_train is (n_train_samples, n_features)
                    X_train gets split up recursively
                y_train(list of obj): The target y values (parallel to X_train)
                    The shape of y_train is n_train_samples
                    y_train gets split up recursively
                current_att(list of int): The size of the current X_train:
                    Used to check if the partition is empty
                prev_length(int): the number of instances in the previous attribute
                    to calculate the fraction

            Notes:
                This is a recursive function
            """
            # CASE 1: all class labels of the partition are the same
            # ==> make a leaf node
            if myutils.all_same_values(y_train):
                return ["Leaf", y_train[0], len(y_train), prev_length]
            
            # CASE 2: there are no more attributes to split on, and still don't have the same class labels
            # ==> take the most frequent class label, and if there is a tie: take the class label that comes first alphabetically
            if len(current_att) == 0: # returns leaf of most frequent class label/first alphabetically
                return ["Leaf", myutils.most_freq_class(y_train), len(y_train), prev_length]

            # for the random random forest classifier: selects F random attributes as partition candidate attributes 
            if self.F is not None:
                temp_attr = current_att[:]
                np.random.shuffle(temp_attr)
                temp_attr = temp_attr[:self.F]
            else:
                temp_attr = current_att[:]

            # finds index of attribute with lowest entropy
            att_to_split_index = myutils.attribute_to_split(X_train, y_train, temp_attr) 
            
            tree = ["Attribute", "att" + str(att_to_split_index)]

            all_attr_dict = myutils.get_frequency(myutils.get_col(self.X_train, att_to_split_index)) 
            all_attr = sorted(all_attr_dict) # sorts attributes alphabetically, so first alphabetical attribute shows up first
            
            subsets = myutils.partition_data(X_train, y_train, att_to_split_index) # splits data based on attribute with lowest entropy
            
            # CASE 3: not all attribute values appear in the training set
            # ==> backtrack and create leaf node (instead of splitting further) with most frequent class label
            curr_row = []
            for key in subsets:
                curr_row.extend(subsets[key][0]) # gets all X attribute values from current row
            
            all_col = myutils.get_col(curr_row, att_to_split_index)
            for attr in all_attr: 
                if attr not in all_col:
                    subsets[attr] = ([],[]) # used to check if all values for an attribute appears in partition
            
            for subset in subsets: # if not all values appear for partition, a leaf is returned of the most frequent value
                if len(subsets[subset][0]) == 0:
                    return ["Leaf", myutils.most_freq_class(y_train), len(y_train), prev_length]
            
            prev_length = len(y_train) # used for denominator for leaf node

            # CASE 4: still have attributes to partition on, all attribute values exist in training set, training set does not have same class label
            # ==> Recurse!
            for subset in subsets:
                X_temp = subsets[subset][0] # all x data of partition
                y_temp = subsets[subset][1] # all y data of partition

                attribute_copied = current_att.copy()
                attribute_copied.remove(att_to_split_index) # removes attribute, so it is not considered when calculating entropy of further subtree
                sub_tree = build_tree(X_temp, y_temp, attribute_copied, prev_length)
            
                tree.append(["Value", subset, sub_tree]) # appends attribute to a value

            return tree

        self.tree = build_tree(X_train, y_train, list(range(len(X_train[0]))), len(y_train))


    def predict(self, X_test):
        """Makes predictions for test instances in X_test.

        Args:
            X_test(list of list of obj): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        
        y_pred = []

        for test in X_test:
            curr_tree = self.tree
            
            # iterate / continue down tree until each test instance is assigned a class label, or until we reach a leaf node for each test instance. 
            while True: 
                # make sure we STOP once we reach a leaf node. 
                if curr_tree[0] == "Leaf":
                    y_pred.append(curr_tree[1]) # assign class label to the current test instance.
                    break

                # if the current node is an attribute node (not a leaf), let's continue down the tree.   
                if curr_tree[0] == "Attribute":
                    attr_num = int(curr_tree[1][3]) # saves attribute index to check if value in X_test is same as value in attribute
                    curr_subtree = curr_tree[2:] # get subtree of the current tree.
                    found = False
                    for sub in curr_subtree: # finds the attribute value that equals the test's value for the same attribute
                        if sub[1] == test[attr_num]:
                            curr_tree = sub[2]
                            found = True
                            break
                    if not found:
                        maj_value = []
                        for sub in curr_subtree:
                            maj_value.append(sub[2][1])
                        y_pred.append(myutils.most_freq_class(maj_value))
                        break

        return y_pred 
    

    def print_decision_rules(self, attribute_names=None, class_name="class"):
        """
            Purpose: Prints the decision rules from the tree in the format "IF att == val AND ... THEN class = label", one rule on each line.

            Args:
                attribute_names (list of str or None): - A list of attribute names to use in the decision rules
                                                    - if None (the list was not provided), use the default attribute names based on indexes (e.g. "att0", "att1", ...) should be used).
                class_name(str): - A string to use for the class name in the decision rules
                                ("class" if a string is not provided and the default name "class" should be used).
            
            Note: - Leaf subtree lists are stored as [type_of_node (e.g., "Attribute", "Value", "Leaf"), node_label (e.g., "True", "False"), numerator_probability, denominator_probability]
                  - Attribute subtree lists are stored as [type_of_node, attribute]
                  - I did reference ChatGPT to guide me through how I would approach this function because I was getting a bit lost in the logic due to the recursion element here.
        """
        # get the number of attributes in the dataset. 
        num_attributes = len(self.X_train[0])

        # handle attribute_names=None parameter.
        if attribute_names is None:
            attribute_names = [f"att{i}" for i in range(num_attributes)]
        
        # traverse tree until one of the following conditions have been met.
        def recurse(subtree, conditions):
            curr_node = subtree[0] # grab the current node type. (e.g., "Attribute", "Value", "Leaf"). 
            
            # base case 1: if leaf, print decision rule:
            if curr_node == "Leaf":
                label = subtree[1] # get the label for the current node. (e.g., "True", "False").
                rule_str = " AND ".join(conditions) # grab all conditions that have led us to this leaf node.
                print(f"IF {rule_str} THEN {class_name} = {label}") # print the rule.

            # base case 2: if attribute node, recurse/iterate down through each branch and "append" a rule as we go.
            elif curr_node == "Attribute":
                att_name = subtree[1] # get the attribute that the current node splits on. 

                for branch in subtree[2:]: # iterate over each branch off of the current node.
                    att_val = branch[1] # get the value of the attribute for the current branch
                    val_subtree = branch[2] # get the subtree that goes off of the current branch.
                    new_rule = conditions + [f"{att_name} == {att_val}"] # add this rule to the decision rule string.
                    recurse(val_subtree, new_rule) # recursively continue to go through the subtree.

        recurse(self.tree, [])
    
    def visualize_tree(self, attribute_names=None):
        """Visualizes a tree via the open source Graphviz graph visualization package

        Args:
            attribute_names(list of str or None): A list of attribute names to use in the decision rules
                (None if a list is not provided and the default attribute names based on indexes
                (e.g. "att0", "att1", ...) should be used).

        Returns:
            g (Graph): a graph that visualizes the Decision Tree
                
        """
        if attribute_names is None: # in case no attribute_names are given, names are put into att0, att1, ... format
            attribute_names = []
            for i in range(len(self.X_train[0])):
                attribute_names.append("att" + str(i))
        
        g = Graph()
        g.attr(rankdir='TB')  #top-to-bottom layout

        node_counter = 0 # node counter needed since an attribute can appear more than once in the tree -- need to differentiate Attribute nodes

        def find_nodes_recursive(curr_tree, last_att, value_label): # this function traverses each branch/leaf in tree, creating a node to be output into a pdf file
            nonlocal node_counter # needs to be nonlocal or else the leaf nodes will connect to attributes that are not meant for them (differentiates nodes)
            subtree_type = curr_tree[0] # this function is similar to the print_decision_rules function 

            node_id = f"node{node_counter}" # creates a unique name for each node
            node_counter += 1 

            if subtree_type == "Leaf":
                leaf_label = f"{curr_tree[1]} ({round(curr_tree[2]/curr_tree[3] * 100, 2)}%)"
                g.node(node_id, label = leaf_label)

                if last_att != "": # creates an edge between each leaf and attribute, with value on relationship line 
                    g.edge(last_att,node_id, label = str(value_label))
                return
            
            if subtree_type == "Attribute":
                prev_att = last_att # saves the last attribute name to check if it is the first attribute

                att_index = int(curr_tree[1][3]) # finds which attribute is being split upon
                last_att = str(attribute_names[att_index]) 

                g.node(node_id, label = last_att, shape = "box") # creates an attribute node
                
                if prev_att != "": 
                # if attribute is first attribute being split on, program 
                # should not create an edge (since there is nothing to conenct to yet)
                    g.edge(prev_att, node_id, label = str(value_label))

                for i in range(2, len(curr_tree)): # similar to print_decision_rules function, need to traverse down each branch
                    value_branch = curr_tree[i]
                    value_label = str(value_branch[1])
            
                    subtree = value_branch[2]

                    find_nodes_recursive(subtree, node_id, value_label)
        

        find_nodes_recursive(self.tree, "", "")

        # display graph 
        return g

class MyRandomForestClassifier:
    """
        Purpose: Represents a random forest classifier.

        Attributes:
            X_train (list of list of obj): - the list of training instances (samples).
                                           - has shape: (n_train_samples, n_features)
            y_train (list of obj): - the target y values (labels parallel to X_train).
                                   - has shape: y_train is n_samples
            N (int): - the number of trees to generate and train (we choose the most accurate M of these trees to use for our forest).
                     - corresponds to "n_estimators" in the sklearn random forest class implementation.
                     - default is 100.
            M (int): - the number of trees to be included in the forest. 
                     - M is a subset of N
                     - this parameter is necessary because we only want to take a certain number of the best trees to for our forest.
                     - if None, let M be N-1 (1 less than the number of trees because M < N)
            F (int): - the number of randomly chosen attributes to be used as candidates for the split at each Node. 
            trees (list of obj): - stores the built decison trees for the forest.
        
            Notes:
                Loosely based on sklearn's RandomForestClassifier: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
    """

    def __init__(self, N=100, M=None, F=None):
        """
            Purpose: Initializer for MyRandomForestClassifier

            Args:
        """
        self.X_train = None
        self.y_train = None

        self.N = N
        self.M = M 
        self.F = F 

        self.trees = []


    def generate_test_set(self, X, y, test_size = 0.33, random_state=None):
        """
            Purpose: Generates a random stratified test and training sets for a given X and y

            Args:
                X (list of list of obj): the list of rows of the X-values 
                    The shape of X is (n_samples, n_features)
                y (list of obj): the target y values (parallel to X)
                    The shape of y is n_samples
                test_size (float): the proportion of the dataset to be in the test set
                random_state (int): integer used for seeding a random number generator
                    for reproducible results
            
            Returns:
                X_train (list of list of obj): The list of training instances (samples)
                    The shape of X_train is (n_train_samples, n_features)
                X_test (list of list of obj): The list of testing instances (samples)
                    The shape of X_test is (n_test_samples, n_features)
                y_train (list of obj): The target y values (parallel to X_train)
                    The shape of y_train is n_train_samples
                y_test (list of obj): The true y values (parallel to X_test)
                    The shape of y_test is n_test_samples


            Notes: - test set = 1/3 of original dataset.
                   - train set = remaining 2/3 of original dataset.
        """
        if random_state is not None:
            np.random.seed(random_state)

        test_length = round(test_size * len(X)) # finds number of testing instances (considering test_size proportion)

        # divide data rows based on class label
        divided_X = {}
        divided_y = {}

        for xval, yval in zip(X, y):
            if yval not in divided_X:
                divided_X[yval] = []
                divided_y[yval] = []
            divided_X[yval].append(xval)
            divided_y[yval].append(yval)

        class_counts = {} # number of rows for each class label
        for label in divided_X:
            class_counts[label] = len(divided_y[label])

        proportion = {} # finds test size per class label size
        for label in divided_X:
            proportion[label] = class_counts[label] * test_length / len(X)

        rounded_cts = {} # rounds test sizes
        for key, value in proportion.items():
            rounded_cts[key] = int(round(value))

        # in case rounding impacts test sizes
        difference = test_length - sum(rounded_cts.values())
        if difference != 0:  # if we lose number of testing rows due to rounding down
            fraction = {} # find classes with highest number/fraction
            for label in divided_X:
                fraction[label] = proportion[label] - int(proportion[label])
            sorted_labels = sorted(fraction, key = lambda k: fraction[k], reverse = True)
            
            # keep adding/subtracting rows until sum(rounded_counts) == test_length
            for label in sorted_labels[:abs(difference)]:
                rounded_cts[label] += np.sign(difference)

        X_test = []
        y_test = []
        X_train = []
        y_train = []

        # shuffles sample indexes inside class label
        for label in divided_X:

            index = np.arange(class_counts[label])
            np.random.shuffle(index)

            test_k = rounded_cts[label]

            # choose test/train indices based on fraction of testing
            test_index = index[:test_k]
            train_index = index[test_k:]

            for i in test_index:
                X_test.append(divided_X[label][i])
                y_test.append(label)

            for i in train_index:
                X_train.append(divided_X[label][i])
                y_train.append(label)
        
        # shuffle training and test, since divided by class label
        def shuffle_parallel(X, y):
            index = np.arange(len(X))
            np.random.shuffle(index)
            return [X[i] for i in index], [y[i] for i in index]

        X_train, y_train = shuffle_parallel(X_train, y_train)
        X_test, y_test = shuffle_parallel(X_test, y_test)

        # save X_train and y_train to class object
        self.X_train = X_train
        self.y_train = y_train

        return X_train, X_test, y_train, y_test
        

    def fit(self, X_train, y_train, random_state = None):
        '''
            Purpose: fits a random forest classifier to X_train and y_train using the TDIDT algorithm

            Args:
                X_train (list of list of obj): - The list of training instances (samples).
                                               - has shape: (n_train_samples, n_features)
                y_train (list of obj): - The target y values (labels corresponding to X_train)
                                       - has shape: n_train_samples
                test_size (float): - the proportion of the dataset to be in the test set
                                   - used to feed into the generate_test_class function
                random_state (int) - integer used for seeding a random number generator
                                     for reproducible results
                                   - used to feed into the Decision Tree class, for when picking F
                                     random attributes to use as candidates to partition data on
            Notes: - construct N decision trees, choose the M most accurate trees of these N to form the forest which will be used during classification tasks.
            
        '''
        
        candidate_trees = []

        if X_train is None:
            X_train = self.X_train
        
        if y_train is None:
            y_train = self.y_train
        
        # if M is not set, M is N - 1 (1 less than the number of trees because M < N (M is a subset to N))
        if self.M is None: 
            self.M = self.N - 1

        # generate, train, and evaluate trees for the forest (up to N trees)
        for __ in range(self.N):

            # generate random subsamples of data for training and evaluating the tree
            X_train, X_val, y_train, y_val = myevaluation.bootstrap_sample(X_train, y_train, random_state=random_state)

            # create a new tree object for the forest
            tree = MyDecisionTreeClassifier(F = self.F)
            
            # train the tree on the training data (samples and corresp. labels)
            tree.fit(X_train, y_train)

            # predict confidence rating for test instances
            y_pred = tree.predict(X_val)

            # compute the accuracy of the model compared to true and predicted class labels
            acc = myevaluation.accuracy_score(y_val, y_pred)

            # append the trained tree and its accuracy score to tuple of trees (growing forest)
            candidate_trees.append((tree, acc))

        # once all N trees have been trained and evaluted against the bootstrap test samples, want to choose
        # the M most accurate to form the forest
        candidate_trees.sort(key = lambda x: x[1], reverse = True) # sort trees by accuracy (highest --> lowest)
        self.trees = [tree for tree, __ in candidate_trees[:self.M]] # use the first M trees (the M trees with the highest accuracy)
                                                                     # to form the forest
    
    def predict(self, X_test):
        '''
            Purpose: predicts the class labels of X_test, found earlier when fitting the Random Forestb
        
            Returns: y_pred (list of obj): - the predicted class labels, parallel to the x-values in the testing set

            Notes: - there are no arguments because the dataset has already been divided into training
                     and testing sets when fitting the classifier (the generate_test_set function has 
                     already been called)
        '''
        y_pred = []


        # iterate through each test in X_test, as each test instance goes through M decision trees
        for test in X_test:

            curr_y_pred = []
            for tree in self.trees: # iterate through each tree, and saving the predicted class label for each tree
                curr_y_pred.extend(tree.predict([test]))
            
            # find the most frequent class label
            y_freq = myutils.get_frequency(curr_y_pred)
            y_pred.append(max(y_freq, key = y_freq.get))
        
        return y_pred

class MyNaiveBayesClassifier:
    """Represents a Naive Bayes classifier.

    Attributes:
        priors(dictionary of float values): The prior probabilities computed for each label in the training set.
            priors: {label1: P(label1), label2: P(label2), etc.}

        conditionals(dictionary of lists of dictionary of float values): The conditional probabilities 
            computed for each attribute value/label pair in the training set.
            conditionals: {label1: [{attribute1: P(attribute1|label1), attribute2: P(attribute2|label1)}],
                           label2: [{attribute1: P(attribute1|label2), attribute2: P(attribute2|label2)}]}

    Notes:
        Loosely based on sklearn's Naive Bayes classifiers: https://scikit-learn.org/stable/modules/naive_bayes.html
        You may add additional instance attributes if you would like, just be sure to update this docstring
        Terminology: instance = sample = row and attribute = feature = column
    """
    def __init__(self):
        """Initializer for MyNaiveBayesClassifier.
        """
        self.priors = None
        self.conditionals = None


    def fit(self, X_train, y_train):
        """Fits a Naive Bayes classifier to X_train and y_train.

        Args:
            X_train(list of list of obj): The list of training instances (samples)
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples

        Notes:
            Since Naive Bayes is an eager learning algorithm, this method computes the prior probabilities
                and the conditional probabilities for the training data.
            You are free to choose the most appropriate data structures for storing the priors
                and conditionals.
        """
        
        self.priors = {}

        classes = myutils.get_frequency(y_train) # finds unique class labels in y_train

        for y_class in classes.keys():
            # sets class label key to (frequency in y_train)/(total rows in y_train) = P(class_label)
            self.priors[y_class] = (classes[y_class] / len(y_train)) 
        
        # following loop creates a 2d list of each column (instead of each row)
        columns = []
        for col_index in range(len(X_train[0])):
            columns.append(myutils.get_col(X_train, col_index))
        
        class_vals = {}
        for key in classes.keys(): # for each unique class
            classifier = []
            for row in columns:
                col_val_freq = {}
                for index, value in enumerate(row): # traverses down each column values
                    if y_train[index] == key:
                        if value in col_val_freq.keys(): # finds numerator for each conditional
                            col_val_freq[value] += 1
                        else:
                            col_val_freq[value] = 1
                
                col_vals = {}

                myKeys = list(col_val_freq.keys())
                myKeys.sort() # sorts keys so all class label's conditional is in the same order
                # order for classes is ascending if numerical, alphabetical if string

                col_val_freq = {i: col_val_freq[i] for i in myKeys} # reorders conditionals to ascending/alphabetical

                for col in col_val_freq.keys():
                    col_vals[col] = col_val_freq[col] / classes[key] # sets conditional to fraction of number of attributes in class label
                
                classifier.append(col_vals) # so conditional is divided using a list by class label

            class_vals[key] = classifier # so each conditional set (by label) is divided using a dictionary
            self.conditionals = class_vals


    def predict(self, X_test):
        """Makes predictions for test instances in X_test.

        Args:
            X_test(list of list of obj): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        y_pred = []
        for row in X_test:
            calc_compare = {}
            for key in self.conditionals:
                calculation = 1 # set to 1 since we're multiplying - can't be 0 or else it will equal 0
                for index, value in enumerate(row):
                    try:
                        calculation *= self.conditionals[key][index][value] # multiplies each conditional based on class label, attribute, and attribute value
                    except KeyError: # in case value is 0 (not sure why it doesn't work without this, since 0 * number = 0)
                        calculation = 0
                
                calc_compare[key] = calculation * self.priors[key] # multiplies by class label probability
            highest_num = 0
            
            # finds highest calculation for each set of labels for each test value
            for key in calc_compare:
                if calc_compare[key] > highest_num:
                    highest_num = calc_compare[key]
                    highest_num_key = key
            y_pred.append(highest_num_key) # highest value's class label is added into a y_pred

        return y_pred
