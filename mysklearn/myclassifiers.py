import numpy as np
from mysklearn import myutils
from mysklearn import myevaluation


class MyDecisionTreeClassifier:
    """Represents a decision tree classifier.

    Attributes:
        X_train(list of list of obj): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
        y_train(list of obj): The target y values (parallel to X_train).
            The shape of y_train is n_samples
        tree(nested list): The extracted tree model.

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
                # print("1")
                return ["Leaf", y_train[0], len(y_train), prev_length]
            
            # CASE 2: there are no more attributes to split on, and still don't have the same class labels
            # ==> take the most frequent class label, and if there is a tie: take the class label that comes first alphabetically
            if len(current_att) == 0: # returns leaf of most frequent class label/first alphabetically
                # print("2")
                return ["Leaf", myutils.most_freq_class(y_train), len(y_train), prev_length]

            # for the random random forest classifier: selects F random attributes as partition candidate attributes 
            if self.F is not None:
                temp_attr = current_att[:]
                np.random.shuffle(temp_attr)
                temp_attr = temp_attr[:self.F]
            else:
                temp_attr = current_att[:]

            # finds index of attribute with lowest entropy
            # print("3")
            # print(f"current_att: {current_att}")
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
            
            while curr_tree[0] != "Leaf": # continues down tree until it reaches a leaf node
                
                if curr_tree[0] == "Attribute":
                    attr_num = int(curr_tree[1][3]) # saves attribute index to check if value in X_test is same as value in attribute

                curr_subtree = curr_tree[2:]
                found = False
                for sub in curr_subtree: # finds the attribute value that equals the test's value for the same attribute
                    if sub[1] == test[attr_num]:
                        curr_tree = sub[2]
                        found = True
                if not found:
                    maj_value = []
                    for sub in curr_subtree:
                        maj_value.append(sub[2][1])
                    y_pred.append(myutils.most_freq_class(maj_value))
                    break

                if curr_tree[0] == "Leaf":
                    y_pred.append(curr_tree[1])

        return y_pred 





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

        self.X_test = None
        self.y_test = None

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

            Notes: - test set = 1/3 of original dataset.
                   - train set = remaining 2/3 of original dataset.
        """
        if random_state is not None:
            np.random.seed(random_state)

        y_freq = myutils.get_frequency(y) # gets the frequency of each unique class label

        divided_data = {}

        for key in y_freq: # divides the data based on the class label (into a dictionary)
            curr_class_X = []
            for index, class_label in enumerate(y):
                if class_label == key:
                    curr_class_X.append(X[index])
            divided_data[key] = curr_class_X

        size_of_test = round(test_size * len(X))

        # temporary lists (before training/testing lists before it is shuffled)
        X_test = []
        y_test = []
        X_train = []
        y_train = []

        self.X_test = []
        self.y_test = []
        self.X_train = []
        self.y_train = []

        for key in y_freq: # this for-loop divides data into test and training sets
            ratio = round(y_freq[key]/len(X) * size_of_test) # finds the number of rows needed for each class label that would maintain the class-label
                                                             # distribution of the original dataset in the testing set

            for __ in range(ratio): # adds X and y rows to the testing set
                rand_index = np.random.randint(0, len(divided_data[key]))

                X_test.append(divided_data[key].pop(rand_index))
                y_test.append(key)
        
        for __ in range(len(X_test)): # shuffles the testing set (or else the values will be grouped by class label)
            rand_index = np.random.randint(0, len(X_test))
            self.X_test.append(X_test.pop(rand_index))
            self.y_test.append(y_test.pop(rand_index))

        for key in divided_data: # adds X and y rows to the training set
            X_train.extend(divided_data[key]) # since the rows for the testing set were found by popping the earlier dictionary, the training set is 
                                              # just the remainder of what is left in the dictionary
            for __ in range(y_freq[key]): y_train.append(key)
        
        for __ in range(len(X_train)): # shuffles the training set (or else the values will be grouped by class label)
            rand_index = np.random.randint(0, len(X_train))
            self.X_train.append(X_train.pop(rand_index))
            self.y_train.append(y_train.pop(rand_index))
        

    def fit(self, X_train, y_train, test_size = 0.33, random_state = None):
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
        
        # if M is not set, M is N - 1 (1 less than the number of trees because M < N (M is a subset to N))
        if self.M is None: 
            self.M = self.N - 1

        # generate test sets (1/3 of dataset) and train sets (remaining 2/3 of dataset)
        self.generate_test_set(X_train, y_train, test_size, random_state=random_state)

        # generate, train, and evaluate trees for the forest (up to N trees)
        for __ in range(self.N):

            # generate random subsamples of data for training and evaluating the tree
            X_train, X_test, y_train, y_test = myevaluation.bootstrap_sample(self.X_train, self.y_train, random_state=random_state)

            # create a new tree object for the forest
            tree = MyDecisionTreeClassifier(F = self.F)
            
            # train the tree on the training data (samples and corresp. labels)
            tree.fit(X_train, y_train)

            # predict confidence rating for test instances
            y_pred = tree.predict(X_test)

            # compute the accuracy of the model compared to true and predicted class labels
            acc = myevaluation.accuracy_score(y_test, y_pred)
            
            # append the trained tree and its accuracy score to tuple of trees (growing forest)
            candidate_trees.append((tree, acc))

        # once all N trees have been trained and evaluted against the bootstrap test samples, want to choose
        # the M most accurate to form the forest
        candidate_trees.sort(key = lambda x: x[1], reverse = True) # sort trees by accuracy (highest --> lowest)
        self.trees = [tree for tree, __ in candidate_trees[:self.M]] # use the first M trees (the M trees with the highest accuracy)
                                                                     # to form the forest
    
    def predict(self):
        '''
            Purpose: predicts the class labels of X_test, found earlier when fitting the Random Forestb
        
            Returns: y_pred (list of obj): - the predicted class labels, parallel to the x-values in the testing set

            Notes: - there are no arguments because the dataset has already been divided into training
                     and testing sets when fitting the classifier (the generate_test_set function has 
                     already been called)
        '''
        y_pred = []

        # iterate through each test in X_test, as each test instance goes through M decision trees
        for test in self.X_test:

            curr_y_pred = []
            for tree in self.trees: # iterate through each tree, and saving the predicted class label for each tree
                curr_y_pred.extend(tree.predict([test]))
            
            # find the most frequent class label
            y_freq = myutils.get_frequency(curr_y_pred)
            y_pred.append(max(y_freq, key = y_freq.get))
        
        return y_pred








class MyNaiveBayesClassifier:
    """
        Purpose: Represents a Naive Bayes classifier.

        Attributes:
            priors (YOU CHOOSE THE MOST APPROPRIATE TYPE): The prior probabilities computed for each label in the training set.
            conditionals (YOU CHOOSE THE MOST APPROPRIATE TYPE): The conditional probabilities computed for each attribute value/label pair in the training set.

        Notes:
            Loosely based on sklearn's Naive Bayes classifiers: https://scikit-learn.org/stable/modules/naive_bayes.html
            You may add additional instance attributes if you would like, just be sure to update this docstring
            Terminology: instance = sample = row and attribute = feature = column
    """
    def __init__(self):
        """
            Purpose: Initializer for MyNaiveBayesClassifier.
        """
        self.priors = None
        self.conditionals = None


    def fit(self, X_train, y_train):
        """
            Purpose: Fits a Naive Bayes classifier to X_train and y_train.

            Args:
                X_train (list of list of obj): - The list of training instances (samples)
                                               - has shape: (n_train_samples, n_features)
                y_train (list of obj): - The target y values (parallel to X_train)
                                       - has shape: n_train_samples

            Notes:
                - Since Naive Bayes is an eager learning algorithm, this method computes the prior probabilities and the conditional probabilities for the training data.
                - You are free to choose the most appropriate data structures for storing the priors and conditionals.
        """
        # ============== COMPUTE PRIORS ============= 
        # priors = the probability of each class/label occuring in the training data.
        total_num_samples = len(y_train) # get the total number of samples in the train set.

        priors = dict() # create a dictionary to store priors.

        # count the number of samples per label.
        for label in y_train:
            if label not in priors:
                priors[label] = 0
            priors[label] += 1

        # iterate through each key (class label) in the dictionary and compute its prior probability: key value / total number of samples. Store it in a dictionary.
        self.priors = {label: priors[label] / total_num_samples for label in priors}
    

        # ============= COUNT CLASS OCCURENCES ===================
        # define an empty dictionary to store data for computing conditionals 
        # note: this will become a nested dictionary stored as: class label --> feature_idx --> feature_val: count. 
        counts = dict()

        # iterate through each sample (aka, row) in the train set. (example: row 1)
        for sample_idx, sample_val in enumerate(X_train):            
            label = y_train[sample_idx] # grab the current sample's class label (there is one of these per instance, or row) (example: "yes")
            
            # if the label is not already a key in the dictionary, make it one.
            if label not in counts:
                counts[label] = dict()
            
            # iterate through each feature in the current sample (aka, row). (example: "color" "shape" "size") and get its value (example: "yellow", "square", "big") so we can increment the correct key in the nested dictionary.
            for feature_idx, feature_val in enumerate(sample_val):
                # if the feature is not already a key in the dictionary, make it one. note: I'm using feature_idx instead of the string feature name for ease.
                if feature_idx not in counts[label]:
                    counts[label][feature_idx] = {}
                
                # if the value (for the current label and current feature) has not initialized, initialize it.
                if feature_val not in counts[label][feature_idx]:
                    counts[label][feature_idx][feature_val] = 0

                # increment the correct key in the nested dictionary (the value for the current label and current feature).
                counts[label][feature_idx][feature_val] += 1 


        # ========== COMPUTE CONDITIONALS =========================
        # compute the conditional probability for each class: P(X_i | class_i)
        # - conditionals formula: P(X_i | class_i)
        # - we'll use these conditionals to compute P(class_i | X_i) = (class_i prior) * prod( P(X_i | class_i) ) (for each attr, X_i) to determine unseen instance class labels. 
        self.conditionals = {} # store P(attr | class) conditionals as class --> attr (idx and val) --> probability
       
        # iterate over each label (key) in the counts dictionary.
        for label in counts:
            self.conditionals[label] = {} # initialize a dict for the each class label. this dictionary will become a nested dict.

            # iterate over each attribute in the current label. (example: "yes" is made up of attributes "color", "shape", and "size").
            for feature_idx in sorted(counts[label].keys()):
                self.conditionals[label][feature_idx] = {} # initialize a nested dict for each attribute. this nested dictionary will have another nested dictionary in it.
                feature_counts = counts[label][feature_idx] # grab the number of instances in the current label and current feature.
                count_for_label = sum(feature_counts.values()) # sum together the counts for the current label and current feature.
                num_unique_feature_vals = len(feature_counts) # find the number of unique values for the current feature.

                # iterate through each value in the dict for the current label and current feature and computes its conditional probability.
                for feature_val, count in sorted(feature_counts.items()):
                    self.conditionals[label][feature_idx][feature_val] = (count + 1) / (count_for_label + num_unique_feature_vals)
    

    def predict(self, X_test):
        """
            Purpose: - Makes predictions for test instances in X_test.
                     - for each class we will compute (class_i prior) * prod( class_i conditions ) for each feature.

            Args:
                X_test( list of list of obj): - The list of testing samples
                                              - has shape: (n_test_samples, n_features)

            Returns:
                y_predicted (list of obj): The predicted target y values (parallel to X_test)
        """
        # if X_test is made up of only 1 instance (if it's not a list), make it a list to be compatible with the rest of my function.
        one_X_test = False # assume X_test is made up of more than 1 instance.
        if not isinstance(X_test[0], list):  
            X_test = [X_test]
            one_X_test = True

        y_preds = [] # initialize a list to store a predicted class label for each unseen instance in X_test.

        # iterate over each test instance to compute its predicted label. 
        for sample in X_test:    
            posteriors = {} # initialize a dictionary for each test sample's posterior probability.

            # iterate over each class label (the keys in the priors dictionary).
            for label in self.priors:
                prior = self.priors[label] # start with the prior. 
                
                probs_multiplied = 1.0

                # iterate over each attribute and its attribute value in the current class label (that we're iterating over).
                for feature_idx, feature_val in enumerate(sample):
                    # multiply each conditional probability together. 
                    cond_prob = self.conditionals[label].get(feature_idx, {}).get(feature_val, 1e-6)
                    probs_multiplied *= cond_prob

                # compute the posterior probability.              
                probability = prior * probs_multiplied

                # add the posterior probability for this class to the posteriors dictionary.
                posteriors[label] = probability

            # let the class with the largest posterior probability be the predicted label for the current unseen instance in X_test. 
            best_class = max(posteriors, key=posteriors.get)
            y_preds.append(best_class)

        # if there was only one instance in X_test, we need to return the predicted class label NOT as a list.
        if one_X_test:
            return y_preds[0]
        else:
            return y_preds













# class MyNaiveBayesClassifier:
#     """Represents a Naive Bayes classifier.

#     Attributes:
#         priors(dictionary of float values): The prior probabilities computed for each label in the training set.
#             priors: {label1: P(label1), label2: P(label2), etc.}

#         conditionals(dictionary of lists of dictionary of float values): The conditional probabilities 
#             computed for each attribute value/label pair in the training set.
#             conditionals: {label1: [{attribute1: P(attribute1|label1), attribute2: P(attribute2|label1)}],
#                            label2: [{attribute1: P(attribute1|label2), attribute2: P(attribute2|label2)}]}

#     Notes:
#         Loosely based on sklearn's Naive Bayes classifiers: https://scikit-learn.org/stable/modules/naive_bayes.html
#         You may add additional instance attributes if you would like, just be sure to update this docstring
#         Terminology: instance = sample = row and attribute = feature = column
#     """
#     def __init__(self):
#         """Initializer for MyNaiveBayesClassifier.
#         """
#         self.priors = None
#         self.conditionals = None


#     def fit(self, X_train, y_train):
#         """Fits a Naive Bayes classifier to X_train and y_train.

#         Args:
#             X_train(list of list of obj): The list of training instances (samples)
#                 The shape of X_train is (n_train_samples, n_features)
#             y_train(list of obj): The target y values (parallel to X_train)
#                 The shape of y_train is n_train_samples

#         Notes:
#             Since Naive Bayes is an eager learning algorithm, this method computes the prior probabilities
#                 and the conditional probabilities for the training data.
#             You are free to choose the most appropriate data structures for storing the priors
#                 and conditionals.
#         """
        
#         self.priors = {}

#         classes = myutils.get_frequency(y_train) # finds unique class labels in y_train

#         for y_class in classes.keys():
#             # sets class label key to (frequency in y_train)/(total rows in y_train) = P(class_label)
#             self.priors[y_class] = (classes[y_class] / len(y_train)) 
        
#         # following loop creates a 2d list of each column (instead of each row)
#         columns = []
#         for col_index in range(len(X_train[0])):
#             columns.append(myutils.get_col(X_train, col_index))
        
#         class_vals = {}
#         for key in classes.keys(): # for each unique class
#             classifier = []
#             for row in columns:
#                 col_val_freq = {}
#                 for index, value in enumerate(row): # traverses down each column values
#                     if y_train[index] == key:
#                         if value in col_val_freq.keys(): # finds numerator for each conditional
#                             col_val_freq[value] += 1
#                         else:
#                             col_val_freq[value] = 1
                
#                 col_vals = {}

#                 myKeys = list(col_val_freq.keys())
#                 myKeys.sort() # sorts keys so all class label's conditional is in the same order
#                 # order for classes is ascending if numerical, alphabetical if string

#                 col_val_freq = {i: col_val_freq[i] for i in myKeys} # reorders conditionals to ascending/alphabetical

#                 for col in col_val_freq.keys():
#                     col_vals[col] = col_val_freq[col] / classes[key] # sets conditional to fraction of number of attributes in class label
                
#                 classifier.append(col_vals) # so conditional is divided using a list by class label

#             class_vals[key] = classifier # so each conditional set (by label) is divided using a dictionary
#             self.conditionals = class_vals


#     def predict(self, X_test):
#         """Makes predictions for test instances in X_test.

#         Args:
#             X_test(list of list of obj): The list of testing samples
#                 The shape of X_test is (n_test_samples, n_features)

#         Returns:
#             y_predicted(list of obj): The predicted target y values (parallel to X_test)
#         """
#         y_pred = []
#         for row in X_test:
#             calc_compare = {}
#             for key in self.conditionals:
#                 calculation = 1 # set to 1 since we're multiplying - can't be 0 or else it will equal 0
#                 for index, value in enumerate(row):
#                     try:
#                         calculation *= self.conditionals[key][index][value] # multiplies each conditional based on class label, attribute, and attribute value
#                     except KeyError: # in case value is 0 (not sure why it doesn't work without this, since 0 * number = 0)
#                         calculation = 0
                
#                 calc_compare[key] = calculation * self.priors[key] # multiplies by class label probability
#             highest_num = 0
            
#             # finds highest calculation for each set of labels for each test value
#             for key in calc_compare:
#                 if calc_compare[key] > highest_num:
#                     highest_num = calc_compare[key]
#                     highest_num_key = key
#             y_pred.append(highest_num_key) # highest value's class label is added into a y_pred

#         return y_pred








# ================= DECISION TREE CLASSIFIER SECTION ========================
class MyDecisionTreeSolo:
    """
        Purpose: Represents a decision tree classifier.

        Attributes:
            X_train (list of list of obj): - The list of training instances (samples).
                                           - has shape: (n_train_samples, n_features)
            y_train (list of obj): - The target y values (labels corresponding to X_train).
                                   - has shape: y_train is n_samples
            tree (nested list): The extracted tree model.

        Notes:
            Loosely based on sklearn's DecisionTreeClassifier: https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
            Terminology: instance = sample = row and attribute = feature = column
    """
    def __init__(self):
        """
            Purpose: Initializer for MyDecisionTreeClassifier.
        """
        self.X_train = None
        self.y_train = None
        self.tree = None


    def fit(self, X_train, y_train):
        """
            Purpose: Fits a decision tree classifier to X_train and y_train using the TDIDT (top down induction of decision tree) algorithm.

            Args:
                X_train (list of list of obj): - The list of training instances (samples).
                                               - has shape: (n_train_samples, n_features)
                y_train (list of obj): - The target y values (labels corresponding to X_train)
                                       - has shape: n_train_samples

            Notes:
                - Since TDIDT is an eager learning algorithm, this method builds a decision tree model from the training data.
                - Build a decision tree using the nested list representation described in class.
                - On a majority vote tie, choose first attribute value based on attribute domain ordering.
                - Store the tree in the tree attribute.
                - Use attribute indexes to construct default attribute names (e.g. "att0", "att1", ...).
        
            - the goal of fit is to create the splits (the nested list) and decide probabilites at each leaf node.
        """
        self.X_train = X_train
        self.y_train = y_train

        # find number of features in list (by looking at the length of the first row in X_train).
        num_attributes = len(X_train[0])

        # get all unique attributes and sort them alphabetically.
        header = [f"att{i}" for i in range(num_attributes)]

        # stich together X_train and y_train
        train_data = [X_train[i] + [y_train[i]] for i in range(len(X_train))]
        
        # build attribute domains (the set of all values that an attribut can take)
        attribute_domains = {} # initialize a dict to store each attribute's unique values (i.e., att1="job_status" has values 1, 2, 3)
        
        # iterate through each col (each attribute corresponds to 1 col) and find unique values in that col using list(set())
        for idx in range(num_attributes):
            attribute_domains[header[idx]] = sorted(list(set(row[idx] for row in X_train)))
        
        # make a copy a header, b/c python is pass by object reference and tdidt will be removing attributes from available_attributes
        available_attributes = header.copy()
        
        # create the decision tree using top down induction decision tree (tdidt)
        # self.tree = tdidt(train_data, available_attributes)
        self.tree = myutils.tdidt_dt(train_data, available_attributes, header, attribute_domains)

        

    def predict(self, X_test):
        """
            Purpose: Makes predictions for test instances in X_test.

        Args:
            X_test (list of list of obj): - The list of testing samples
                                          - has shape: (n_test_samples, n_features)

        Returns:
            y_predicted (list of obj): The predicted target y values (labels corresponding to X_test)
        """
        # find number of features in list.
        num_attributes = len(X_test[0])

        # get all unique attributes and sort them alphabetically.
        header = [f"att{i}" for i in range(num_attributes)]

        y_predicted = []

        # generate a prediction for each instance in the test set (the instances we want to classify)
        for instance in X_test:
            pred = myutils.tdidt_predict_dt(self.tree, instance, header)
            y_predicted.append(pred)

        return y_predicted 



