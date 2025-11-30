import numpy as np
import math

from mysklearn import myutils
from mysklearn.myutils import tdidt, tdidt_predict

from mysklearn import myevaluation
from mysklearn.myevaluation import bootstrap_sample, accuracy_score


class MyDecisionTreeClassifier:
    """
        Purpose: Represents a decision tree classifier.

        Attributes:
            X_train (list of list of obj): - The list of training instances (samples).
                                           - has shape: (n_train_samples, n_features)
            y_train (list of obj): - The target y values (labels corresponding to X_train).
                                   - has shape: y_train is n_samples
            tree (nested list): The extracted tree model.
            F (int): the number of attributes to use consider for the split at each node when building the decision tree.
        Notes:
            Loosely based on sklearn's DecisionTreeClassifier: https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
            Terminology: instance = sample = row and attribute = feature = column
    """
    def __init__(self, F=None):
        """
            Purpose: Initializer for MyDecisionTreeClassifier.
        """
        self.X_train = None
        self.y_train = None
        self.tree = None
        self.F = F


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
        F = self.F

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
        # F = math.floor(math.sqrt(num_attributes)) # set the number of attributes to consider as candidates for selecting an attribute to split on at each node.
        self.tree = tdidt(train_data, available_attributes, header, attribute_domains, F)

        

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
            pred = tdidt_predict(self.tree, instance, header)
            y_predicted.append(pred)

        return y_predicted 




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
    # def __init__(self, N, M, F):
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


    def generate_test_set(self, X, y, test_size = 0.33, random_state = None):
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

        

    def fit(self, X_train, y_train):
        '''
            Purpose: fits a random forest classifier to X_train and y_train using the TDIDT algorithm

            Args:
                X_train (list of list of obj): - The list of training instances (samples).
                                               - has shape: (n_train_samples, n_features)
                y_train (list of obj): - The target y values (labels corresponding to X_train)
                                       - has shape: n_train_samples
                M (int): - the number of trees to be included in the forest. 
                         - M is a subset of N (where N is the total number of trees trained and evaluated)
                         - necessary because we only want to take a certain number of the best trees to for our forest.
                         - if None, let M be N-1 (1 less than the number of trees because M < N)
            Notes: - construct N decision trees, choose the M most accurate trees of these N to form the forest which will be used during classification tasks.
            
        '''
        N = self.N
        M = self.M if self.M is not None else N-1
        F = self.F

        candidate_trees = [] # to store the M best trees (M subset of N) (N = number of trees trained))

        # generate test sets (1/3 of dataset) and train sets (remaining 2/3 of dataset).
        self.generate_test_set(X_train, y_train)
        X_train = self.X_train
        y_train = self.y_train

        # generate, train, and evaluate trees for the forest.
        for i in range(N):

            # generate random subsamples of data for training and evaluating the tree.
            X_sample, X_out_of_bag, y_sample, y_out_of_bag = bootstrap_sample(X_train, y_train)

            # create a new tree object for the forest.
            tree = MyDecisionTreeClassifier(F=F)

            # train the tree on the training data (samples and corresp. labels).
            tree.fit(X_sample, y_sample)

            # predict confidence rating for test instances. 
            y_pred = tree.predict(X_out_of_bag)

            # compute the accuracy and error rate of the model by comparing true and predicted conf. ratings.
            acc = accuracy_score(y_out_of_bag, y_pred)
            err = 1 - acc
        
            # append the trained tree and its acc to our tuple of trees (our growing forest).
            candidate_trees.append((tree, acc))
        
        # once all N trees have been trained and evaluated against the bootstrap test samples, we want to choose the M most accurate to form the forest.
        candidate_trees.sort(key=lambda x: x[1], reverse=True) # sort trees by acc (highest --> lowest).
        self.trees = [tree for tree, acc in candidate_trees[:M]] # use the first M trees (the M trees with the highest acc) to form the forest.


    def predict():
        pass



