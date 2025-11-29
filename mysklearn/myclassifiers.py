import numpy as np
from mysklearn import myutils

class MyRandomForestClassifier:
    """Represents a random forest classifier.

    Attributes:
        X_train (list of list of obj): the list of training instances (samples).
            The shape of X_train is (n_train_samples, n_features)
        y_train (list of obj): the target y values (parallel to X_train).
            The shape of y_train is n_samples
        X_test (list of list of obj): the list of testing instances (samples).
            The shape of X_test is (n_test_samples, n_features)
        y_test (list of obj): the (true) y values (parallel to X_test).
            The shape of y_test is n_samples
        
    """
    def __init__(self):
        """Initializer for MyRandomForestClassifier
        
        """
        self.X_train = None
        self.y_train = None

        self.X_test = None
        self.y_test = None

    def generate_test_set(self, X, y, test_size = 0.33, random_state = None):
        """Generates a random stratified test and training sets for a given X and y

        Args:
            X (list of list of obj): the list of rows of the X-values 
                The shape of X is (n_samples, n_features)
            y (list of obj): the target y values (parallel to X)
                The shape of y is n_samples
            test_size (float): the proportion of the dataset to be in the 
                test set
            random_state (int): integer used for seeding a random number generator
                for reproducible results
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

        

    def fit():
        pass

    def predict():
        pass