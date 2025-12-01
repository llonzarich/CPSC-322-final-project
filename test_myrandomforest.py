import numpy as np
from mysklearn.myclassifiers import MyDecisionTreeClassifier, MyRandomForestClassifier
from mysklearn import myutils

def test_generate_test_split():
    '''
        Purpose: test function for generating test set split.
    '''
    # interview dataset
    X_train_interview = [
            ["Senior", "Java", "no", "no"], # False
            ["Senior", "Java", "no", "yes"], # False
            ["Mid", "Python", "no", "no"], # True
            ["Junior", "Python", "no", "no"], # True
            ["Junior", "R", "yes", "no"], # True
            ["Junior", "R", "yes", "yes"], # False
            ["Mid", "R", "yes", "yes"], # True
            ["Senior", "Python", "no", "no"], # False
            ["Senior", "R", "yes", "no"], # True
            ["Junior", "Python", "yes", "no"], # True
            ["Senior", "Python", "yes", "yes"], # True
            ["Mid", "Python", "no", "yes"], # True
            ["Mid", "Java", "yes", "no"], # True
            ["Junior", "Python", "no", "yes"] # False
        ]
    y_train_interview = ["False", "False", "True", "True", "True", "False", "True", "False", "True", "True", "True", "True", "True", "False"]
    
    myForest = MyRandomForestClassifier()
    myForest.generate_test_set(X_train_interview, y_train_interview)

    # check to see that the sizes of both the training and test sets combined are 
    # the same size as the original dataset (same number of rows)
    assert len(myForest.X_test) + len(myForest.X_train) == 14

    # check to see that a row in the testing set is not in the training set
    # (no duplicates between training and testing sets), and vice versa
    for row in myForest.X_test:
        assert row not in myForest.X_train

    for row in myForest.X_train:
        assert row not in myForest.X_test

    # check to see the test set has the same class label distribution as 
    # the original dataset
    actual_y_freq = myutils.get_frequency(y_train_interview)
    testing_y_freq = myutils.get_frequency(myForest.y_test)
    for key in testing_y_freq:
        assert round(testing_y_freq[key]/len(myForest.y_test)) == round(actual_y_freq[key]/len(y_train_interview))


def test_random_forest_classifier_fit():
    '''
        Purpose: test function for random forest classifier model training function.
    '''
    # interview dataset
    X_train_interview = [
            ["Senior", "Java", "no", "no"], # False
            ["Senior", "Java", "no", "yes"], # False
            ["Mid", "Python", "no", "no"], # True
            ["Junior", "Python", "no", "no"], # True
            ["Junior", "R", "yes", "no"], # True
            ["Junior", "R", "yes", "yes"], # False
            ["Mid", "R", "yes", "yes"], # True
            ["Senior", "Python", "no", "no"], # False
            ["Senior", "R", "yes", "no"], # True
            ["Junior", "Python", "yes", "no"], # True
            ["Senior", "Python", "yes", "yes"], # True
            ["Mid", "Python", "no", "yes"], # True
            ["Mid", "Java", "yes", "no"], # True
            ["Junior", "Python", "no", "yes"] # False
        ]
    y_train_interview = ["False", "False", "True", "True", "True", "False", "True", "False", "True", "True", "True", "True", "True", "False"]

    myForest = MyRandomForestClassifier(N=20, M=7, F=2)
    myForest.fit(X_train_interview, y_train_interview)

    # retrieve the forest from the fit() method.
    forest = myForest.trees

    assert forest is not None # make sure the forest has trees (the forest should be a list of tree obj's)
    assert len(forest) == 7   # make sure the forest has the correct # of trees.
    
    # make sure each tree is a decision tree class obj.
    for tree in forest:
        assert isinstance(tree, MyDecisionTreeClassifier)

    # make sure each tree isn't empty.
    for tree in forest:
        assert tree.tree is not None



def test_random_forest_classifier_predict():
    pass
