from mysklearn.myclassifiers import MyRandomForestClassifier, MyDecisionTreeClassifier
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
    X_train, X_test, y_train, y_test = myForest.generate_test_set(X_train_interview, y_train_interview)

    # check to see that the sizes of both the training and test sets combined are 
    # the same size as the original dataset (same number of rows)
    assert len(X_test) + len(X_train) == 14

    # check to see that a row in the testing set is not in the training set
    # (no duplicates between training and testing sets), and vice versa
    for row in X_test:
        assert row not in X_train

    for row in X_train:
        assert row not in X_test

    # check to see the test set has the same class label distribution as 
    # the original dataset
    actual_y_freq = myutils.get_frequency(y_train_interview)
    testing_y_freq = myutils.get_frequency(y_test)
    for key in testing_y_freq:
        assert round(testing_y_freq[key]/len(y_test)) == round(actual_y_freq[key]/len(y_train_interview))

    # check to see if training and test sets are both equal length
    assert len(X_train) == len(y_train)

    assert len(X_test) == len(y_test)


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
    X_train, X_test, y_train, y_test = myForest.generate_test_set(X_train_interview, y_train_interview)
    myForest.fit(X_train, y_train)

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
    myForest.fit(X_train_interview, y_train_interview, random_state=8)

    assert len(myForest.predict()) == 5 # make sure there are 5 class predictions (0.33 of original dataset, as divided)

    y_pred = []
    for test in myForest.X_test:
        # finds all predictions from all M trees
        test_predictions = []
        for tree in myForest.trees:
            test_predictions.extend(tree.predict([test]))
        
        mfv = {} # finds most frequent value/predicted class label
        for pred in test_predictions:
            if pred not in mfv:
                mfv[pred] = 0
            mfv[pred] += 1
        y_pred.append(max(mfv, key=mfv.get))

    assert myForest.predict() == y_pred # checks the most frequent class label for each tests' forest is set as the class label for the test instance

