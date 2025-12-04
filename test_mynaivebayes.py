import numpy as np

from mysklearn.myclassifiers import MyNaiveBayesClassifier

def test_naive_bayes_classifier_fit():
    '''
        Purpose: test for the naive bayes classification model training function.
    '''
    # case 1: use the 8-instance training set example (from class), asserting against our desk check of the priors and conditional probabilities.
    header1 = ["att1", "att2"]
    X_train1 = [
            [1, 5],
            [2, 6],
            [1, 5],
            [1, 5],
            [1, 6],
            [2, 6],
            [1, 5],
            [1, 6]
        ]
    y_train1 = ["yes", "yes", "no", "no", "yes", "no", "yes", "yes"] # parallel to X_train

    # train the model.
    clf1 = MyNaiveBayesClassifier()
    clf1.fit(X_train1, y_train1)

    # check class priors. 
    assert np.isclose(clf1.priors["yes"], 5/8)
    assert np.isclose(clf1.priors["no"], 3/8)

    # check conditional probabilities.
    assert np.isclose(clf1.conditionals["yes"][0][1], 4/5)
    assert np.isclose(clf1.conditionals["yes"][0][2], 1/5)

    assert np.isclose(clf1.conditionals["yes"][1][5], 2/5)
    assert np.isclose(clf1.conditionals["yes"][1][6], 3/5)

    assert np.isclose(clf1.conditionals["no"][0][1], 2/3)
    assert np.isclose(clf1.conditionals["no"][0][2], 1/3)

    assert np.isclose(clf1.conditionals["no"][1][5], 2/3)
    assert np.isclose(clf1.conditionals["no"][1][6], 1/3)


    # case 2: use the 15 instance training set example (from LA7), asserting against your desk check of the priors and conditional probabilities.
    header2 = ["standing", "job_status", "credit_rating", "buys_iphone"]
    X_train2 = [
            [1, 3, "fair"],
            [1, 3, "excellent"], 
            [2, 3, "fair"],
            [2, 2, "fair"],
            [2, 1, "fair"],
            [2, 1, "excellent"],
            [2, 1, "excellent"],
            [1, 2, "fair"],
            [1, 1, "fair"],
            [2, 2, "fair"],
            [1, 2, "excellent"],
            [2, 2, "excellent"],
            [2, 3, "fair"],
            [2, 2, "excellent"],
            [2, 3, "fair"]
        ]
    y_train2 = ["no", "no", "yes", "yes", "yes", "no", "yes", "no", "yes", "yes", "yes", "yes", "yes", "no", "yes"]
    
    clf2 = MyNaiveBayesClassifier()
    clf2.fit(X_train2, y_train2)

    # check class priors. 
    assert np.isclose(clf2.priors["yes"], 10/15)
    assert np.isclose(clf2.priors["no"], 5/15)

    # check conditional probabilities.
    assert np.isclose(clf2.conditionals["yes"][0][1], 2/10)
    assert np.isclose(clf2.conditionals["yes"][0][2], 8/10)
    assert np.isclose(clf2.conditionals["yes"][1][1], 3/10)
    assert np.isclose(clf2.conditionals["yes"][1][2], 4/10)
    assert np.isclose(clf2.conditionals["yes"][1][3], 3/10)
    assert np.isclose(clf2.conditionals["yes"][2]["fair"], 7/10)
    assert np.isclose(clf2.conditionals["yes"][2]["excellent"], 3/10)

    assert np.isclose(clf2.conditionals["no"][0][1], 3/5)
    assert np.isclose(clf2.conditionals["no"][0][2], 2/5)
    assert np.isclose(clf2.conditionals["no"][1][1], 1/5)
    assert np.isclose(clf2.conditionals["no"][1][2], 2/5)
    assert np.isclose(clf2.conditionals["no"][1][3], 2/5)
    assert np.isclose(clf2.conditionals["no"][2]["fair"], 2/5)
    assert np.isclose(clf2.conditionals["no"][2]["excellent"], 3/5)

    # case 3: Use Bramer 3.2 Figure 3.1 train dataset example, asserting against the priors and conditional probabilities in Figure 3.2.
    header3 = ["day", "season", "wind", "rain"]
    X_train3 = [
        ["weekday", "spring", "none", "none"],
        ["weekday", "winter", "none", "slight"],
        ["weekday", "winter", "none", "slight"],
        ["weekday", "winter", "high", "heavy"],
        ["saturday", "summer", "normal", "none"],
        ["weekday", "autumn", "normal", "none"],
        ["holiday", "summer", "high", "slight"],
        ["sunday", "summer", "normal", "none"],
        ["weekday", "winter", "high", "heavy"],
        ["weekday", "spring", "none", "none"],
        ["saturday", "spring", "high", "heavy"],
        ["weekday", "summer", "high", "slight"],
        ["saturday", "winter", "normal", "none"],
        ["weekday", "summer", "high", "none"],
        ["weekday", "winter", "normal", "heavy"],
        ["saturday", "autumn", "high", "slight"],
        ["weekday", "autumn", "none", "heavy"],
        ["holiday", "spring", "normal", "slight"],
        ["weekday", "spring", "normal", "none"],
        ["weekday", "spring", "normal", "slight"],
    ]
    y_train3 = ["on time", "on time", "on time", "late", "on time", "very late", "on time", "on time", "very late", "on time", "cancelled", "on time", "late", "on time", "very late", "on time", "on time", "on time", "on time", "on time"]
    
    clf3 = MyNaiveBayesClassifier()
    clf3.fit(X_train3, y_train3)

    # check class priors. 
    assert np.isclose(clf3.priors["on time"], 0.70)
    assert np.isclose(clf3.priors["very late"], 0.15)
    assert np.isclose(clf3.priors["late"], 0.10)
    assert np.isclose(clf3.priors["cancelled"], 0.05)

    # check conditional probabilities.
    assert np.isclose(clf3.conditionals["on time"][0]["weekday"], 0.64, atol=1e-2)
    assert np.isclose(clf3.conditionals["on time"][1]["winter"], 0.14, atol=1e-2)
    assert np.isclose(clf3.conditionals["on time"][2]["high"], 0.29, atol=1e-2)
    assert np.isclose(clf3.conditionals["on time"][3]["heavy"], 0.07, atol=1e-2)

    assert np.isclose(clf3.conditionals["very late"][0]["weekday"], 1, atol=1e-2)
    assert np.isclose(clf3.conditionals["very late"][1]["winter"], 0.67, atol=1e-2)
    assert np.isclose(clf3.conditionals["very late"][2]["high"], 0.33, atol=1e-2)
    assert np.isclose(clf3.conditionals["very late"][3]["heavy"], 0.67, atol=1e-2)

    assert np.isclose(clf3.conditionals["late"][0]["weekday"], 0.5, atol=1e-2)
    assert np.isclose(clf3.conditionals["late"][1]["winter"], 1, atol=1e-2)
    assert np.isclose(clf3.conditionals["late"][2]["high"], 0.5, atol=1e-2)
    assert np.isclose(clf3.conditionals["late"][3]["heavy"], 0.5, atol=1e-2)

    # assert np.isclose(clf3.conditionals["cancelled"][0]["weekday"], 0)
    # assert np.isclose(clf3.conditionals["cancelled"][1]["winter"], 0)
    assert np.isclose(clf3.conditionals["cancelled"][2]["high"], 1, atol=1e-2)
    assert np.isclose(clf3.conditionals["cancelled"][3]["heavy"], 1, atol=1e-2)



def test_naive_bayes_classifier_predict():
    # case 1: use the 8-instance training set example (from class), asserting against our desk check of the priors and conditional probabilities.
    header1 = ["att1", "att2"]
    X_train1 = [
            [1, 5],
            [2, 6],
            [1, 5],
            [1, 5],
            [1, 6],
            [2, 6],
            [1, 5],
            [1, 6]
        ]
    y_train1 = ["yes", "yes", "no", "no", "yes", "no", "yes", "yes"] # parallel to X_train
    X_test1 = [1, 5]

    # get predicted class label for our unseen instance using our naive bayes classifier.
    clf1 = MyNaiveBayesClassifier()
    clf1.fit(X_train1, y_train1)
    y_pred1 = clf1.predict(X_test1)

    # get true class label for our unseen instance using desk calculations.
    p_yes1 = 5/8
    p_1_given_yes = 4/5
    p_5_given_yes = 2/5
    p_yes_given_Xtest1 = p_yes1 * p_1_given_yes * p_5_given_yes
    
    p_no1 = 3/8
    p_1_given_no = 2/3
    p_5_given_no = 2/3
    p_no_given_Xtest1 = p_no1 * p_1_given_no * p_5_given_no

    if p_yes_given_Xtest1 > p_no_given_Xtest1:
        expected_class1 = "yes"
    else: # note: if the probabilite are the same, I just choose the label.
        expected_class1 = "no"
    
    assert expected_class1 == y_pred1


    # case 2: use the 15 instance training set example (from LA7), asserting against your desk check of the priors and conditional probabilities.
    header2 = ["standing", "job_status", "credit_rating", "buys_iphone"]
    X_train2 = [
            [1, 3, "fair"],
            [1, 3, "excellent"], 
            [2, 3, "fair"],
            [2, 2, "fair"],
            [2, 1, "fair"],
            [2, 1, "excellent"],
            [2, 1, "excellent"],
            [1, 2, "fair"],
            [1, 1, "fair"],
            [2, 2, "fair"],
            [1, 2, "excellent"],
            [2, 2, "excellent"],
            [2, 3, "fair"],
            [2, 2, "excellent"],
            [2, 3, "fair"]
        ]
    y_train2 = ["no", "no", "yes", "yes", "yes", "no", "yes", "no", "yes", "yes", "yes", "yes", "yes", "no", "yes"]
    X_test2 = [
        [2, 2, "fair"], 
        [1, 1, "excellent"]
    ] 

    # get predicted class label for our unseen instances using our naive bayes classifier.
    clf2 = MyNaiveBayesClassifier()
    clf2.fit(X_train2, y_train2)

    y_pred2 = clf2.predict(X_test2)

    # get true class label for our unseen instances using (previously calculated) desk calculations.
    expected_label_Xtest2_1 = "yes" # from LA7
    expected_label_Xtest2_2 = "no" # from LA7

    assert y_pred2[0] == expected_label_Xtest2_1
    assert y_pred2[1] == expected_label_Xtest2_2

    

    # case 3: Use Bramer 3.2 Figure 3.1 train dataset example, asserting against the priors and conditional probabilities in Figure 3.2.
    header3 = ["day", "season", "wind", "rain"]
    X_train3 = [
        ["weekday", "spring", "none", "none"],
        ["weekday", "winter", "none", "slight"],
        ["weekday", "winter", "none", "slight"],
        ["weekday", "winter", "high", "heavy"],
        ["saturday", "summer", "normal", "none"],
        ["weekday", "autumn", "normal", "none"],
        ["holiday", "summer", "high", "slight"],
        ["sunday", "summer", "normal", "none"],
        ["weekday", "winter", "high", "heavy"],
        ["weekday", "spring", "none", "none"],
        ["saturday", "spring", "high", "heavy"],
        ["weekday", "summer", "high", "slight"],
        ["saturday", "winter", "normal", "none"],
        ["weekday", "summer", "high", "none"],
        ["weekday", "winter", "normal", "heavy"],
        ["saturday", "autumn", "high", "slight"],
        ["weekday", "autumn", "none", "heavy"],
        ["holiday", "spring", "normal", "slight"],
        ["weekday", "spring", "normal", "none"],
        ["weekday", "spring", "normal", "slight"],
    ]
    y_train3 = ["on time", "on time", "on time", "late", "on time", "very late", "on time", "on time", "very late", "on time", "cancelled", "on time", "late", "on time", "very late", "on time", "on time", "on time", "on time", "on time"]
    X_test3 = [
        ["weekday", "winter", "high", "heavy"],
        ["weekday", "summer", "high", "heavy"],
        ["sunday", "summer", "normal", "slight"]
    ]

    # get predicted class label for our unseen instances using our naive bayes classifier.
    clf3 = MyNaiveBayesClassifier()
    clf3.fit(X_train3, y_train3)
    y_pred3 = clf3.predict(X_test3)

    # get true class label for our unseen instances using (previously calculated) desk calculations.
    expected_label_Xtest3_1 = "very late"
    expected_label_Xtest3_2 = "on time"
    expected_label_Xtest3_3 = "on time"

    assert expected_label_Xtest3_1 == y_pred3[0]
    assert expected_label_Xtest3_2 == y_pred3[1]
    assert expected_label_Xtest3_3 == y_pred3[2]