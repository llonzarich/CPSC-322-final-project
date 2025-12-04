from mysklearn.myclassifiers import MyNaiveBayesClassifier

# note: order is actual/received student value, expected/solution
def test_naive_bayes_classifier_fit():
    myNaive = MyNaiveBayesClassifier() # reused for all examples

    # in-class Naive Bayes example (lab task #1)
    header_inclass_example = ["att1", "att2"]
    X_train_inclass_example = [
        [1, 5], # yes
        [2, 6], # yes
        [1, 5], # no
        [1, 5], # no
        [1, 6], # yes
        [2, 6], # no
        [1, 5], # yes
        [1, 6] # yes
    ]
    y_train_inclass_example = ["yes", "yes", "no", "no", "yes", "no", "yes", "yes"]

    myNaive.fit(X_train_inclass_example, y_train_inclass_example)

    assert myNaive.priors == {'yes': 5/8, 'no': 3/8} # 5 "yes"'s and 3 "no"'s out of 8 rows
    
    expected_conditionals = {'yes': [{1: 4/5, 2: 1/5}, {5: 2/5, 6: 3/5}], 'no': [{1: 2/3, 2: 1/3}, {5: 2/3, 6: 1/3}]}
    for key in myNaive.conditionals: # for 'yes': [{1: 4/5, 2: 1/5}, {5: 2/5, 6: 3/5}]
        for index, column in enumerate(myNaive.conditionals[key]): # for 0, {1: 4/5, 2: 1/5}
            for pred_key, value in column.items(): # for 1, 4/5
                assert value ==  expected_conditionals[key][index][pred_key] # 4/5

    # LA7 (fake) iPhone purchases dataset
    header_iphone = ["standing", "job_status", "credit_rating"]
    X_train_iphone = [
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
    y_train_iphone = ["no", "no", "yes", "yes", "yes", "no", "yes", "no", "yes", "yes", "yes", "yes", "yes", "no", "yes"]
    myNaive.fit(X_train_iphone, y_train_iphone)

    expected_priors = {'no': 5/15, 'yes': 10/15}

    for key in myNaive.priors:
        assert myNaive.priors[key] == expected_priors[key]

    expected_conditionals = {'yes': [{1: 2/10, 2: 8/10}, {1: 3/10, 2: 4/10, 3: 3/10}, {'fair': 7/10, 'excellent': 3/10}], 
                             'no': [{1: 3/5, 2: 2/5}, {1: 1/5, 2: 2/5, 3: 2/5}, {'fair': 2/5, 'excellent': 3/5}]}
    
    for key in myNaive.conditionals:
        for index, column in enumerate(myNaive.conditionals[key]):
            for pred_key, value in column.items():
                assert value ==  expected_conditionals[key][index][pred_key]

    # Bramer 3.2 train dataset
    header_train = ["day", "season", "wind", "rain"]
    X_train_train = [
        ["weekday", "spring", "none", "none"], # on time
        ["weekday", "winter", "none", "slight"], # on time
        ["weekday", "winter", "none", "slight"], # on time
        ["weekday", "winter", "high", "heavy"], # late
        ["saturday", "summer", "normal", "none"], # on time
        ["weekday", "autumn", "normal", "none"], # very late
        ["holiday", "summer", "high", "slight"], # on time
        ["sunday", "summer", "normal", "none"], # on time
        ["weekday", "winter", "high", "heavy"], # very late
        ["weekday", "summer", "none", "slight"], # on time
        ["saturday", "spring", "high", "heavy"], # cancelled
        ["weekday", "summer", "high", "slight"], # on time
        ["saturday", "winter", "normal", "none"], # late
        ["weekday", "summer", "high", "none"], # on time
        ["weekday", "winter", "normal", "heavy"], # very late
        ["saturday", "autumn", "high", "slight"], # on time
        ["weekday", "autumn", "none", "heavy"], # on time
        ["holiday", "spring", "normal", "slight"], # on time
        ["weekday", "spring", "normal", "none"], # on time
        ["weekday", "spring", "normal", "slight"] # on time
    ]
    y_train_train = ["on time", "on time", "on time", "late", "on time", "very late", "on time",
                    "on time", "very late", "on time", "cancelled", "on time", "late", "on time",
                    "very late", "on time", "on time", "on time", "on time", "on time"]
    
    myNaive.fit(X_train_train, y_train_train)

    expected_priors = {'on time': 14/20, 'late': 2/20, 'very late': 3/20, 'cancelled': 1/20}
    for key in myNaive.priors:
        assert myNaive.priors[key] == expected_priors[key]

    expected_conditionals = {'on time': [{'weekday': 9/14, 'saturday': 2/14, 'holiday': 2/14, 'sunday': 1/14},
                                       {'spring': 4/14, 'winter': 2/14, 'autumn': 2/14, 'summer': 6/14}, 
                                       {'none': 5/14, 'high': 4/14, 'normal': 5/14},
                                       {'none': 5/14, 'slight': 8/14, 'heavy': 1/14}],
                            'late': [{'weekday': 1/2, 'saturday': 1/2, 'holiday': 0/2, 'sunday': 0/2},
                                       {'spring': 0/2, 'winter': 2/2, 'autumn': 0/2, 'summer': 0/2}, 
                                       {'none': 0/2, 'high': 1/2, 'normal': 1/2},
                                       {'none': 1/2, 'slight': 0/2, 'heavy': 1/2}],
                            'very late': [{'weekday': 3/3, 'saturday': 0/3, 'holiday': 0/3, 'sunday': 0/3},
                                       {'spring': 0/3, 'winter': 2/3, 'autumn': 1/3, 'summer': 0/3}, 
                                       {'none': 0/3, 'high': 1/3, 'normal': 2/3},
                                       {'none': 1/3, 'slight': 0/3, 'heavy': 2/3}],
                            'cancelled': [{'weekday': 0/1, 'saturday': 1/1, 'holiday': 0/1, 'sunday': 0/1},
                                       {'spring': 1/1, 'winter': 0/1, 'autumn': 0/1, 'summer': 0/1}, 
                                       {'none': 0/1, 'high': 1/1, 'normal': 0/1},
                                       {'none': 0/1, 'slight': 0/1, 'heavy': 1/1}]}

    for key in myNaive.conditionals:
        for index, column in enumerate(myNaive.conditionals[key]):
            for pred_key, value in column.items():
                assert value ==  expected_conditionals[key][index][pred_key]
    
    
def test_naive_bayes_classifier_predict():
    # in-class Naive Bayes example (lab task #1)
    header_inclass_example = ["att1", "att2"]
    X_train_inclass_example = [
        [1, 5], # yes
        [2, 6], # yes
        [1, 5], # no
        [1, 5], # no
        [1, 6], # yes
        [2, 6], # no
        [1, 5], # yes
        [1, 6] # yes
    ]
    y_train_inclass_example = ["yes", "yes", "no", "no", "yes", "no", "yes", "yes"]

    myNaive = MyNaiveBayesClassifier()
    myNaive.fit(X_train_inclass_example, y_train_inclass_example)
    X_test = [[1, 5]]

    # yes: P(yes) * P(1 | yes) * P(5 | yes) ==> (5/8)(4/5)(2/5) = 0.2
    # no: P(no) * (1 | no) * P(5 | no) ==> (3/8)(2/3)(2/3) = 0.17
    # since yes > no --> class = yes
    assert myNaive.predict(X_test) == ["yes"] 

    header_iphone = ["standing", "job_status", "credit_rating"]
    X_train_iphone = [
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
    y_train_iphone = ["no", "no", "yes", "yes", "yes", "no", "yes", "no", "yes", "yes", "yes", "yes", "yes", "no", "yes"]
    myNaive.fit(X_train_iphone, y_train_iphone)
    
    X_test = [[2, 2, "fair"], [1, 1, "excellent"]]
    
    # For [2, 2, "fair"]:
    #       yes: (10/15)(8/10)(4/10)(7/10) = 0.149 <--
    #       no: (5/15)(2/5)(2/5)(2/5) = 0.021
    # For [1, 1, "excellent"]:
    #       yes: (10/15)(2/10)(3/10)(3/10) = 0.012 
    #       no: (5/15)(3/5)(1/5)(3/5) = 0.024 <--
    assert myNaive.predict(X_test) == ["yes", "no"]

    header_train = ["day", "season", "wind", "rain"]
    X_train_train = [
        ["weekday", "spring", "none", "none"], # on time
        ["weekday", "winter", "none", "slight"], # on time
        ["weekday", "winter", "none", "slight"], # on time
        ["weekday", "winter", "high", "heavy"], # late
        ["saturday", "summer", "normal", "none"], # on time
        ["weekday", "autumn", "normal", "none"], # very late
        ["holiday", "summer", "high", "slight"], # on time
        ["sunday", "summer", "normal", "none"], # on time
        ["weekday", "winter", "high", "heavy"], # very late
        ["weekday", "summer", "none", "slight"], # on time
        ["saturday", "spring", "high", "heavy"], # cancelled
        ["weekday", "summer", "high", "slight"], # on time
        ["saturday", "winter", "normal", "none"], # late
        ["weekday", "summer", "high", "none"], # on time
        ["weekday", "winter", "normal", "heavy"], # very late
        ["saturday", "autumn", "high", "slight"], # on time
        ["weekday", "autumn", "none", "heavy"], # on time
        ["holiday", "spring", "normal", "slight"], # on time
        ["weekday", "spring", "normal", "none"], # on time
        ["weekday", "spring", "normal", "slight"] # on time
    ]
    y_train_train = ["on time", "on time", "on time", "late", "on time", "very late", "on time",
                    "on time", "very late", "on time", "cancelled", "on time", "late", "on time",
                    "very late", "on time", "on time", "on time", "on time", "on time"]
    
    myNaive.fit(X_train_train, y_train_train)
    X_test = [["weekday", "winter", "high", "heavy"], 
              ["weekday", "summer", "high", "heavy"],
              ["sunday", "summer", "normal", "slight"]]
    assert myNaive.predict(X_test) == ["very late", "on time", "on time"]
