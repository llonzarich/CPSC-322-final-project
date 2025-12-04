import numpy as np

from mysklearn.myclassifiers import MyDecisionTreeClassifier

def test_decision_tree_classifier_fit():
    '''
        Purpose: test for decision tree classifier model training function.

        Notes: - the attribute values are sorted alphabetically
               - fit() doesn't (and shouldn't) accept attribute names, so use generic "att#" labels

    '''
    # ==========================================================================================================================================================
    # test case 1: use the 14 instance "interview" training set example, asserting against the tree constructed in 'B Attribute Selection (Entropy) Lab Task #1'
    # ==========================================================================================================================================================
    print("TEST CASE 1")
    header_interview = ["level", "lang", "tweets", "phd", "interviewed_well"]
    
    header_interview = ["att0", "att1", "att2", "att3"]
    attribute_domains = {"att0": ["Junior", "Mid", "Senior"], # att0 = "level"
                        "att1": ["Java", "Python", "R"],      # att1 = "lang"
                        "att2": ["no", "yes"],                # att2 = "tweets"
                        "att3": ["no", "yes"]}                # att3 = "phd"
    
    X_train_interview = [
        ["Senior", "Java", "no", "no"],
        ["Senior", "Java", "no", "yes"],
        ["Mid", "Python", "no", "no"],
        ["Junior", "Python", "no", "no"],
        ["Junior", "R", "yes", "no"],
        ["Junior", "R", "yes", "yes"],
        ["Mid", "R", "yes", "yes"],
        ["Senior", "Python", "no", "no"],
        ["Senior", "R", "yes", "no"],
        ["Junior", "Python", "yes", "no"],
        ["Senior", "Python", "yes", "yes"],
        ["Mid", "Python", "no", "yes"],
        ["Mid", "Java", "yes", "no"],
        ["Junior", "Python", "no", "yes"]
    ]
    y_train_interview = ["False", "False", "True", "True", "True", "False", "True", "False", "True", "True", "True", "True", "True", "False"]

    tree_interview = \
            ["Attribute", "att0",   # split attribute = level (14)
                ["Value", "Junior",     # level = junior 
                    ["Attribute", "att3",    # split attribute = phd (5)
                        ["Value", "no",           # phd = no
                            ["Leaf", "True", 3, 5]   # probability 3/5 (3 instances have no: 3 True, 0 False)
                        ],
                        ["Value", "yes",          # phd = yes
                            ["Leaf", "False", 2, 5]  # probability 2/5 (2 instances have yes: 0 True, 2 False)
                        ]
                    ]
                ],
                ["Value", "Mid",
                    ["Leaf", "True", 4, 14]
                ],
                ["Value", "Senior",    # level = senior
                    ["Attribute", "att2",   # split attribute = tweets (5)
                        ["Value", "no",         # tweets = no
                            ["Leaf", "False", 3, 5]   # probability 3/5 (3 instances have no: 0 True, 3 False)
                        ],
                        ["Value", "yes",        # tweets = yes
                            ["Leaf", "True", 2, 5]  # probability 2/5 (2 instances have yes: 2 True, 0 False)
                        ]
                    ]
                ]
            ]
    
    # train the decision tree model.
    dt_clf1 = MyDecisionTreeClassifier()
    dt_clf1.fit(X_train_interview, y_train_interview)

    # retrieve the tree from my fit method using the MyDecisionTree attribute 'tree'.
    tree1 = dt_clf1.tree

    assert dt_clf1.tree is not None
    assert tree1 == tree_interview


    # ==========================================================================================================================================================
    # test case 2: use the 15-instance iPhone training set example from LA7, asserting against the tree created with a desk check
    # ==========================================================================================================================================================
    print("TEST CASE 2")
    header_iphone = ["standing", "job_status", "credit_rating", "buys_iphone"]
    header_iphone = ["att0", "att1", "att2"]
    
    attribute_domains = {"att0": [1, 2],                 # att0 = standing
                         "att1": [1, 2, 3],              # att1 = job_status
                         "att2": ["excellent", "fair"]}  # att2 = credit_rating
    
    X_train_iphone = [
        [1, 3, "fair"],        # no
        [1, 3, "excellent"],   # no
        [2, 3, "fair"],        # yes
        [2, 2, "fair"],        # yes
        [2, 1, "fair"],        # yes
        [2, 1, "excellent"],   # no
        [2, 1, "excellent"],   # yes
        [1, 2, "fair"],        # no
        [1, 1, "fair"],        # yes
        [2, 2, "fair"],        # yes
        [1, 2, "excellent"],   # yes
        [2, 2, "excellent"],   # yes
        [2, 3, "fair"],        # yes
        [2, 2, "excellent"],   # no
        [2, 3, "fair"]         # yes
    ]
    y_train_iphone = ["no", "no", "yes", "yes", "yes", "no", "yes", "no", "yes", "yes", "yes", "yes", "yes", "no", "yes"]

    tree_iphone = \
        ["Attribute", "att0",       # attribute = standing (10)
            ["Value", 1,                # standing = 1 (15)
                ["Attribute", "att1",       # attribute = job_status (5)
                    ["Value", 1,                # job_status = 1 
                        ["Leaf", "yes", 1, 5]      # prob = 1/5 (1 yes, 0 no out of 5 instances)
                    ],
                    ["Value", 2,                # job status = 2
                        ["Attribute", "att2",      # attribute = credit_rating (2 instances)
                            ["Value", "excellent",     # credit_rating = excellent
                                ["Leaf", "yes", 1, 2]      # prob = 1/2 (1 yes, 0 no out of 2 instances)
                            ],
                            ["Value", "fair",          # credit_rating = fair
                                ["Leaf", "no", 1, 2]       # prob = 1/2 (0 yes, 1 no out of 2 instances)
                            ]
                        ]
                    ], 
                    ["Value", 3,                # job status = 3
                        ["Leaf", "no", 2, 5]       # prob = 2/5 (0 yes, 2 no out of 5 instances)
                    ]             
                ] 
            ],
            ["Value", 2,                 # standing = 2 (15)
                ["Attribute", "att2",       # attribute = credit_rating (10)
                    ["Value", "excellent",     # credit_rating = excellent
                        ["Leaf", "no", 4, 10]  # probability = 2/10 (there are 4 excellent instances: 2 yes, 2 no ==> we would split using another attribute after 'excellent' but one of the partitions for the next split attribute is empty, so we remove that branch and make a leaf node here instead)
                    ],
                    ["Value", "fair",         # credit_rating = fair 
                        ["Leaf", "yes", 6, 10]   # probability = 6/10 (6 yes, 0 no out of 10 instances)
                    ]
                ]
            ]
        ]


    # train a decision tree classifier object.
    dt_clf2 = MyDecisionTreeClassifier()
    dt_clf2.fit(X_train_iphone, y_train_iphone)

    tree2 = dt_clf2.tree

    assert dt_clf2.tree is not None
    assert tree2 == tree_iphone



def test_decision_tree_classifier_predict():
    '''
        Purpose: test for MyDecisionTreeClassifier predict method.
    '''
    # ==========================================================================================================================================================
    # test case 1: use the 14 instance "interview" training set example, asserting against the tree constructed in 'B Attribute Selection (Entropy) Lab Task #1'
    # ==========================================================================================================================================================
    header_interview = ["level", "lang", "tweets", "phd", "interviewed_well"]
    X_train_interview = [
        ["Senior", "Java", "no", "no"],
        ["Senior", "Java", "no", "yes"],
        ["Mid", "Python", "no", "no"],
        ["Junior", "Python", "no", "no"],
        ["Junior", "R", "yes", "no"],
        ["Junior", "R", "yes", "yes"],
        ["Mid", "R", "yes", "yes"],
        ["Senior", "Python", "no", "no"],
        ["Senior", "R", "yes", "no"],
        ["Junior", "Python", "yes", "no"],
        ["Senior", "Python", "yes", "yes"],
        ["Mid", "Python", "no", "yes"],
        ["Mid", "Java", "yes", "no"],
        ["Junior", "Python", "no", "yes"]
    ]
    y_train_interview = ["False", "False", "True", "True", "True", "False", "True", "False", "True", "True", "True", "True", "True", "False"]
    X_test_interview = [
        ["Junior", "Java", "yes", "no"],
        ["Junior", "Java", "yes", "yes"]
    ]
    y_test_interview = ["True", "False"]

    dt_clf1 = MyDecisionTreeClassifier()
    dt_clf1.fit(X_train_interview, y_train_interview)

    y_pred_interview = dt_clf1.predict(X_test_interview)

    assert y_test_interview == y_pred_interview


    # ==========================================================================================================================================================
    # test case 2: use the 15-instance iPhone training set example from LA7, asserting against the tree created with a desk check
    # ==========================================================================================================================================================
    header_iphone = ["standing", "job_status", "credit_rating", "buys_iphone"]
    header_iphone = ["att0", "att1", "att2"]
    
    attribute_domains = {"att0": [1, 2],                 # att0 = standing
                         "att1": [1, 2, 3],              # att1 = job_status
                         "att2": ["excellent", "fair"]}  # att2 = credit_rating
    
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
    X_test_iphone = [
        [2, 2, "fair"],
        [1, 1, "excellent"]
    ]
    y_test_iphone = ["yes", "yes"]

    dt_clf2 = MyDecisionTreeClassifier()
    dt_clf2.fit(X_train_iphone, y_train_iphone)
    
    y_pred_iphone = dt_clf2.predict(X_test_iphone)

    assert y_test_iphone == y_pred_iphone