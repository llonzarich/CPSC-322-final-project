# Confidence Level Detection

## Project Description
This project implements three different machine learning algorithms to classify a person's confidence level as 'confident', 'neutral', or 'low' from 9 total attributes. There are 6 continuous numerical attributes, representing the individual's body features, and 3 categorical attributes, representing the individual's body posture.

Exploratory Data Analysis is performed on our dataset to understand the dataset structure and identify characteristics that could be useful for improving classifier performance.


</br>


# How to Run 
Everything that is needed to run this project lives inside our final report Jupyter notebook (named 'Final_Report.ipynb'). Simply run each sequential cell in the notebook to step through our data story.


</br>



# Project Organization
In this repository, you will find all files to perform confidence level classification using...
* A decision tree
  * Invokes the Top Down Induction for Decision Trees (TDIDT) algorithm.
* A random forest
  * Trains 20 trees using the TDIDT algorithm on bootstrapped samples of the training data.
  * Evaluate the accuracy of each tree and retains the 7 most acccurate trees, which form the final ensemble (the random forest).
  * Predictions are generated using majority voting among the 7 decision trees in the forest.
- Naive Bayes

The mysklearn folder contains the majority of the code necessary for this project:
* myclassifiers.py: 
  * MyDecisionTreeClassifier class 
  * MyRandomForestClassifier class
  * MyNaiveBayesClassifier class
* myevaluation.py
  * train_test_split() to generate train and test sets from the dataset
  * kfold_split() to split the dataset into cross validation folds
  * stratified_kfold_split() to split the dataset into stratified cross validation folds
  * confusion_matrix()
  * accuracy_score()
  * binary_precision_score()
  * multiclass_precision_score()
  * binary_recall_score()
  * mutliclass_recall_score()
  * binary_f1_score()
  * multiclass_f1_score()
  * bootstrap_sample() to split the dataset into bootstrapped training set and out-of-bag test set
* mypytable.py
  * MyPyTable class, which contains methods for handling a 2D table of data with column names
* myutils.py
  * get_frequency() to find all unique values in a given column
  * all_same_values() to check whether all class labels for a given list of instances are the same
  * attribute_to_split() to find the attribute with the lowest entropy to split on at a given node.
  * calculate_entropy()
  * partition_data() to partition given instances into "bins" based on their attribute values
  * get_col()
  * most_freq_class() to find the most frequently occuring class label
  * my_discretizer() to discretize continuous values to categorical values
  * cross_val_predict() to compute the k-fold cross validation and evaluate model performance on each fold

In addition to the mysklearn folder, this repository contains the following:
* Final_report.ipynb
  * The Jupyter notebook used to run our project 
  * Introduces our project and dataset, implements data analysis (e.g., EDA, data preprocessing, visuals showing interesting insight), implements classifiers, evaluates classifier performance, summarizes results and key findings, and concludes with a list of the sources used.
* Project_Proposal.ipynb
  * The Jupyter notebook which outlines our project proposal, including an introduction to our project, goals, and an analysis on the significance of this project.
* confidence_features.csv
  * our dataset
* test_mydecisiontree.py
  * Contains unit tests for the decision tree class
* test_myevaluation.py
  * Contains unit tests for the functions in myevaluation.py
* test_mynaivebayes.py
  * Contains unit tests for the Naive Bayes class
* test_myrandomforest.py
  * Contains unit tests for the random forest class