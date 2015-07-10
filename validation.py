#!/usr/bin/python
# -*- coding: utf-8 -*-

'''
An adapted version of test_classifier function from tester.py


Created on 07/09/2015
'''

__author__='ucaiado'



#load dependencies
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.cross_validation import StratifiedShuffleSplit

#create string to print the results
s = "\nClassifier Used\n------------------\n{}\n"
s+= "\nClassification Report\n------------------\n"
s+= "Accuracy: {:>0.{display_precision}f}\tPrecision: {:>0.{display_precision}f}"
s+= "\tRecall: {:>0.{display_precision}f}\tF1: {:>0.{display_precision}f}\t"
s+= "F2: {:>0.{display_precision}f}\n"
PERF_FORMAT_STRING = s

s = "Confusion Matrix\n------------------\nTotal predictions: {:4d}\t"
s+= "True positives: {:4d}\tFalse positives: {:4d}\n\t\t\t\tFalse negatives:"
s+= " {:4d}\tTrue negatives: {:4d}\n"
RESULTS_FORMAT_STRING = s
#create function
def test_classifier(clf, features, labels, folds = 1000):
    '''
    Return a dictionary and print the measurements of precision-recall of the 
    Enron data using stratified shuffle split  cross validation due to the small
    the size of the dataset
    '''
    cv = StratifiedShuffleSplit(labels, folds, random_state = 42)
    true_negatives = 0
    false_negatives = 0
    true_positives = 0
    false_positives = 0
    for train_idx, test_idx in cv: 
        features_train = []
        features_test  = []
        labels_train   = []
        labels_test    = []
        for ii in train_idx:
            features_train.append( features[ii] )
            labels_train.append( labels[ii] )
        for jj in test_idx:
            features_test.append( features[jj] )
            labels_test.append( labels[jj] )
        
        ### fit the classifier using training set, and test on test set
        clf.fit(features_train, labels_train)
        predictions = clf.predict(features_test)
        for prediction, truth in zip(predictions, labels_test):
            if prediction == 0 and truth == 0:
                true_negatives += 1
            elif prediction == 0 and truth == 1:
                false_negatives += 1
            elif prediction == 1 and truth == 0:
                false_positives += 1
            else:
                true_positives += 1
    try:
        total_predictions = true_negatives + false_negatives + false_positives 
        total_predictions+= true_positives
        accuracy = 1.0*(true_positives + true_negatives)/total_predictions
        precision = 1.0*true_positives/(true_positives+false_positives)
        recall = 1.0*true_positives/(true_positives+false_negatives)
        f_total = (2*true_positives + false_positives+false_negatives)
        f1 = 2.0 * true_positives/f_total
        f2 = (1+2.0*2.0) * precision*recall/(4*precision + recall)
        
        d_rtn = {"accuracy": accuracy, "precision": precision, "f1":f1, "f2":f2,
                 "total_predictions": total_predictions, 
                 "true_positives": true_positives,
                 "false_positives": false_positives,
                 "false_negatives": false_negatives,
                 "true_negatives": true_negatives
                }
        
        print PERF_FORMAT_STRING.format(clf, accuracy, precision, recall, f1, f2, display_precision = 5)
        print RESULTS_FORMAT_STRING.format(total_predictions, true_positives, false_positives, false_negatives, true_negatives)
        print ""

        return d_rtn
    except:
        print "Got a divide by zero when trying out:", clf