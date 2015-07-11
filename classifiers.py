#!/usr/bin/python
# -*- coding: utf-8 -*-

'''
Load and different classifiers


Created on 07/09/2015
'''

__author__='ucaiado'



from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.grid_search import GridSearchCV
import validation
from pprint import pprint



d_clf = {
    "DecisionTree" : DecisionTreeClassifier, 
    "KNeighbors" : KNeighborsClassifier,
    "SVM" : SVC,
    "AdaBoost" : AdaBoostClassifier,
    "GaussianNB" : GaussianNB,
    "RandomForest" : RandomForestClassifier
}

d_params = None

'''
Begin of Help Functions
'''

class Params:
    '''
    Hold dictionaries of parameters to be tunned
    '''
    def __init__(self, s_classifier):
        '''
        Initialize a Params instance
        '''
        if s_classifier == "DecisionTree": 
            self.d = self._getDecisionTree()
        elif s_classifier == "KNeighbors": 
            self.d = self._getKNeighbors()
        elif s_classifier == "SVM": 
            self.d = self._getSVC()
        elif s_classifier == "AdaBoost": 
            self.d = self._getDAdaBoost()
        elif s_classifier == "GaussianNB": 
            self.d = self._getGaussianNB()
        elif s_classifier == "RandomForest": 
            self.d = self._getRandomForest()

    def getDict(self):
        '''
        Return a discionary with all parameters to be tested
        '''
        return self.d
        
    def _getDecisionTree(self):
        '''
        Return a dictionary of the parameters to be tunned for a decision tree 
        classifiers
        '''

        d_params = dict(reduce_dim__n_components=[1, 2, 3],
                        DecisionTree__max_depth  = [3,10,50])   


        return d_params

    def _getKNeighbors(self):
        '''
        Return a dictionary of the parameters to be tunned for a KNeighbors
        classifiers
        '''

        d_params = dict(reduce_dim__n_components=[1, 2, 3],
                        KNeighbors__n_neighbors=[2, 3, 4],
                        #manhatan, euclidian, Minkowski
                        KNeighbors__p  = [1,2,3])        

        return d_params

    def _getSVC(self):
        '''
        Return a dictionary of the parameters to be tunned for a SVM 
        classifiers
        '''
        d_params = dict(reduce_dim__n_components=[2, 3],
                        SVM__kernel=['poly', 'linear', 'rbf'],
                        SVM__degree = [2, 3, 4],
                        SVM__C = [0.5, 1.0, 1.5, 2.])
        return d_params

    def _getDAdaBoost(self):
        '''
        Return a dictionary of the parameters to be tunned for a AdaBoost 
        classifiers
        '''
        l_base_estimator = [DecisionTreeClassifier(min_samples_split=10),
                           DecisionTreeClassifier(min_samples_split=30),
                           DecisionTreeClassifier(min_samples_split=50)]
        d_params = dict(reduce_dim__n_components=[1, 2, 3],
                        AdaBoost__base_estimator=l_base_estimator,
                        AdaBoost__n_estimators  = [10, 20, 50, 100])


        return d_params

    def _getGaussianNB(self):
        '''
        Return a dictionary of the parameters to be tunned for a GaussianNB
        classifiers
        '''
        d_params = dict(reduce_dim__n_components=[1, 2, 3])

        return d_params

    def _getRandomForest(self):
        '''
        Return a dictionary of the parameters to be tunned for a RandomForest 
        classifiers
        '''
        # n_estimators
        # max_depth
        # min_samples_split
        d_params = dict(reduce_dim__n_components=[1, 2, 3],
                        RandomForest__max_depth  = [3,10,50],
                        RandomForest__n_estimators  = [10, 20, 50, 100])         

        return d_params


'''
End of Help Functions
'''

class Classifier:
    '''
    Train and test different classifiers and show some summaries about its 
    performance
    '''
    def __init__(self, s_classifier):
        '''
        Initialize a Classifier instance and save all parameters as attributes
        '''
        self.name = s_classifier
        estimators = [('reduce_dim', PCA()),
                      (s_classifier, d_clf[s_classifier]())]
        self.clf = Pipeline(estimators)
        self.already_tuned = False

    def getFeatureImportance():
        '''
        return an array with the feature importance when the classifier is 
        related to decision tree
        '''
        #as I am using PCA, it will return the weight of the PCA components 
        if self.name in ["DecisionTree", "RandomForest",  "AdaBoost"]:
            return self.clf.steps[1][1].feature_importances_ 


    def gridSearch(self, features, labels):
        '''
        Execute a grid search using all data set passed
        features: numpy array with the features to be used to test models
        labels: numpy array with the real output 
        '''
        #initialize the grid serach object and look for the best parameters
        d_params = Params(self.name).getDict()   
        grid_search = GridSearchCV(self.clf, param_grid=d_params)
        grid_search.fit(features, labels)
        #update the classifier with the best estimator
        self.clf = grid_search.best_estimator_
        self.already_tuned = True
        #print a summary
        print "Best Score: {0:.4f}".format(grid_search.best_score_)
        print "\nPARAMETERS TESTED\n------------------\n"
        pprint(d_params)
        print "\nBEST PARAMETER\n------------------\n"
        pprint(grid_search.best_params_)


    def crossValidation(self, features, labels): 
        '''
        Cross validate the data set passed
        features: numpy array with the features to be used to test models
        labels: numpy array with the real output 
        '''
        if self.already_tuned : print "!!!!ALREADY TUNNED"
        d_rtn = validation.test_classifier(self.clf, features, labels);
        self.d_performance = d_rtn





