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
import pandas as pd



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
    def __init__(self, s_classifier=""):
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
        else:
            self.d = None

    def printParameters(self):
        '''
        Print all parameters tested for each algo
        '''
        for s_algo in ["DecisionTree", "KNeighbors", "SVM", "AdaBoost", 
        "GaussianNB", "RandomForest"]:
            o_aux = Params(s_algo)
            print "\n{}\n----------".format(s_algo)
            d_aux = o_aux.getDict()
            if s_algo == "AdaBoost":
                l_aux = [z.get_params()['min_samples_split'] for z 
                in  d_aux['AdaBoost__base_estimator']]
                d_aux.pop('AdaBoost__base_estimator')
                d_aux['AdaBoost__base_estimator_min_samples_split'] = l_aux
            pprint(d_aux)

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
                        KNeighbors__n_neighbors=[2, 3, 4, 10, 30],
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

class MLMethods:
    '''
    Hold and train all the ML methods at once
    '''
    def __init__(self):
        '''
        Initialize a MLMethods instance
        '''
        self.d_clf = {
            "DecisionTree" : Classifier("DecisionTree"), 
            "KNeighbors" : Classifier("KNeighbors"),
            "SVM" : Classifier("SVM"),
            "AdaBoost" : Classifier("AdaBoost"),
            "GaussianNB" : Classifier("GaussianNB"),
            "RandomForest" : Classifier("RandomForest")
        }

    def gridSearchAll(self,features, labels):
        '''
        Execute a grid search for the best parameters for the algoritms
        features: numpy array with the features to be used to test models
        labels: numpy array with the real output        
        '''
        #gridserach all algos
        for key in self.d_clf:
            self.d_clf[key].gridSearch(features, labels, report = False)
        #get a dataframe with the best scores
        df = self.getSummary('GridSearch')
        return df

    def crossValidationAll(self,features, labels):
        '''
        Cross validate all ML methods at once
        features: numpy array with the features to be used to test models
        labels: numpy array with the real output
        '''
        #cross validates all algos
        for key in self.d_clf:
            self.d_clf[key].crossValidation(features, labels, report = False)
        #get a dataframe with the APR of each algo
        df = self.getSummary('crossValidation')
        return df

    def getReport(self, s_type, s_ml):
        '''
        get the report of the results for a specific steps and ML algoritms
        s_type: string with the name of step (crossValidation or GridSearch)
        s_ml: string with the name of the algorithm desired
        '''
        if s_type=='crossValidation':
            self.d_clf[s_ml]. reportCrossValidation()
        elif s_type=='GridSearch':
            self.d_clf[s_ml].reportGridSearch()

    def getSummary(self, s_type):
        '''
        get a summary for a particular step of all algoritms and return a 
        dataframe
        s_type: string with the name of step (crossValidation or GridSearch)
        '''
        l = []
        #get the best scores fot the tests made by grid search method
        if s_type=='GridSearch':
            for key in self.d_clf:
                f_rtn  = self.d_clf[key].gs_best_score
                if f_rtn:
                    d_rtn = {'BestScore':"{0:.4f}".format(f_rtn)}
                else:
                    d_rtn = {'BestScore': None}
                l.append(d_rtn)
        
        #get the APR of each ML tested
        elif s_type=='crossValidation':
            for key in self.d_clf:
                d_aux = self.d_clf[key].d_performance
                d_rtn = {'accuracy': None, 'precision': None,'recall':None}    
                if d_aux['accuracy']:
                    for key_2 in ['accuracy','precision','recall']:
                        d_rtn[key_2] = "{0:.4f}".format(d_aux[key_2])
                l.append(d_rtn)

        #make a dataframe with the names of each ML as an index
        df= pd.DataFrame(l)
        df.index = self.d_clf.keys()

        return df

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

    def getFeatureImportance(self):
        '''
        return an array with the feature importance when the classifier is 
        related to decision tree
        '''
        #as I am using PCA, it will return the weight of the PCA components 
        if self.name in ["DecisionTree", "RandomForest",  "AdaBoost"]:
            return self.clf.steps[1][1].feature_importances_ 


    def gridSearch(self, features, labels, report = True):
        '''
        Execute a grid search using all data set passed
        features: numpy array with the features to be used to test models
        labels: numpy array with the real output
        report: boolean indicating if should print report of results
        '''
        #initialize the grid serach object and look for the best parameters
        d_params = Params(self.name).getDict()   
        grid_search = GridSearchCV(self.clf, param_grid=d_params)
        grid_search.fit(features, labels)
        #update the classifier with the best estimator
        self.clf = grid_search.best_estimator_
        self.already_tuned = True
        #save all parameters as attributes
        self.gs_best_score = grid_search.best_score_
        self.gs_best_params = grid_search.best_params_
        #print a summary
        if report:
            self.reportGridSearch()

    def reportGridSearch(self):
        '''
        print the results of the GridSearchCV method
        '''
        d_params = Params(self.name).getDict() 
        print "Best Score: {0:.4f}".format(self.gs_best_score)
        print "\nPARAMETERS TESTED\n------------------"
        pprint(d_params)
        print "\nBEST PARAMETER\n------------------"
        pprint(self.gs_best_params)
        print "\n"

    def crossValidation(self, features, labels, report = True): 
        '''
        Cross validate the data set passed
        features: numpy array with the features to be used to test models
        labels: numpy array with the real output 
        '''
       
        d_rtn, s_rtn = validation.test_classifier(self.clf, features, labels);
        # d_rtn = validation.test_classifier(self.clf, features, labels);
        self.d_performance = d_rtn
        self.cv_report = s_rtn

        if report:
            self.reportCrossValidation()
        # self.d_performance = d_rtn
        # d_rtn, s_rtn

    def reportCrossValidation(self):
        '''
        print the results of the crossValidation method
        '''
        if self.already_tuned : print "!!!!ALREADY TUNED"
        print self.cv_report





