#!/usr/bin/python
# -*- coding: utf-8 -*-

'''
Load and train different classifiers


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
    def __init__(self, usePCA = True):
        '''
        Initialize a MLMethods instance
        '''
        self.d_clf = {
            "DecisionTree" : Classifier("DecisionTree", usePCA = usePCA), 
            "KNeighbors" : Classifier("KNeighbors", usePCA = usePCA),
            "SVM" : Classifier("SVM", usePCA = usePCA),
            "AdaBoost" : Classifier("AdaBoost", usePCA = usePCA),
            "GaussianNB" : Classifier("GaussianNB", usePCA = usePCA),
            "RandomForest" : Classifier("RandomForest", usePCA = usePCA)
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

    def get_pcaComponents(self, s_ml):
        '''
        Return the PCs with the coefficients for each original feature. It is a 
        n_components by n_features array
        s_ml: string with the name of the algorithm desired
        source: https://discussions.udacity.com/t/interpreting-the-number-of
        -components-kept-in-pca/25574
        '''
        pipeline  = self.d_clf[s_ml].clf
        my_pca = pipeline.steps[0][1]
        return my_pca.components_        

class Classifier:
    '''
    Train and test different classifiers and show some summaries about its 
    performance
    '''
    def __init__(self, s_classifier, usePCA = True):
        '''
        Initialize a Classifier instance and save all parameters as attributes
        '''
        self.name = s_classifier
        if usePCA:
            estimators = [('reduce_dim', PCA()),
                          (s_classifier, d_clf[s_classifier]())]
        else:
            estimators = [(s_classifier, d_clf[s_classifier]())]            
        self.clf = Pipeline(estimators)
        self.already_tuned = False
        self.d_performance = None

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

    def crossValidation(self, features, labels, report = True, l_columns= None): 
        '''
        Cross validate the data set passed
        features: numpy array with the features to be used to test models
        labels: numpy array with the real output
        l_columns: list of labels names for a condensed datframe report
        '''
        #keep the last data
        if self.d_performance: d_last= self.d_performance.copy()
        else: d_last = None


        d_rtn, s_rtn = validation.test_classifier(self.clf, features, labels);
        # d_rtn = validation.test_classifier(self.clf, features, labels);
        self.d_performance = d_rtn
        self.cv_report = s_rtn

        #print a report
        if report:
            self.reportCrossValidation()

        if l_columns:
            #creates a dataframe with the currently accuracy,precision,recall 
            df_rtn = pd.Series(d_rtn).map(lambda x: '%2.4f' % x)
            df_rtn = pd.DataFrame(df_rtn)
            df_rtn.columns = [l_columns[len(l_columns)-1]]
            df_rtn = df_rtn.ix[["accuracy","precision","recall"],:]
            #creates a dataframe with the last accuracy,precision,recall
            if d_last:
                df_rtn2 = pd.Series(d_last).map(lambda x: '%2.4f' % x)
                df_rtn2 = pd.DataFrame(df_rtn2)
                df_rtn2.columns = [l_columns[0]]
                df_rtn2 = df_rtn2.ix[["accuracy","precision","recall"],:]
                #create a data frame with 2 columns
                df_rtn[l_columns[0]] = None
                df_rtn[l_columns[0]] = list(df_rtn2[l_columns[0]].values)
                df_rtn = df_rtn.ix[:,l_columns]
            #return the datafarme
            return df_rtn


    def reportCrossValidation(self):
        '''
        print the results of the crossValidation method
        '''
        if self.already_tuned : print "!!!!ALREADY TUNED"
        print self.cv_report





