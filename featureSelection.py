#!/usr/bin/python
# -*- coding: utf-8 -*-

'''
Select features to be used by ML algorithms


Created on 07/09/2015
'''

__author__='ucaiado'

import pandas as pd
import numpy as np
import sys
from sklearn.feature_selection import SelectPercentile, f_classif

'''
Begin of Help Functions
'''


def selectFeatures(features, labels, features_list):
    '''
    Select features according to the 20th percentile of the highest scores. 
    Return a list of features selected  and a dataframe showing the ranking 
    of each feature related to their p values
    features: numpy array with the features to be used to test sklearn models
    labels: numpy array with the real output 
    features_list: a list of names of each feature
    '''
    #feature selection
    selector = SelectPercentile(f_classif, percentile=20)
    selector.fit(features, labels)
    features_transformed = selector.transform(features)
    #filter names to be returned
    l_rtn = [x for x, t in zip(features_list, 
        list(selector.get_support())) if t]
    # pd.DataFrame(features_transformed, columns = l_labels2).head()
    #calculate scores
    scores = -np.log10(selector.pvalues_)
    scores /= scores.max()
    df_rtn = pd.DataFrame(pd.Series(dict(zip(features_list,scores))))
    df_rtn.columns = ["pValue_Max"]
    df_rtn = df_rtn.sort("pValue_Max", ascending=False)
    # df_rtn["different_from_zero"]=((df!=0).sum()*1./df.shape[0])


    return l_rtn, df_rtn

'''
End of Help Functions
'''


class Features(object):
    '''
    Test features and create new ones
    '''
    def __init__(self):
        '''
        Initialize a Features instance
        '''
        self.payments_features = ['bonus', 'deferral_payments', 
        'deferred_income', 'director_fees', 'expenses','loan_advances', 
        'long_term_incentive', 'other','salary']
        self.stock_features = ['exercised_stock_options','restricted_stock', 
        'restricted_stock_deferred']
        self.email_features = ['from_messages', 'from_poi_to_this_person',
        'from_this_person_to_poi','shared_receipt_with_poi', 'to_messages']
        self.new_features = ['biggest_expenses', 'percentual_exercised']
       
    def getFeaturesList(self, o_dataset, o_eda, f_validNumMin = 0.):
        '''
        Return a list of columns names from the self data
        f_validNumMin: float with the minimum percentual of valid numbers in 
        each feature to be tested
        o_dataset: an object with the dataset loaded
        o_eda: an object with the eda methods
        '''
        l_columns = self.payments_features + self.stock_features 
        l_columns+=  self.email_features + self.new_features
        df_rtn = o_eda.notValidNumbersTable(o_dataset)
        na_exclude = (df_rtn.T<f_validNumMin).values
        l_exclude = list(df_rtn.loc[list(na_exclude)[0]].index)        
        l_rtn = [ x for x in l_columns if x not in l_exclude]        

        return l_rtn

    def getFeaturesAndLabels(self, o_dataset,o_eda = None, l_columns = False, 
        scaled = False, f_validNumMin = 0.):
        '''
        Return two nuumpy arrays with labels and features splitted
        scaled: boolean. should return scaled features?
        f_validNumMin: float with the minimum percentual of a valid number from
        a feature to be tested
        l_columns: target features to be filtered. If any, use all.
        o_dataset: an object with the dataset loaded
        '''
        #load data needed
        df = o_dataset.getData(scaled = scaled)
        if not l_columns:
            l_columns = self.getFeaturesList(o_dataset, o_eda, f_validNumMin)
        #split data
        na_labels = df.poi.values.astype(np.float32)
        na_features = df.loc[:,l_columns].values.astype(np.float32)
        return na_labels, na_features          

    def createNewFeatures(self, o_dataset):
        '''
        create the features biggest_expenses and percentual_exercised. Save them
        as new columns in df attribute in o_dataset
        o_dataset: an object with the dataset loaded
        '''
        #get a copy of the data
        df = o_dataset.getData()
        #compare the expenses to the biggest one scaling it
        # f_min = df.expenses.astype(float).min()
        # f_max = df.expenses.astype(float).max() 
        # df_t2 = (df.expenses.astype(float) - f_min)/(f_max - f_min)
        df_aux = df.salary.astype(float)
        df_aux[df_aux==0]=None
        df_t2 = df.expenses.astype(float)/df_aux
        df_t2 = pd.DataFrame(df_t2)
        df_t2.columns = ["biggest_expenses"]
        # df_t2 = df_t2.fillna(df_t2.mean())
        df_t2["poi"]=df.poi
        # df_t2 = df_t2.fillna(0)
        #scale the new feature
        f_min = df_t2.min()
        f_max = df_t2.max()
        df_t2 = (df_t2-f_min)/ (f_max - f_min)
        #compare the exercised stock options to total payment
        df_aux = df.total_payments.astype(float)
        df_aux[df_aux==0]=None
        df_t3 = df.exercised_stock_options.astype(float)/df_aux
        df_t3 = pd.DataFrame(df_t3)
        #scale the new feature
        f_min = df_t3.min()
        f_max = df_t3.max()
        df_t3 = (df_t3-f_min)/ (f_max - f_min)
        # df_t3 = df_t3.fillna(df_t3.mean())
        # df_t3 = df_t3.fillna(0)
        #exclude some outliers just to this plot
        df_t3.columns = ["percentual_exercised"]
        df_t3["poi"]=df.poi
        #include the new features in the original dataset
        df['biggest_expenses'] = df_t2['biggest_expenses']
        df["percentual_exercised"] = df_t3["percentual_exercised"]
        o_dataset.setData(df)        

    def scallingAll(self, o_dataset):
        '''
        Scale each group of features, keep the result as an attribute 
        '''
        #load data
        df = o_dataset.getData()
        l_payment = self.payments_features
        l_stock = self.stock_features
        l_email = self.email_features
        #scale money related features
        df_aux = df.loc[:,l_payment + l_stock]
        f_max  = df_aux.max().max()
        f_min = df_aux.min().min()
        df_aux = (df_aux - f_min) * 1./(f_max - f_min)
        df.loc[:,l_payment + l_stock] =   df_aux.values
        #scale email features
        df_aux = df.loc[:,l_email ]
        f_max  = df_aux.max().max()
        f_min = df_aux.min().min()
        df_aux = (df_aux - f_min) * 1./(f_max - f_min)
        df.loc[:,l_email ] =   df_aux.values
        #keep results and show description
        o_dataset.df_scaled = df

    def select(self, features, labels, features_list):
        '''
        Select features using selectFeatures function. Return a list with the 
        features selected  and a p-values ranking.
        features: numpy array with the features to be used to test sklearn models
        labels: numpy array with the real output 
        features_list: a list of names of each feature
        '''
        l_rtn, df_rtn = selectFeatures(features, labels, features_list)
        return l_rtn, df_rtn 

        