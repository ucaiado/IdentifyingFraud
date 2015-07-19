#!/usr/bin/python
# -*- coding: utf-8 -*-

'''
Load and manage the data to be used in the project


Created on 07/09/2015
'''

__author__='ucaiado'


import pandas as pd
import numpy as np
import sys
import pickle
import numpy as np
from sklearn.preprocessing import Imputer
sys.path.append("tools/")

'''
Begin of Help Functions
'''


'''
End of Help Functions
'''

class LoadEnron:
    '''
    Load and handle the data
    '''
    def __init__(self):
        '''
        Initialize a DataSet instance
        '''
        self._loadData()
        self.payments_features = ['bonus', 'deferral_payments', 
        'deferred_income', 'director_fees', 'expenses','loan_advances', 
        'long_term_incentive', 'other','salary']
        self.stock_features = ['exercised_stock_options','restricted_stock', 
        'restricted_stock_deferred']
        self.email_features = ['from_messages', 'from_poi_to_this_person',
        'from_this_person_to_poi','shared_receipt_with_poi', 'to_messages']
        self.new_features = ['biggest_expenses', 'percentual_exercised']      

    def _loadData(self):
        '''
        Return a dataframe with all data that will be used 
        '''
        data_dict = pickle.load(open("final_project_dataset.pkl", "r") )
        df = pd.DataFrame(data_dict).T
        self.df = df
        self.df_scaled = []

    def getData(self, scaled = False):
        '''
        get a copy of the data set from this instance
        scaled: boolean. Get scaled or non scaled dataframe
        '''
        if scaled: return self.df_scaled.copy()
        else: return self.df.copy()

    def getValue(self, key, column, scaled = False):
        '''
        get the value of a specific datapoint, given the key and a column's name
        '''
        if scaled: 
            return self.df_scaled.loc[key, column]
        else: 
            return self.df.loc[key, column]

    def setData(self, df):
        '''
        Set a new data set to this instance
        df:  dataframe to replace the df attribute
        '''
        self.df = df

    def excludeOutliers(self, l_outliers):
        '''
        Exclude the data points in l_outliers
        '''
        df = self.getData()
        df.drop(l_outliers, inplace= True)
        self.setData(df)

    def fill_and_remove(self, s_strategy="zeros", l_features = False, 
        b_remove = True):
        '''
        fill all Nan values in numerical data with zeros and then remove data 
        points that all features are equal to zero
        l_features: a list of features to be tested. If any, all features will 
        be used
        b_remove: boolean indicating if should remove keys where all data is 0
        s_strategy: string with the strategy used to fill NaNs. Can be "mean",
        "median" and "zeros"
        '''
        df = self.getData()
        #pre-process data
        if not l_features:
            l_features = self.payments_features + self.stock_features 
            l_features+= self.email_features
        df.loc[:, l_features] = df.loc[:, l_features].astype(float)
        #filling Nan with the strategy selected
        if s_strategy == "zeros":
            df.loc[:, l_features] = df.loc[:, l_features].fillna(0)
        else:
            na_X = df.loc[:, l_features].values
            imp = Imputer(missing_values='NaN', strategy=s_strategy, axis=0)
            df.loc[:, l_features] = imp.fit_transform(na_X)

        #exclude datapoint where every number is equal to 0
        if b_remove:
            df = df.ix[((df.loc[:, l_features]!=0).sum(axis=1)!=0),:]
        #saving the new dataframe       
        self.setData(df)
        #correct scaled df
        if type(self.df_scaled)!=list:
            df2 = self.df_scaled
            df2 = df2.ix[((df.loc[:, l_features]!=0).sum(axis=1)!=0).index,:]
            self.df_scaled = df2             