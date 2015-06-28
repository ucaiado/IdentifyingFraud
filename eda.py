#!/usr/bin/python
# -*- coding: utf-8 -*-

'''
Explore data that will be used in the project


Created on 25/06/2015
'''

__author__='ucaiado'


'''
Begin of Help Functions
'''


'''
End of Help Functions
'''


import pandas as pd
import numpy as np
import seaborn as sns
import sys
import pickle
sys.path.append("tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import test_classifier, dump_classifier_and_data

sns.set_palette("deep", desat=.6)
sns.set_context(rc={"figure.figsize": (8, 4)})


class Eda(object):
    '''
    Load data and summarize it executing exploratory data analysis
    '''
    def __init__(self):
        '''
        Initialize a Eda instance
        '''
        self._loadData()

    def _loadData(self):
        '''
        Return a dataframe with all data that will be used 
        '''
        data_dict = pickle.load(open("final_project_dataset.pkl", "r") )
        df = pd.DataFrame(data_dict).T
        self.df = df

    def getData(self):
        '''
        get a copy of the data set from this instance
        '''
        return self.df.copy()

    def setData(self, df):
        '''
        Set a new data set to this instance
        df:  dataframe to replace the df attribute
        '''
        self.df = df

    def NanAtFetaures(self):
        '''
        plot a bar plot with the percentual off Nan in each feature
        '''
        df = self.getData()
        ax  = ((df=="NaN").sum()/146.*100).plot(kind = "bar")
        ax.set_title("Percentual of Null Values in each Feature\n")

    def scatter(self, s_xFeature, s_yFeature):
        '''
        Plot a scatter plot with the labels passed
        s_xFeature: string with the column label to be used as the x axis
        s_yFeature: string with the column label to be used as the y axis
        '''

        df = self.getData()
        df_plot = df.ix[:, [s_xFeature, s_yFeature]]
        df_plot = df_plot[(df_plot=="NaN").sum(axis = 1)==0].astype(float)
        g = sns.jointplot(s_xFeature, s_yFeature, data=df_plot, kind="reg",
         size=6)

    def getDecile(self, s_feature, i_pos = -1):
        '''
        get the labels for the last decile in the data set, given a s_feature
        '''
        df = self.getData()
        df_t = df.ix[:, [s_feature]].astype("float")
        f_lastDecile = (df_t.quantile(np.arange(0.1,1.1,0.1)))
        f_lastDecile = f_lastDecile.iloc[i_pos].values[0]
        return df.ix[(df_t>=f_lastDecile)[s_feature],[s_feature, "poi"]]

    def describe(self):
        '''
        Return a  dataframe with the stats of all numeric data in the instance
        dataset
        '''
        #filter out email feature
        df = self.getData()
        l_labels = [x for x in df.columns if x!="email_address"]
        df_rtn = (df.ix[:,l_labels].astype(float).describe().T)
        #format numbers
        for key in df_rtn:
            if key == "count": s_txt = '{:,.2f}'
            else: s_txt = '{:,.0f}'
            df_rtn[key] = df_rtn[key].map(s_txt.format)

        return df_rtn


    def checkSummation(self):
        '''
        Return a dataframe with just the data points that presented any 
        difference between the data and the summations
        '''
        df = self.getData()
        #test the stock information
        l_labels = [u'poi', u'restricted_stock', u'restricted_stock_deferred', 
        u'exercised_stock_options', u'total_stock_value']
        df_check = df.drop(['email_address'], axis =1)
        df_check = df_check.astype(float)
        df_check = df_check.ix[:,l_labels]
        df_check['Delta'] = 0
        df_check.Delta = df_check.ix[:,l_labels[1:-1]].sum(axis = 1)
        df_check.Delta -= df_check.total_stock_value
        df_check.fillna(0, inplace=True)
        df_t1 = df_check[df_check.Delta!=0].T
        #test payment information
        l_label = [u'poi', u'bonus', u'deferral_payments', u'deferred_income', 
        u'director_fees', u'expenses', u'loan_advances', u'long_term_incentive',
        u'other',u'salary', u'total_payments']
        df_check = df.drop(['email_address'], axis =1)
        df_check = df_check.astype(float)
        df_check = df_check.ix[:,l_label]
        df_check['Delta'] = 0
        df_check.Delta = df_check.ix[:,l_label[1:-1]].sum(axis = 1)
        df_check.Delta -= df_check.total_payments
        df_check.fillna(0, inplace=True)
        df_t2 = df_check[df_check.Delta!=0].T

        return df_t1, df_t2




