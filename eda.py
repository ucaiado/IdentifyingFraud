#!/usr/bin/python
# -*- coding: utf-8 -*-

'''
Explore data that will be used in the project


Created on 25/06/2015
'''

__author__='ucaiado'


import pandas as pd
import numpy as np
import seaborn as sns
import sys
import pickle
import matplotlib.pyplot as plt
sys.path.append("tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import test_classifier, dump_classifier_and_data


sns.set_palette("deep", desat=.6)
sns.set_context(rc={"figure.figsize": (8, 4)})


'''
Begin of Help Functions
'''


'''
End of Help Functions
'''



class Eda(object):
    '''
    Load data and summarize it executing exploratory data analysis
    '''
    def __init__(self):
        '''
        Initialize a Eda instance
        '''
        self._loadData()
        self.payments_features = ['bonus', 'deferral_payments', 
        'deferred_income', 'director_fees', 'expenses','loan_advances', 
        'long_term_incentive', 'other','salary']
        self.stock_features = ['exercised_stock_options','restricted_stock', 
        'restricted_stock_deferred']
        self.email_features = ['from_messages', 'from_poi_to_this_person',
        'from_this_person_to_poi','shared_receipt_with_poi', 'to_messages']
       


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
        #save the result as an attribute 
        self.data_description = df_rtn

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

    def excludeOutliers(self, l_outliers):
        '''
        Exclude the data pints in l_outliers
        '''
        df = self.getData()
        df.drop(l_outliers, inplace= True)
        self.setData(df)

    def fill_and_remove(self):
        '''
        fill all Nan values in numerical data with zeros and then remove data 
        points that all features are equal to zero
        '''
        df = self.getData()
        #filling Nan with 0 and exclude them
        l_features = self.payments_features + self.stock_features 
        l_features+= self.email_features
        df.loc[:, l_features] = df.loc[:, l_features].astype(float)
        df.loc[:, l_features] = df.loc[:, l_features].fillna(0)
        df = df.ix[((df.loc[:, l_features]!=0).sum(axis=1)!=0),:]
        #saving the new dataframe       
        self.setData(df)


    def compareFeaturesCreated(self):
        '''
        plot the box plot of the new features that will be created, 
        biggest_expenses and percentual_exercised. Keep the results as 
        attributes called new_feature_1 and new_feature_2
        '''
        #get a copy of the data
        df = self.getData()
        #compare the expenses to the biggest one scaling it
        f_min = df.expenses.astype(float).min()
        f_max = df.expenses.astype(float).max() 
        df_t2 = (df.expenses.astype(float) - f_min)/(f_max - f_min)
        df_t2 = pd.DataFrame(df_t2)
        df_t2.columns = ["biggest_expenses"]
        df_t2["poi"]=df.poi
        df_t2 = df_t2.fillna(0)
        #exclude some points just to plot
        df_t2_plot = df_t2[df_t2.biggest_expenses<0.80]
        #compare the exercised stock options to total payment
        l_features =  ['exercised_stock_options', 'total_payments']
        df_t3 = df[l_features[0]].astype(float)/df[l_features[1]].astype(float)
        df_t3 = pd.DataFrame(df_t3)
        #scale the new feature
        f_min = df_t3.min()
        f_max = df_t3.max()
        df_t3 = (df_t3-f_min)/ (f_max - f_min)
        df_t3 = df_t3.fillna(0)
        #exclude some outliers just to this plot
        df_t3.columns = ["percentual_exercised"]
        df_t3["poi"]=df.poi
        f_max = df_t3.percentual_exercised.max()
        df_t3_plot = df_t3[df_t3.percentual_exercised != f_max]
        #plot the both camparitions in one figure
        f, l_ax = plt.subplots(1,2)
        ax1 = sns.boxplot(x="poi", y="biggest_expenses", 
            data=df_t2_plot, ax = l_ax[0]);
        ax2 = sns.boxplot(x="poi", y="percentual_exercised",
            data=df_t3_plot, ax = l_ax[1]);
        ax2.set_title("Option Exercised Compared to\n Total Payments");
        ax1.set_title("How far Is Each One From \nThe Biggest Expense");
        f.tight_layout()
        #include the new features in the original dataset
        df['biggest_expenses'] = df_t2['biggest_expenses']
        df["percentual_exercised"] = df_t3["percentual_exercised"]
        self.setData(df)
        self.new_features = ['biggest_expenses', 'percentual_exercised']


    def scallingAll(self):
        '''
        Scale each group of features, keep the result as an attribute 
        '''
        #load data
        df = self.getData()
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
        self.df_scaled = df
        # df.loc[:,l_payment + l_stock + l_email].astype(float).describe().T
