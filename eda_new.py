#!/usr/bin/python
# -*- coding: utf-8 -*-

'''
Explore the data that will be used in the project


Created on 06/25/2015
'''

__author__='ucaiado'


import pandas as pd
import numpy as np
import seaborn as sns
import sys
import pickle
import matplotlib.pyplot as plt

sys.path.append("tools/")
sns.set_palette("deep", desat=.6)
sns.set_context(rc={"figure.figsize": (8, 4)})


'''
Begin of Help Functions
'''


'''
End of Help Functions
'''
def precision_recall_curve(f_LoadEnron, f_Classifier, o_features, features_list2):
    '''
    Plot a precision recall curve for a KNeighbors algorithm
    f_LoadEnron: an function called LoadEnron from dataser library
    f_Classifier:an function called Classifier from Classifiers libarry
    o_features: and instance from featureSelections'Feature object
    features_list2: a list of all features to be tested
    '''
    #initialize variable
    o_stats = Eda()
    l_precision = []
    l_recall = []
    l_percentile = range(10,100, 10)
    for i_perc in l_percentile:
        #loading data
        o_enron = f_LoadEnron()
        o_enron.excludeOutliers(l_outliers =  ["BELFER ROBERT",
            "BHATNAGAR SANJAY", "TOTAL"])
        o_features.createNewFeatures(o_enron)
        o_enron.fill_and_remove(l_features= features_list2,
            s_strategy="zeros", b_remove=True)
        #selecting features based on X percentile
        labels, features= o_features.getFeaturesAndLabels(o_enron, 
            l_columns=features_list2 , o_eda = o_stats)
        l_selectedFetures, df_rtn = o_features.select(features, labels, 
            features_list2, percentile = i_perc) 
        #exclude datapoint based on X percentile features
        o_enron.fill_and_remove(l_features= l_selectedFetures,
            s_strategy="zeros", b_remove=True)
        #trainf and test the classifier
        labels, features= o_features.getFeaturesAndLabels(o_enron, 
            l_columns=l_selectedFetures , o_eda = o_stats)
        clf2 = f_Classifier("KNeighbors", usePCA = False, scale_on_pipe=False)
        clf2.crossValidation(features, labels,  report = False, 
            l_columns= ['aux'])
        l_precision.append(clf2.d_performance["precision"])
        l_recall.append(clf2.d_performance["recall"])  
    #create a data frame
    df = pd.DataFrame([l_precision, l_recall], columns = l_percentile, 
        index = ['Precision', 'Recall']).T
    df.index.name = "Percentile" 
    #plot precision recall
    ax = df.plot()
    ax.set_title("Precision-Recall\n"); 




class Eda(object):
    '''
    execute a basic exploratory data analysis in a given data set and 
    '''
    def __init__(self):
        '''
        Initialize a Eda instance
        '''
        pass     

    def NanAtFetaures(self, o_dataset):
        '''
        plot a bar plot with the percentual off Nan in each feature
        o_dataset: an object with the dataset loaded
        '''
        df = o_dataset.getData()
        ax  = ((df=="NaN").sum()/146.*100).plot(kind = "bar")
        ax.set_title("Percentual of Null Values in each Feature\n")

    def scatter(self, s_xFeature, s_yFeature, o_dataset):
        '''
        Plot a scatter plot with the labels passed
        s_xFeature: string with the column label to be used as the x axis
        s_yFeature: string with the column label to be used as the y axis
        o_dataset: an object with the dataset loaded
        '''

        df = o_dataset.getData()
        df_plot = df.ix[:, [s_xFeature, s_yFeature]]
        df_plot = df_plot[(df_plot=="NaN").sum(axis = 1)==0].astype(float)
        g = sns.jointplot(s_xFeature, s_yFeature, data=df_plot, kind="reg",
         size=6)

    def getDecile(self, s_feature, o_dataset,  i_pos = -1):
        '''
        get the labels for the last decile in the data set, given a s_feature
        o_dataset: an object with the dataset loaded
        '''
        df = o_dataset.getData()
        df_t = df.ix[:, [s_feature]].astype("float")
        f_lastDecile = (df_t.quantile(np.arange(0.1,1.1,0.1)))
        f_lastDecile = f_lastDecile.iloc[i_pos].values[0]
        return df.ix[(df_t>=f_lastDecile)[s_feature],[s_feature, "poi"]]

    def describe(self, o_dataset):
        '''
        Return a  dataframe with the stats of all numeric data in the instance
        dataset
        o_dataset: an object with the dataset loaded
        '''
        #filter out email feature
        df = o_dataset.getData()
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


    def checkSummation(self, o_dataset):
        '''
        Return a dataframe with just the data points that presented any 
        difference between the data and the summations
        '''
        df = o_dataset.getData()
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


    def notValidNumbersTable(self, o_dataset):
        '''
        Return a dataframe with the percentual of not NaN values in the data 
        set
        '''
        #recover datasets
        df = o_dataset.getData(original = True)
        df2 = o_dataset.getData()
        #set numbers as floats
        l_features = [x for x in df.columns if x not in ['email_address','poi']]
        df.loc[:, l_features] = df.loc[:, l_features].astype(float)
        #input new variable
        l_new = ['biggest_expenses', 'percentual_exercised'] 
        df[l_new]=df2.ix[:,l_new]
        #count zeros
        df.fillna(0, inplace = True)
        df_rtn = pd.DataFrame(((df!=0).sum()*1./df.shape[0]))
        df_rtn.columns=["ValidNumbers"]
        df_rtn = df_rtn.sort("ValidNumbers", ascending=False)
        return df_rtn

    def compareFeaturesCreated(self, o_dataset):
        '''
        plot the box plot of the new features that will be created, 
        biggest_expenses and percentual_exercised. Keep the results as 
        attributes called new_feature_1 and new_feature_2
        '''
        #get a copy of the data
        df = o_dataset.getData()
        #load biggest expenses feature
        df_t2 = df.loc[:,["biggest_expenses", "poi"]]
        df_t2 = df_t2.fillna(0)
        #exclude some points just to plot
        df_t2_plot = df_t2[df_t2.biggest_expenses<0.005]
        #load percentual_exercised feature
        df_t3 = df.loc[:,["percentual_exercised", "poi"]]
        df_t3 = df_t3.fillna(0)        
        f_max = df_t3.percentual_exercised.max()
        df_t3_plot = df_t3[df_t3.percentual_exercised < 0.10]

        #plot the both camparitions in one figure
        f, l_ax = plt.subplots(1,2)
        ax1 = sns.boxplot(x="poi", y="biggest_expenses", 
            data=df_t2_plot, ax = l_ax[0]);
        ax2 = sns.boxplot(x="poi", y="percentual_exercised",
            data=df_t3_plot, ax = l_ax[1]);
        ax2.set_title("Option Exercised Compared to\n Total Payments");
        ax1.set_title("Money from Enron spent by Employees\n Compared to Their Salaries");
        f.tight_layout()     