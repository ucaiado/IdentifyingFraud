#!/usr/bin/python
# -*- coding: utf-8 -*-

'''
Load different classifiers to be tested


Created on 07/09/2015
'''

__author__='ucaiado'



from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB




'''
Begin of Help Functions
'''


'''
End of Help Functions
'''



clf = DecisionTreeClassifier(max_depth=2)
clf = KNeighborsClassifier(n_neighbors=2)
clf = SVC(kernel='rbf', degree=3)
clf = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(min_samples_split=50))
clf = GaussianNB()    
clf = RandomForestClassifier(min_samples_split=50)
