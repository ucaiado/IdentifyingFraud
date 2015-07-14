#!/usr/bin/python

import sys
import pickle
sys.path.append("tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import test_classifier, dump_classifier_and_data

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi', 'bonus', 'salary', 'exercised_stock_options']

### Load the dictionary containing the dataset
data_dict = pickle.load(open("final_project_dataset.pkl", "r") )

### Task 2: Remove outliers
key_to_remove = ['TOTAL',"BELFER ROBERT","BHATNAGAR SANJAY"]
for key in key_to_remove:
    data_dict.pop(key)

### Task 3: Create new feature(s)
#### Load the classes created to this project
import dataset
import featureSelection

o_enron = dataset.LoadEnron()
o_features = featureSelection.Features()
#### Clean the data set to look like data_dict
l_exclude = ["BELFER ROBERT","BHATNAGAR SANJAY", "TOTAL"]
o_enron.excludeOutliers(l_outliers =  l_exclude)
o_enron.fill_and_remove(b_remove = False)
o_features.createNewFeatures(o_enron)

####insert features created in data_dict
for key in data_dict:
    for new_feature in ['biggest_expenses', 'percentual_exercised']:
        data_dict[key][new_feature] = o_enron.getValue(key, new_feature)


### Store to my_dataset for easy export below.
my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a varity of classifiers.
## --------------------------------------------------------------------------##
## !!Different classifiers were tested in ipython notebook named Report. Please,
## !!check it out.
## --------------------------------------------------------------------------##
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA


estimators = [('reduce_dim', PCA()),
              ('KNeighbors', KNeighborsClassifier())]
clf = Pipeline(estimators)

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script.
### Because of the small size of the dataset, the script uses stratified
### shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html
clf.set_params(KNeighbors__n_neighbors =  4)
clf.set_params(KNeighbors__p =  2)
clf.set_params(reduce_dim__n_components =  2)


test_classifier(clf, my_dataset, features_list)

### Dump your classifier, dataset, and features_list so 
### anyone can run/check your results.

dump_classifier_and_data(clf, my_dataset, features_list)