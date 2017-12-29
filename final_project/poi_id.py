#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
						

""" Section I - Features and basic data analysis """
##########################################################################################################
print "#####################################################################################\n"
### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi','salary'] # You will need to use more features

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

all_features = []
for i in data_dict.keys():
	all_features = data_dict[i].keys()
print "List of all feauters :", all_features
print "\n\n"

all_features = ['salary', 'to_messages', 'deferral_payments', 'total_payments', 'exercised_stock_options', 'bonus', 
                'restricted_stock', 'shared_receipt_with_poi', 'restricted_stock_deferred', 'total_stock_value', 
                'expenses', 'loan_advances', 'from_messages', 'other', 'from_this_person_to_poi', 'poi', 
                'director_fees', 'deferred_income', 'long_term_incentive', 'email_address', 'from_poi_to_this_person']


features_list = ['poi','salary', 'bonus', 'total_payments', 'total_stock_value', 'poi_email_ratio',
                 'deferral_payments','deferred_income', 'director_fees', 'exercised_stock_options', 
                 'expenses', 'loan_advances','long_term_incentive', 'other', 'restricted_stock', 
                 'restricted_stock_deferred'] 

print "Total number of data points in the data : ", len(data_dict)
print"\n"

no_of_poi = 0
for i in data_dict.keys():
	if data_dict[i]['poi']:
		no_of_poi += 1

print "Number of poi :", no_of_poi
print "Number of non poi :", len(data_dict) - no_of_poi
print "\n"
print "\n\n"


print "#####################################################################################\n\n\n"
#############################################################################################################

### Task 2: Remove outliers


import matplotlib.pyplot

def visualise(data_dict, features):
	data_to_plot = featureFormat(data_dict, features)

	for i in data_to_plot:
		if i[1] != 'NaN' and i[0] !='NaN':
			# Uncomment this line for email ratio outliers
			# matplotlib.pyplot.scatter(i[0]/i[1], i[1])
			matplotlib.pyplot.scatter(i[0], i[1])

	matplotlib.pyplot.xlabel(features[0])
	matplotlib.pyplot.ylabel(features[1])
	matplotlib.pyplot.show()

#visualise(data_dict, ['salary', 'bonus'])
data_dict.pop('TOTAL', 0)
#visualise(data_dict, ['salary', 'bonus'])


print "One outlier removed for salary and bonus as \"Total\" is of no concern to us\n"


#visualise(data_dict, ['from_this_person_to_poi', 'from_poi_to_this_person'])
print "One outlier removed for email ratio \n"
data_dict.pop("THE TRAVEL AGENCY IN THE PARK",0)
#visualise(data_dict, ['from_this_person_to_poi', 'from_poi_to_this_person'])



print "#####################################################################################\n\n\n"
#############################################################################################################


### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
my_dataset = data_dict

for i,person in my_dataset.items():
	try:
		r = float(person['from_poi_to_this_person'] + person['from_this_person_to_poi'])/float(person['to_messages']+person['from_messages'])
		person['poi_email_ratio'] = r
	except ValueError:
		person['poi_email_ratio'] = 0

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

### Feature Scaling
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
features = scaler.fit_transform(features)


### Feature Selection
from sklearn.feature_selection import SelectKBest
num_features = 10
selector = SelectKBest(k = num_features)
selector.fit(features, labels)
scores = selector.scores_
feature_and_score = sorted(zip(features_list[1:], scores), key = lambda l: l[1], reverse = True)
k_best_features = []
for i in range(num_features):
	k_best_features.append(feature_and_score[i][0])
best_features_list = ['poi'] + k_best_features
print best_features_list

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

print "#####################################################################################\n\n\n"
#############################################################################################################


### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html
# Provided to give you a starting point. Try a variety of classifiers.
from sklearn.naive_bayes import GaussianNB
clf_NB = GaussianNB()

from sklearn.tree import DecisionTreeClassifier
clf_DT = DecisionTreeClassifier()

from sklearn.ensemble import AdaBoostClassifier
clf_Ada = AdaBoostClassifier()

from sklearn.linear_model import Lasso
clf_lasso = Lasso()

print "#####################################################################################\n\n\n"
#############################################################################################################

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

from time import time

params_DT = {
	'criterion' : ('gini', 'entropy'),
	'min_samples_split' : [2,5]
}

params_lasso = {
	'max_iter' : [10000,50000]
}

from sklearn.model_selection import GridSearchCV


# Naive Bayes
t0 = time()
clf_NB.fit(features_train, labels_train)
print "Training time of Naive Bayes", round(time()-t0, 3)
t0 = time()
pred_NB = clf_NB.predict(features_test)
print "Prediction time of Naive Bayes", round(time()-t0, 3)

print("\n")


# Decision Tree
t0 = time()
clf_DT = GridSearchCV(clf_DT, params_DT)
clf_DT.fit(features_train, labels_train)
print "Training time of Decision Tree", round(time()-t0, 3)
t0 = time()
pred_DT = clf_DT.predict(features_test)
print "Prediction time of Decision Tree", round(time()-t0, 3)

print("\n")


# Adaboost
t0 = time()
clf_Ada.fit(features_train, labels_train)
print "Training time of Adaboost", round(time()-t0, 3)
t0 = time()
pred_Ada = clf_Ada.predict(features_test)
print "Prediction time of Adaboost", round(time()-t0, 3)

print("\n")


# Lasso Regression
t0 = time()
clf_lasso = GridSearchCV(clf_lasso, params_lasso)
clf_lasso.fit(features_train, labels_train)
print "Training time of Lasso Regression", round(time()-t0, 3)

print("\n")



### Evaluation

from sklearn.metrics import precision_score, recall_score
print "precision score for the Gaussian Naive Bayes Classifier : ",precision_score(labels_test,pred_NB)
print "recall score for the Gaussian Naive Bayes Classifier : ",recall_score(labels_test,pred_NB)

print "precision score for the Decision tree Classifier : ",precision_score(labels_test,pred_DT)
print "recall score for the Decision tree Classifier : ",recall_score(labels_test,pred_DT)

print "precision score for the AdaBoost Classifier : ",precision_score(labels_test,pred_Ada)
print "recall score for the AdaBoost Classifier : ",recall_score(labels_test,pred_Ada)

print "R2 score for the Lasso Regression Training set : ",clf_lasso.score(features_train, labels_train)
print "R2 score for the Lasso Regression Test set : ",clf_lasso.score(features_test, labels_test)



print "#####################################################################################\n\n\n"
#############################################################################################################
### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

clf = clf_NB
dump_classifier_and_data(clf, my_dataset, features_list)