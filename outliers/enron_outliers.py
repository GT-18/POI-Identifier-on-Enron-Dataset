#!/usr/bin/python

import pickle
import sys
import matplotlib.pyplot
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit


### read in data dictionary, convert to numpy array
data_dict = pickle.load( open("../final_project/final_project_dataset.pkl", "r") )
features = ["salary", "bonus"]
data = featureFormat(data_dict, features)


### your code below

outlier = "random"
data = sorted(data, key = lambda x:x[0])
for i in data_dict:
	if data_dict[i]["salary"] == data[-1][0]:
		print(i, data_dict[i]["salary"])
		outlier = i
		break

data_dict.pop(outlier,None)
data = featureFormat(data_dict, features)

for i in data_dict:
	if data_dict[i]["salary"] >= 1000000 and data_dict[i]["bonus"] >= 5000000 and data_dict[i]["salary"] != "NaN" and data_dict[i]["bonus"] != "NaN":
		print(i, data_dict[i]["salary"], data_dict[i]["bonus"])

for point in data:
    salary = point[0]
    bonus = point[1]
    matplotlib.pyplot.scatter( salary, bonus )

matplotlib.pyplot.xlabel("salary")
matplotlib.pyplot.ylabel("bonus")
matplotlib.pyplot.show()


