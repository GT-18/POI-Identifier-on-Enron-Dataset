#!/usr/bin/python

""" 
    Starter code for exploring the Enron dataset (emails + finances);
    loads up the dataset (pickled dict of dicts).

    The dataset has the form:
    enron_data["LASTNAME FIRSTNAME MIDDLEINITIAL"] = { features_dict }

    {features_dict} is a dictionary of features associated with that person.
    You should explore features_dict as part of the mini-project,
    but here's an example to get you started:

    enron_data["SKILLING JEFFREY K"]["bonus"] = 5600000
    
"""

import pickle

enron_data = pickle.load(open("../final_project/final_project_dataset.pkl", "r"))
# Q1 - print(len(enron_data))

# Q2 - for i in enron_data.keys():
# 	print(len(enron_data[i]))

# Q3 - ct=0
# for i in enron_data.keys():
# 	ct += enron_data[i]["poi"]
# print(ct)

# Q4 - Look for poi_names.txt 

# Q5 - for i in enron_data["PRENTICE JAMES"].keys():
# 	print(i," : ",enron_data["PRENTICE JAMES"][i])

# Q6 - for i in enron_data["COLWELL WESLEY"].keys():
# 	print(i," : ",enron_data["COLWELL WESLEY"][i])

for i in enron_data["SKILLING JEFFREY K"].keys():
 	print(i," : ",enron_data["SKILLING JEFFREY K"][i])
