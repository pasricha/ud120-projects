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

enron_data = pickle.load(open("../final_project/final_project_dataset.pkl", "rb"))


print 'Data points: ', len(enron_data)

print 'Number of features: ', len(enron_data['SKILLING JEFFREY K'])

poi = 0
for person in enron_data.keys():
    if enron_data[person]['poi'] == 1:
        poi += 1
print 'POIs: ', poi

# Total POIs: 35 (in ../final_project/poi_names.txt)

print 'Stock belonging to James Prentice: ', \
    enron_data['PRENTICE JAMES']['total_stock_value']

print 'Messages from Wesley Colwell to POIs', \
    enron_data['COLWELL WESLEY']['from_this_person_to_poi']

print 'Stock options exercised by Jeffery Skilling: ', \
    enron_data['SKILLING JEFFREY K']['exercised_stock_options']

nan_total_payments = 0
for person in enron_data.keys():
    if enron_data[person]['total_payments'] == 'NaN':
        nan_total_payments += 1
print 'Percentage of people with "NaN" total payments: ', \
    float(nan_total_payments) / len(enron_data)

nan_total_payments_poi = 0
for person in enron_data.keys():
    if enron_data[person]['poi']:
        if enron_data[person]['total_payments'] == 'NaN':
            nan_total_payments += 1
print 'Percentage of POIs with "NaN" total payments: ', \
    float(nan_total_payments_poi) / poi

print 'With 10 more POIs having "NaN" total payments,'
print 'Number of people: ', 10 + len(enron_data)
print 'Number of people with "NaN" total payments: ', 10 + nan_total_payments
print 'New Number of POIs', 10 + poi
print 'Number of POIs with "NaN" total payments', 10 + nan_total_payments_poi



