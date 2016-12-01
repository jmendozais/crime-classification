import csv
import pandas
import datetime
import numpy as np

'''
'''

default_headers = 'Dates,DayOfWeek,PdDistrict,X,Y,Category'.split(',')

categories = 'ARSON,ASSAULT,BAD CHECKS,BRIBERY,BURGLARY,DISORDERLY CONDUCT,DRIVING UNDER THE INFLUENCE,DRUG/NARCOTIC,DRUNKENNESS,EMBEZZLEMENT,EXTORTION,FAMILY OFFENSES,FORGERY/COUNTERFEITING,FRAUD,GAMBLING,KIDNAPPING,LARCENY/THEFT,LIQUOR LAWS,LOITERING,MISSING PERSON,NON-CRIMINAL,OTHER OFFENSES,PORNOGRAPHY/OBSCENE MAT,PROSTITUTION,RECOVERED VEHICLE,ROBBERY,RUNAWAY,SECONDARY CODES,SEX OFFENSES FORCIBLE,SEX OFFENSES NON FORCIBLE,STOLEN PROPERTY,SUICIDE,SUSPICIOUS OCC,TREA,TRESPASS,VANDALISM,VEHICLE THEFT,WARRANTS,WEAPON LAWS'.split(',')
categories_ordinal = {}
for i in range(len(categories)):
    categories_ordinal[categories[i]] = float(i)

days = 'sunday,monday,thursday,wednesday,tuesday,friday,saturday'.split(',')
days_ordinal = {}
for i in range(len(days)):
    days_ordinal[days[i]] = float(i)

districs = ['CENTRAL','NORTHERN','INGLESIDE','PARK','MISSION','TENDERLOIN','RICHMOND','TARAVAL','BAYVIEW','SOUTHERN']
districs_ordinal = {}
for i in range(len(districs)):
    districs_ordinal[districs[i]] = float(i)

def date_feature(timestamp):
    timestamp = pandas.to_datetime(timestamp, infer_datetime_format=True)
    time_in_secs = (timestamp - datetime.datetime(2000, 1, 1)).total_seconds()
    return [timestamp.month, timestamp.day, timestamp.hour, timestamp.minute, time_in_secs]

def load(headers=default_headers):
    X = []
    Y = []
    with open('data/train.csv', 'rb') as file:
        reader = csv.DictReader(file, delimiter=',')
        for row in reader:
            X.append([])
            for i in range(len(headers)):
                feature = None
                if headers[i] == 'Dates':
                    feature = date_feature(row[headers[i]])

                elif headers[i] == 'DayOfWeek':
                    feature = [days_ordinal[row[headers[i]].lower()]]

                elif headers[i] == 'PdDistrict':
                    feature = [districs_ordinal[row[headers[i]]]]

                elif headers[i] == 'Category':   
                    feature = [categories_ordinal[row[headers[i]]]]
                    Y.append(feature)

                else:
                    feature = [float(row[headers[i]])]

                if headers[i] != 'Category' and feature != None:
                    X[-1].extend(feature)

    return np.array(X), np.array(Y)

if __name__ == '__main__':
    X, Y = load()
    print X.shape, Y.shape
    np.save('X', X)
    np.save('Y', Y)
