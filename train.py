import numpy as np
import pandas as pd
import seaborn as sns
#import tensorflow as tf
import matplotlib.pyplot as plt
import sklearn.linear_model
from datetime import datetime, date, time
import time
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os, types
import pandas as pd
#from botocore.client import Config
#import ibm_boto3
import pickle

import pymongo
import pandas as pd


client = pymongo.MongoClient("mongodb://admin:Pa11word-1@3.239.40.22/admin") # defaults to port 27017

db = client.dbuser
collection_name = db.apicdata
cursor = collection_name.find()

list_cur = list(cursor)
df_data_1 = pd.DataFrame(list_cur)
df_data_1.drop(['_id'], axis='columns', inplace=True)



#df_data_1= pd.read_csv('Final_report.csv')
#df_data_1.drop(['Unnamed: 0'], axis='columns', inplace=True)
df_data_1.drop(['latency_info2'], axis='columns', inplace=True)
df_data_1.drop(['bytes_sent'], axis='columns', inplace=True)
df_data_1.drop(['rateLimit'], axis='columns', inplace=True)

replace_list = {"200 OK":1, "401": 2, "500":3 }
replace_map = {'status_code': {'200 OK': 1, '401': 2, '500': 3, }}

labels = df_data_1['status_code'].astype('category').cat.categories.tolist()
replace_map_comp = {'status_code' : {k: v for k,v in zip(labels,list(range(1,len(labels)+1)))}}

df_data_1_replace = df_data_1.copy()
df_data_1_replace.replace(replace_map_comp, inplace=True)

df_data_1_replace['ScheduledDay_year'] = pd.to_datetime(df_data_1_replace['datetimestamp']).dt.year
df_data_1_replace['ScheduledDay_month'] = pd.to_datetime(df_data_1_replace['datetimestamp']).dt.month
df_data_1_replace['ScheduledDay_week'] = pd.to_datetime(df_data_1_replace['datetimestamp']).dt.week
df_data_1_replace['ScheduledDay_day'] = pd.to_datetime(df_data_1_replace['datetimestamp']).dt.day
df_data_1_replace['ScheduledDay_hour'] = pd.to_datetime(df_data_1_replace['datetimestamp']).dt.hour
df_data_1_replace['ScheduledDay_minute'] = pd.to_datetime(df_data_1_replace['datetimestamp']).dt.minute
df_data_1_replace['ScheduledDay_dayofweek'] = pd.to_datetime(df_data_1_replace['datetimestamp']).dt.dayofweek

df_data_1_replace.drop(['datetimestamp'],axis='columns', inplace=True)

column_to_move = df_data_1_replace.pop("status_code")
df_data_1_replace.insert(len(df_data_1_replace.columns), "status_code", column_to_move)

class MultiColumnLabelEncoder:

    def __init__(self, columns=None):
        self.columns = columns # array of column names to encode


    def fit(self, X, y=None):
        self.encoders = {}
        columns = X.columns if self.columns is None else self.columns
        for col in columns:
            self.encoders[col] = LabelEncoder().fit(X[col])
        return self


    def transform(self, X):
        output = X.copy()
        columns = X.columns if self.columns is None else self.columns
        for col in columns:
            output[col] = self.encoders[col].transform(X[col])
        return output


    def fit_transform(self, X, y=None):
        return self.fit(X,y).transform(X)


    def inverse_transform(self, X):
        output = X.copy()
        columns = X.columns if self.columns is None else self.columns
        for col in columns:
            output[col] = self.encoders[col].inverse_transform(X[col])
        return output

    

multi = MultiColumnLabelEncoder(columns =   ['client_ip','gateway_ip'])
multi = multi.fit(df_data_1_replace)
OriginalX = multi.transform(df_data_1_replace)



pickle.dump(multi, open('Preprocessing_1.pkl', 'wb'))

X=OriginalX.loc[:, OriginalX.columns != 'status_code']
y = OriginalX['status_code']



sc = StandardScaler()

no_columns=len(df_data_1_replace.columns)
train_columns=no_columns-1

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
X_train.iloc[:, 0:train_columns]= sc.fit_transform(X_train.iloc[:, 0:train_columns])
X_test.iloc[:, 0:train_columns] = sc.transform(X_test.iloc[:, 0:train_columns])
X_train.head()

pickle.dump(sc, open('Preprocessing_2.pkl', 'wb'))

from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 300, random_state = 0)
classifier.fit(X, y)
#r2_score_clf = classifier.score(X_test, y_test)

class Score:
    def Score():
        r2_score_clf = classifier.score(X_test, y_test)
        return r2_score_clf*100
    
print('Accuracy of the model is : '+ str(round(Score.Score(),2)) + ' %')


pickle.dump(classifier, open('APIPredict.pkl', 'wb'))
#print(r2_score_clf*100,'%')
from pandas_profiling import ProfileReport
profile = ProfileReport(OriginalX)
profile.to_file("templates\graph.html")
