from flask import Flask, jsonify, request
import joblib
import re
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn.linear_model
from datetime import datetime, date, time
from sklearn.preprocessing import LabelEncoder
import os, types
import pandas as pd

import pickle
import flask

app = Flask(__name__)
clf = joblib.load('APIPredict.pkl')

from train import MultiColumnLabelEncoder
preprocessing_1 = joblib.load('Preprocessing_1.pkl')
preprocessing_2 = joblib.load('Preprocessing_2.pkl')


###################################################
def pre_processing(df_data_1_replace):
    df_data_1_replace['ScheduledDay_year'] = pd.to_datetime(df_data_1_replace['datetimestamp']).dt.year
    df_data_1_replace['ScheduledDay_month'] = pd.to_datetime(df_data_1_replace['datetimestamp']).dt.month
    df_data_1_replace['ScheduledDay_week'] = pd.to_datetime(df_data_1_replace['datetimestamp']).dt.week
    df_data_1_replace['ScheduledDay_day'] = pd.to_datetime(df_data_1_replace['datetimestamp']).dt.day
    df_data_1_replace['ScheduledDay_hour'] = pd.to_datetime(df_data_1_replace['datetimestamp']).dt.hour
    df_data_1_replace['ScheduledDay_minute'] = pd.to_datetime(df_data_1_replace['datetimestamp']).dt.minute
    df_data_1_replace['ScheduledDay_dayofweek'] = pd.to_datetime(df_data_1_replace['datetimestamp']).dt.dayofweek

    df_data_1_replace.drop(['datetimestamp'], axis='columns', inplace=True)

    # column_to_move = df_data_1_replace.pop("status_code")
    # df_data_1_replace.insert(len(df_data_1_replace.columns), "status_code", column_to_move)
    # df_data_1_replace.head(5)

    no_columns = len(df_data_1_replace.columns)
    train_columns = no_columns
    # OriginalX = df_data_1_replace.apply(LE)
    try:
        OriginalX = preprocessing_1.transform(df_data_1_replace)
        X = preprocessing_2.transform(OriginalX.iloc[:, 0:train_columns])

    except:
        X='Exception'


    return X


###################################################


@app.route('/')
def index():
    return flask.render_template('index.html')


@app.route('/predict', methods=['POST'])

def predict():
    to_predict_list = request.form.to_dict()

    data = [to_predict_list]
    df = pd.DataFrame(data, columns=['datetimestamp', 'client_ip', 'gateway_ip'])
    #df['status_code'] = '0'
    transformed_data = pre_processing(df)
    if(transformed_data!='Exception'):
        pred = clf.predict(transformed_data)
    else:
        pred="-1"
    # prob = clf.predict_proba(count_vect.transform([review_text]))
    # pr =  1

    if pred[0] == 1:
        prediction = "200"
        # pr = prob[0][0]
    elif pred[0] == 2:
        prediction = "401"
    elif pred[0] == 3:
        prediction = "500"
    elif pred == "-1":
        prediction = "-1"
    else:
        prediction="999"

    return flask.render_template('index.html', prediction=prediction)


if __name__ == '__main__':
    # clf = joblib.load('quora_model.pkl')
    # count_vect = joblib.load('quora_vectorizer.pkl')

    app.run(debug=False)
    from train import MultiColumnLabelEncoder

    # app.run(host='localhost', port=8081)


