# In[ ]:

from flask import Flask, render_template, url_for, request

# ML PACKAGES LIB
import nltk
import string
import re
import joblib
import pickle
import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from sklearn import model_selection
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from imblearn.pipeline import Pipeline, make_pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import BorderlineSMOTE

# In[ ]:


app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/klasifikasi')
def klasifikasi():
    return render_template('klasifikasi.html')


@app.route('/data')
def data():
    data = pd.read_csv("data/data_tweets.csv", delimiter=';')
    return render_template('data.html',data=data)


if __name__ == '__main__':
    app.run(debug=True)
