# In[ ]:

from typing import List
from flask import Flask, render_template, url_for, request, flash

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
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
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

# Package Flasks
import os
from flask import Response
from werkzeug.utils import secure_filename

# In[ ]:


app = Flask(__name__)

UPLOAD_FOLDER = 'data'
ALLOWED_EXTENSIONS = {'csv'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


stop_factory = StopWordRemoverFactory() 
stopword = stop_factory.get_stop_words()
factory = StemmerFactory()
stemmer = factory.create_stemmer()

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def stemming(tweet):
    return stemmer.stem(tweet)

def remove_punct(tweet):
  tweet = re.sub(r'https?://[A-Za-z0-9./]+', '', tweet)
  tweet = re.sub(r'[^a-zA-Z0-9]',' ', str(tweet))
  tweet = re.sub(r'\b\w[1,2]\b','', str(tweet))
  tweet = re.sub(r'\s\s+',' ', str(tweet))
  return tweet

def to_lowercase(words): 
    new_words = [] 
    for word in words: 
        new_word = word.lower() 
        new_words.append(new_word) 
    return new_words

def tokenization(tweet):
    tweet = re.split("\W+", tweet)
    return tweet

def remove_stopwords(tweet):
    tweet = [word for word in tweet if word not in stopword]
    return tweet

@app.route('/jsondata', methods=['GET'])
def jsondata():
    df = pd.read_csv('data/data_tweets.csv')
    jsondata = df.to_json(orient="records")
    return Response(jsondata, mimetype='application/json')

@app.route('/', methods=['GET'])
def index():
     return render_template('index.html')

@app.route('/maps', methods=['GET'])
def maps():
     return render_template('maps.html')



@app.route('/klasifikasi')
def klasifikasi():
    return render_template('klasifikasi.html')
    
@app.route('/data')
def data():
    data = pd.read_csv("data/data_tweets.csv")
    return render_template('data.html',data=data)

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('klasifikasi.html')
        
        file = request.files['file']

        if file.filename == '':
            return render_template('klasifikasi.html')
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            
            loaded_model = joblib.load('data/modelNB.pkl')
            dt = pd.read_csv("data/"+ filename, delimiter=';')

            dt['clean_tweet'] = dt['tweet'].apply(lambda x: remove_punct(x))
            dt['clean_tweet'] = to_lowercase(dt['clean_tweet'])
            dt['stemming'] = dt['clean_tweet'].apply(lambda x: remove_punct(x))
            dt['tokenizing'] = dt['stemming'].apply(lambda x: tokenization(x))
            dt['stop_removal'] = dt['tokenizing'].apply(lambda x: remove_stopwords(x))

            # Save Load Data Praproses
            dt.to_csv(r'data/temp.csv', index = False)
            data = pd.read_csv('data/temp.csv')
            predictLabel = loaded_model.predict(data['stop_removal'])
            data['label'] = predictLabel
            
            # Save hasil klasifikasi
            data.to_csv(r'data/temp.csv', index = False,columns=['id','created_at','user_id','username','name','tweet','geo','clean_tweet','stemming','tokenizing','stop_removal','label'])             
            
            # Penggabungan Data Test dengan Data Utama (Corpus)
            df1 = pd.read_csv('data/data_tweets.csv')
            df2 = pd.read_csv('data/temp.csv')

            frames = [df1, df2]
            result = pd.concat(frames)

            result.to_csv(r'data/data_tweets.csv', index = False)

            return render_template('predict.html',data=data)
    return render_template('predict.html')
if __name__ == '__main__':
    app.run(debug=True)

# %%
