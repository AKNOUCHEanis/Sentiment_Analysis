# -*- coding: utf-8 -*-
"""
Created on Sat Oct 23 18:42:46 2021

@author: DELL VOSTRO
"""

import numpy as np 
import pandas as pd 
pd.set_option('display.max_colwidth', -1)
from time import time

import string
import os

from pprint import pprint
import collections
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="darkgrid")
sns.set(font_scale=1.3)

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.metrics import classification_report
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
#from sklearn.externals
import joblib
import gensim

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import warnings
warnings.filterwarnings('ignore')
np.random.seed(37)

from DataPreparation import DataPreparation

import tensorflow as tf
import gensim.downloader

if __name__=='__main__':
    
    #Load and shuffle the data
    df = pd.read_csv('../data/Tweets.csv')
    df = df.reindex(np.random.permutation(df.index))
    df = df[['text', 'airline_sentiment']]
    
    #print(df.groupby('airline_sentiment').count())
    
    
    cleanText=DataPreparation()
    df_cleaned=cleanText.clean_text(df,'text','airline_sentiment')
    X, Y, bow = cleanText.countVectorizer(df_cleaned, 'text', 'airline_sentiment')
    
    
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

    """
    #MultinomialNB
    mnb=MultinomialNB()
    
    parameters_mnb = {
        'alpha': (0.25, 0.5, 0.75)}
    
    grid_search = GridSearchCV(mnb, parameters_mnb, n_jobs=-1, verbose=1)
    
    t0 = time()
    grid_search.fit(X_train, y_train)
    print("MultinomialNB: with countvectorizer")
    print("done in %0.3fs" % (time() - t0))
    print()
    
    print("Train score with best_estimator_: %0.3f" % grid_search.best_estimator_.score(X_train, y_train))
    print("Test score with best_estimator_: %0.3f" % grid_search.best_estimator_.score(X_test, y_test))
    print("\n")
    print("Classification Report Test Data")
    print(classification_report(y_test, grid_search.best_estimator_.predict(X_test)))
    """
    
    
    #RegressionLogistique
    reglog = LogisticRegression()
    parameters_reglog = {
    'C': (0.25, 0.5, 1.0),
    'penalty': ('l1', 'l2')
    }
    
    grid_search = GridSearchCV(reglog, parameters_reglog, n_jobs=-1, verbose=1)
    
    t0 = time()
    grid_search.fit(X_train, y_train)
    print("RegressionLogistic: with countvectorizer")
    print("done in %0.3fs" % (time() - t0))
    print()
    
    print("Train score with best_estimator_: %0.3f" % grid_search.best_estimator_.score(X_train, y_train))
    print("Test score with best_estimator_: %0.3f" % grid_search.best_estimator_.score(X_test, y_test))
    print("\n")
    print("Classification Report Test Data")
    print(classification_report(y_test, grid_search.best_estimator_.predict(X_test)))
    
    """
    #word2vec
    #nltk.download('punkt')
    SIZE=50
    X_train, X_test, y_train, y_test = train_test_split(df_cleaned['text'], Y, test_size=0.2)
    
    X_train_tokenized = X_train.apply(lambda x : word_tokenize(x))
    X_test_tokenized  = X_test.apply( lambda x : word_tokenize(x))
    

    #modele pré-entrainé
    #model = gensim.downloader.load('glove-twitter-25')

    
    model = gensim.models.Word2Vec(X_train_tokenized, min_count=1, size=SIZE, window=5, workers=4)
    
    #model.most_similar("bad",topn=10)
    X_train_w2v = cleanText.word2vec(model, X_train.apply(lambda x: x.split(' ')).tolist(), SIZE)
    X_test_w2v  = cleanText.word2vec(model, X_test.apply(lambda x: x.split(' ')).tolist(), SIZE)
    
    
    #RegressionLogistique
    reglog = LogisticRegression()
    parameters_reglog = {
    'C': (0.25, 0.5, 1.0),
    'penalty': ('l1', 'l2')
    }
        
    grid_search = GridSearchCV(reglog, parameters_reglog, n_jobs=-1, verbose=1)
        
    t0 = time()
    X_train_w2v_pad = tf.keras.preprocessing.sequence.pad_sequences(
    X_train_w2v, maxlen=SIZE, dtype='int32', padding='pre',
    truncating='pre', value=0.0
    )

    X_test_w2v_pad = tf.keras.preprocessing.sequence.pad_sequences(
    X_test_w2v, maxlen=SIZE, dtype='int32', padding='pre',
    truncating='pre', value=0.0
    )
    
    grid_search.fit(X_train_w2v_pad, y_train)
    print("RegressionLogistic: with countvectorizer")
    print("done in %0.3fs" % (time() - t0))
    print()
        
    print("Train score with best_estimator_: %0.3f" % grid_search.best_estimator_.score(X_train_w2v_pad, y_train))
    print("Test score with best_estimator_: %0.3f" % grid_search.best_estimator_.score(X_test_w2v_pad, y_test))
    print("\n")
    print("Classification Report Test Data")
    print(classification_report(y_test, grid_search.best_estimator_.predict(X_test_w2v_pad)))
    """
    


    
    
    
