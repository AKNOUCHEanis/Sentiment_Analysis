# -*- coding: utf-8 -*-

import pandas as pd
import re
import emoji
import string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np

class DataPreparation():
    
    def __init__(self):
        pass
    
    def clean_text(self, df, col, target):
        
        data_cleaned=[]
        
        for i in range(df[col].count()):
            text = df.iloc[i][col]
            text = re.sub(r'@\w+','',text)  #supprime les mentions
            text = re.sub(r'http.?://[^\s]+[\s]?', '', text)  #supprime les urls
            text = emoji.demojize(text)
            text = text.replace('_','')
            
            punct = string.punctuation
            trantab = str.maketrans(punct, len(punct)*' ')  # Every punctuation symbol will be replaced by a space
            text = text.translate(trantab)
            
            text = re.sub(r'\d+', '', text)
            text = text.lower()
            
            #Remove stop words
            stopwords_list = stopwords.words('english')
            
            # Some words which might indicate a certain sentiment are kept via a whitelist
            whitelist = ["n't", "not", "no"]
            words = text.split() 
            clean_words = [word for word in words if ( word not in stopwords_list or word in whitelist) and len(word) > 1] 
            text = " ".join(clean_words) 
            
            #stemming
            porter = PorterStemmer()
            words = text.split() 
            stemmed_words = [porter.stem(word) for word in words]
            text = " ".join(stemmed_words)
            
            #words to delete after stemming
            stopwords_list2 = ['aa', 'ty', 'ye']
            words = text.split() 
            clean_words = [word for word in words if ( word not in stopwords_list2) and len(word) > 1] 
            text = " ".join(clean_words) 
            
            data_cleaned.append([text, df.iloc[i][target]])
            
            
        return pd.DataFrame(data_cleaned, columns=['text', 'airline_sentiment'])
    
    def countVectorizer(self, df, col, target):
        
        cv = CountVectorizer()
        corpus = df[col].values.tolist()
        X = cv.fit_transform(corpus)
        bow = cv.get_feature_names()
        y=df[target].tolist()
        y_num=[]
        for i in range(len(y)):
            if y[i]=="negative":
                y_num.append(-1)
            elif y[i]=="positive":
                y_num.append(1)
            elif y[i]=="neutral":
                y_num.append(0)
            else:
                print("Erreur dans y !")
                
                
        
        return X.toarray(), y_num, bow 
        
        
         
    def compute_avg_w2v_vector(self, model, tweet, SIZE):
        list_of_word_vectors = [model.wv[w] for w in tweet if w in model.wv.vocab.keys()]
        
        if len(list_of_word_vectors) == 0:
            result = [0.0]*SIZE
        else:
            result = np.sum(list_of_word_vectors, axis=0) / len(list_of_word_vectors)
            
        return result

    def word2vec(self, model, data, SIZE):
        vectors=[]
        for i in range(len(data)):
          vectors.append(self.compute_avg_w2v_vector(model, data[i], SIZE))
        
        return vectors
        
        
        
        
        
        
        
        
        