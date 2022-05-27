# -*- coding: utf-8 -*-
"""
Created on Fri May 27 11:12:50 2022

@author: CSU5KOR
"""
import os
import string
import pickle
import pandas as pd
from fastapi import FastAPI
from gensim import corpora,models
from  nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

punctuations=string.punctuation
stop_words=stopwords.words('english')
Stemmer=PorterStemmer()
#Lemmatizer=WordNetLemmatizer()
stopwords_verbs = ['say', 'get', 'go', 'know', 'may', 'need', 'like', 'make', 'see', 'want', 'come', 'take', 'use', 'would', 'can']
stopwords_other = ['one', 'mr', 'bbc', 'image', 'getty', 'de', 'en', 'caption', 'also', 'copyright', 'something']
all_stopwords=stopwords_other+stopwords_verbs+stop_words

def text_preprocessor(text,StopWords,stemming=True,return_tokens=True):
    words=word_tokenize(text)
    words=list(map(lambda x: x.lower(),words))
    words=list(map(lambda x: x.strip(),words))
    words=[i for i in words if i not in StopWords]
    if stemming==True:
        words=list(map(lambda x: Stemmer.stem(x),words))
    else:
        words=list(map(lambda x: Lemmatizer.lemmatize(x),words))
    words=list(map(lambda x:x.translate(str.maketrans('','',punctuations)),words))
    words=[i for i in words if i!='']
    words=[i for i in words if not i.isdigit()]
    final_text=' '.join([i for i in words if not i.isdigit()])
    if return_tokens==True:
        return words
    else:
        return final_text

app=FastAPI()

@app.get("/topic-modelling-api-english")

def model_predict(text=None):
    if text is None:
        output="no text given"
    else:
        data_dir=r'C:\Users\CSU5KOR\Desktop\work\BMW-Phase-2\Codes'
        tokens_new=text_preprocessor(text,StopWords=all_stopwords)

        dictionary_LDA_new=corpora.Dictionary([tokens_new])

        corpus_new=[dictionary_LDA_new.doc2bow(tokens_new)]
        #load model
        file_name='bbc_topic_model.pkl'
        with open(os.path.join(data_dir,file_name), 'rb') as f:
              bbc_model = pickle.load(f)
        test_string_1=bbc_model[corpus_new[0]]
        test_string=bbc_model.show_topic(test_string_1[0][0])
        word_list=[]
        prob_list=[]
        for i,prob in test_string:
            word_list.append(i)
            prob_list.append(prob)
        word_list=pd.DataFrame(word_list)
        prob_list=pd.DataFrame(prob_list)
        keyword=pd.concat([word_list,prob_list],axis=1)
        keyword.columns=['keywords','scores']
        output=keyword
    return output
            