# -*- coding: utf-8 -*-
"""
Created on Wed May 25 11:26:07 2022

@author: CSU5KOR
"""
import string
import os
import numpy as np
import pandas as pd
import spacy
from spacy_langdetect import LanguageDetector
#topic modelling
#https://towardsdatascience.com/the-complete-guide-for-topics-extraction-in-python-a6aaa6cedbbc
data_dir=r'C:\Users\CSU5KOR\Desktop\work\BMW-Phase-2\Codes'
reviews=pd.read_csv(os.path.join(data_dir,'Text_classification_data.csv'))

# Extract desired columns and view the dataframe 
reviews_subset= reviews[['Review Text','Recommended IND','Class Name']].dropna()
#reviews.head(10)
unique_classes=list(set(reviews_subset['Class Name'].tolist()))

count_classes=pd.DataFrame(reviews_subset.groupby(['Class Name']).size())
count_classes['class_name']=count_classes.index
count_classes.columns=['Count','class_name']
count_classes_subset=count_classes['class_name'][count_classes['Count']>=500].tolist()

reviews_subset=reviews_subset[reviews_subset['Class Name'].isin(count_classes_subset)]

nlp=spacy.load(r'C:\Users\CSU5KOR\Desktop\work\BMW-Phase-2\Codes\dist\en_core_web_sm-3.2.0\en_core_web_sm\en_core_web_sm-3.2.0')

test_doc=nlp(reviews_subset['Review Text'].tolist()[0])

entity=test_doc.ents
json_text=test_doc.to_json()
nlp.pipe_names
#############################################################################################
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.pipeline import Pipeline
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
###############################################################################################
punctuations=string.punctuation
#punctuations=punctuations.split('')
stop_words=list(spacy.lang.en.stop_words.STOP_WORDS)
Stemmer=PorterStemmer()
def text_preprocessor(text):
    words=word_tokenize(text)
    words=list(map(lambda x: x.lower(),words))
    words=list(map(lambda x: x.strip(),words))
    words=[i for i in words if i not in stop_words]
    words=list(map(lambda x: Stemmer.stem(x),words))
    words=list(map(lambda x:x.translate(str.maketrans('','',punctuations)),words))
    words=[i for i in words if i!='']
    final_text=' '.join([i for i in words if not i.isdigit()])
    return final_text
reviews_subset['processed_text']=reviews_subset.apply(lambda row : text_preprocessor(row['Review Text']),axis=1)
corpus=reviews_subset['processed_text'].tolist()
vectorizer = TfidfVectorizer()
x=vectorizer.fit_transform(corpus)
features=vectorizer.get_feature_names()
y=np.array(reviews_subset['Recommended IND'])
x=x.toarray()
##################################################################################################
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
model=LogisticRegression()
model.fit(x_train,y_train)
y_pred=model.predict(x_test)

precision_recall_fscore_support(y_test,y_pred)
##################################################################################################
#prediction
new_text='I do not like this dress'
processed_text=text_preprocessor(new_text)
x_new=vectorizer.transform([processed_text])
x_new=x_new.toarray()
model.predict(x_new)
###################################################################################################