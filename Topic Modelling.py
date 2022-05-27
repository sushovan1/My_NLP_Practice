# -*- coding: utf-8 -*-
"""
Created on Thu May 26 13:47:36 2022

@author: CSU5KOR
"""

#topic modelling
#https://towardsdatascience.com/the-complete-guide-for-topics-extraction-in-python-a6aaa6cedbbc
#https://github.com/FelixChop/MediumArticles/blob/master/LDA-BBC.ipynb
#language_list-https://en.wikipedia.org/wiki/List_of_ISO_639-1_codes
#https://www.baeldung.com/cs/topic-modeling-coherence-score
#https://ieeexplore.ieee.org/document/8259775
#https://neptune.ai/blog/pyldavis-topic-modelling-exploration-tool-that-every-nlp-data-scientist-should-know
import nltk
#nltk.set_proxy('http://rb-proxy-de.bosch.com:8080',('CSU5KOR','Boschwork%402027'))
#nltk.set_proxy('http://csu5kor:Boschwork%402027@rb-proxy-de.bosch.com:8080')
#nltk.download('wordnet')
import os
import string
import numpy as np
import pickle
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
import pandas as pd
from langdetect import detect
from gensim import corpora,models
from gensim.models import Phrases
from gensim.models.coherencemodel import CoherenceModel
####################################################################################
punctuations=string.punctuation
stop_words=stopwords.words('english')
Stemmer=PorterStemmer()
Lemmatizer=WordNetLemmatizer()
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
######################################################################################
data_dir=r'C:\Users\CSU5KOR\Desktop\work\BMW-Phase-2\Codes'
data = pd.read_csv(os.path.join(data_dir,'Topic_modelling_data_bbc.csv'))

data=data.dropna().reset_index(drop=True)
language_count=data.groupby('lang').size()

data['detected_lang']=data.apply(lambda row: detect(row['articles']),axis=1)
detected_lang_count=data.groupby('detected_lang').size()
#select english only
english_text_data=data[data['detected_lang']=='en']

english_text_data['processed_texts']=english_text_data.apply(lambda row: text_preprocessor(row['articles'],all_stopwords,stemming=True,return_tokens=True),axis=1)

tokens = english_text_data['processed_texts'].tolist()
bigram_model = Phrases(tokens)
trigram_model = Phrases(bigram_model[tokens], min_count=1)
tokens = list(trigram_model[bigram_model[tokens]])

dictionary_LDA = corpora.Dictionary(tokens)
dictionary_LDA.filter_extremes(no_below=3)
corpus = [dictionary_LDA.doc2bow(tok) for tok in tokens]

np.random.seed(123456)
num_topics = 20
lda_model = models.LdaModel(corpus, num_topics=num_topics, \
                                  id2word=dictionary_LDA, \
                                  passes=20, alpha=[0.01]*num_topics, \
                                  eta=[0.01]*len(dictionary_LDA.keys()))

for i,topic in lda_model.show_topics(formatted=True, num_topics=num_topics, num_words=20):
    print(str(i)+": "+ topic)
    print()
    
import pyLDAvis
import pyLDAvis.gensim_models as gensimvis
lda_viz = gensimvis.prepare(topic_model=lda_model, corpus=corpus, dictionary=dictionary_LDA,sort_topics=False)
#pyLDAvis.enable_notebook()
pyLDAvis.save_html(lda_viz, os.path.join(data_dir,'lda_2.html'))
#####################################################################################
coherence_model_lda = CoherenceModel(model=lda_model, texts=tokens, dictionary=dictionary_LDA, coherence='u_mass')
coherence_lda = coherence_model_lda.get_coherence()
print('\\nCoherence Score: ', coherence_lda)
#######################################################################################
#prediction
new_text="Russian President Vladimir Putin urged Eurasian Economic Union partners to choose natural partners and neighbors over other well developed economies on Thursday.\
Putin spoke via video link at the Eurasian Economic Union (EAEU) forum opening in Bishkek. He added there are more than 180 projects worth $300 billion on the EAEU forum agenda.\
EAEU is an economic union of post-Soviet economies initiated by Russia in 2015 to create a single market for Russia, Belarus, Kazakhstan, Armenia and Kyrgyzstan."

tokens_new=text_preprocessor(new_text,StopWords=all_stopwords)

dictionary_LDA_new=corpora.Dictionary([tokens_new])

corpus_new=[dictionary_LDA_new.doc2bow(tokens_new)]
lda_model[corpus_new[0]]
#########################################################################################
file_name='bbc_topic_model.pkl'

with open(os.path.join(data_dir,file_name),'wb') as f:
    pickle.dump(lda_model,f)
#load model
with open(os.path.join(data_dir,file_name), 'rb') as f:
      bbc_model = pickle.load(f)
  
bbc_model[corpus_new[0]]
#################################################################################
#for API creation
test_string_1=[(16, 0.996138)]
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
