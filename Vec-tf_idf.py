# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 22:14:57 2019

@author: HATHWAR
"""

from gensim.models import Word2Vec
from scipy import spatial
import pickle
from nltk.tokenize import sent_tokenize, word_tokenize 
import warnings 

import gensim
print(dir(gensim))
raw_documents=[]
docs=[]
#Corpus generation
docs.append(open("test2eco.txt",encoding="utf-8")) 
docs.append(open("test1eco.txt",encoding="utf-8"))
docs.append(open("fixedeco.txt",encoding="utf-8"))
#docs.append(open("test3eco.txt",encoding="utf-8"))
docs.append(open("test4eco.txt",encoding="utf-8"))
for ele in docs:
    ff=ele.read().strip()
    ele.close()
    raw_documents.append(ff)

gen_docs = [[w.lower() for w in word_tokenize(text)] 
            for text in raw_documents]
dictionary = gensim.corpora.Dictionary(gen_docs)
warnings.filterwarnings(action = 'ignore')
for i in range(len(dictionary)):
    corpus = [dictionary.doc2bow(gen_doc) for gen_doc in gen_docs]
tf_idf = gensim.models.TfidfModel(corpus)
dictionary = gensim.corpora.Dictionary(gen_docs)

#Query doc
document = open("test4eco.txt","r",encoding="utf-8").read()

query_doc = [w.lower() for w in word_tokenize(document.strip())]
query_doc_bow = dictionary.doc2bow(query_doc)
query_doc_tf_idf = tf_idf[query_doc_bow]
data=[]
for i in sent_tokenize(document): 
    temp = []
    # tokenize the sentence into words 
    for j in word_tokenize(i): 
        temp.append(j.lower()) 
  
    data.append(temp)

#Document vector generation
model1 = Word2Vec(data, min_count = 1,size = 100)
count=0
vecsum=[]
for i in range(0,100):
    vecsum.append(0)
for tup in query_doc_tf_idf:
    try:
        vecsum=vecsum+model1[dictionary[tup[0]]]*tup[1]
        count=count+1
    except:
        continue

test4eco=vecsum/count
pickle.dump(test4eco,open("test4eco.pickle","rb"))
test2eco=vecsum/count
test1eco=vecsum/count
eco=vecsum/count
1-spatial.distance.cosine(test4eco,eco)
