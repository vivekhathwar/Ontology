import os
import pickle
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize,sent_tokenize
from scipy import spatial

def docvec(document):
    #Fitting word2vec
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
    model1 = Word2Vec(data, min_count = 1,size = 100)

    #Getting document vector
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
    vector=vecsum/count
    return vector

#Reading test document
testdoc= open("TEST/Yashpal-committee-report.txt","r",encoding="utf-8").read()

#Finding cosine similarity
def cossim(testvec,comp):
    return(1-spatial.distance.cosine(testvec,comp))

###################################################################################################
"""Level"""
dictionary=pickle.load(open("Level/dictionary.pickle","rb"))
tf_idf=pickle.load(open("Level/tf_idf.pickle","rb"))
testvec=docvec(testdoc)
print("Level")
#Bachelors
print("Baccalaureate")
ba1=open("Level/Baccalaureate/bachelors.txt",encoding="utf-8").read()
ba2=open("Level/Baccalaureate/bachelor1.txt",encoding="utf-8").read()
ba3=open("Level/Baccalaureate/bachelor2.txt",encoding="utf-8").read()
ba1v=docvec(ba1)
ba2v=docvec(ba2)
ba3v=docvec(ba3)
print(cossim(testvec,ba1v),end="\t")
print(cossim(testvec,ba2v),end="\t")
print(cossim(testvec,ba3v),end="\t")
#Continuing
print("\nContinuing")
con1=open("Level/Continuing/continuing1.txt",encoding="utf-8").read()
con2=open("Level/Continuing/continuing2.txt",encoding="utf-8").read()
con3=open("Level/Continuing/continuing3.txt",encoding="utf-8").read()
con4=open("Level/Continuing/continuing4.txt",encoding="utf-8").read()
con1v=docvec(con1)
con2v=docvec(con2)
con3v=docvec(con3)
con4v=docvec(con4)
print(cossim(testvec,con1v),end="\t")
print(cossim(testvec,con2v),end="\t")
print(cossim(testvec,con3v),end="\t")
print(cossim(testvec,con4v),end="\t")
#Masters
print("\nMasters")
mas1=open("Level/Masters/masters.txt",encoding="utf-8").read()
mas2=open("Level/Masters/masters1.txt",encoding="utf-8").read()
mas1v=docvec(mas1)
mas2v=docvec(mas2)
print(cossim(testvec,mas1v),end="\t")
print(cossim(testvec,mas2v),end="\t")
#Doctoral
print("\nDoctoral")
doc1=open("Level/Doctoral/Doctoral1.txt",encoding="utf-8").read()
doc2=open("Level/Doctoral/Doctoral2.txt",encoding="utf-8").read()
doc3=open("Level/Doctoral/Doctoral3.txt",encoding="utf-8").read()
doc4=open("Level/Doctoral/Doctoral4.txt",encoding="utf-8").read()
doc5=open("Level/Doctoral/Doctoral5.txt",encoding="utf-8").read()
doc1v=docvec(doc1)
doc2v=docvec(doc2)
doc3v=docvec(doc3)
doc4v=docvec(doc4)
doc5v=docvec(doc5)
print(cossim(testvec,doc1v),end="\t")
print(cossim(testvec,doc2v),end="\t")
print(cossim(testvec,doc3v),end="\t")
print(cossim(testvec,doc4v),end="\t")
print(cossim(testvec,doc5v),end="\t")
###################################################################################################
"""Policies"""
print("\n\nPolicies")
dictionary=pickle.load(open("policies/dictionary.pickle","rb"))
tf_idf=pickle.load(open("policies/tf_idf.pickle","rb"))
testvec=docvec(testdoc)
#Funding
print("Funding")
fun1=open("policies/Funding/funding.txt",encoding="utf-8").read()
fun2=open("policies/Funding/funding1.txt",encoding="utf-8").read()
fun3=open("policies/Funding/funding2.txt",encoding="utf-8").read()
fun4=open("policies/Funding/funding3.txt",encoding="utf-8").read()
fun1v=docvec(fun1)
fun2v=docvec(fun2)
fun3v=docvec(fun3)
fun4v=docvec(fun4)
print(cossim(testvec,fun1v),end="\t")
print(cossim(testvec,fun2v),end="\t")
print(cossim(testvec,fun3v),end="\t")
print(cossim(testvec,fun4v),end="\t")
#Governance
print("\nGovernance")
gov1=open("policies/Governance/governance1.txt",encoding="utf-8").read()
gov2=open("policies/Governance/governance2.txt",encoding="utf-8").read()
gov3=open("policies/Governance/governance3.txt",encoding="utf-8").read()
gov1v=docvec(gov1)
gov2v=docvec(gov2)
gov3v=docvec(gov3)
print(cossim(testvec,gov1v),end="\t")
print(cossim(testvec,gov2v),end="\t")
print(cossim(testvec,gov3v),end="\t")
#Pesonnel
print("\nPersonnel")
per1=open("policies/personnel/personnel1.txt",encoding="utf-8").read()
per2=open("policies/personnel/personnel2.txt",encoding="utf-8").read()
per1v=docvec(per1)
per2v=docvec(per2)
print(cossim(testvec,per1v),end="\t")
print(cossim(testvec,per2v),end="\t")
#Temporal
print("\nTemporal")
tem1=open("policies/Temporal/temporal1.txt",encoding="utf-8").read()
tem2=open("policies/Temporal/temporal2.txt",encoding="utf-8").read()
tem1v=docvec(tem1)
tem2v=docvec(tem2)
print(cossim(testvec,tem1v),end="\t")
print(cossim(testvec,tem2v),end="\t")
###################################################################################################
"""Outcomes"""
print("\n\nOutcomes")
dictionary=pickle.load(open("outcomes/dictionary.pickle","rb"))
tf_idf=pickle.load(open("outcomes/tf_idf.pickle","rb"))
testvec=docvec(testdoc)
#Cultural
print("Cultural")
cul1=open("outcomes/Cultural/Cultural1.txt",encoding="utf-8").read()
cul2=open("outcomes/Cultural/Cultural2.txt",encoding="utf-8").read()
cul3=open("outcomes/Cultural/Cultural3.txt",encoding="utf-8").read()
cul4=open("outcomes/Cultural/Cultural4.txt",encoding="utf-8").read()
cul1v=docvec(cul1)
cul2v=docvec(cul2)
cul3v=docvec(cul3)
cul4v=docvec(cul4)
print(cossim(testvec,cul1v),end="\t")
print(cossim(testvec,cul2v),end="\t")
print(cossim(testvec,cul3v),end="\t")
print(cossim(testvec,cul4v),end="\t")
#Economic
print("\nEconomic")
eco1=open("outcomes/Economic/eco1.txt",encoding="utf-8").read()
eco2=open("outcomes/Economic/eco2.txt",encoding="utf-8").read()
eco3=open("outcomes/Economic/eco3.txt",encoding="utf-8").read()
eco4=open("outcomes/Economic/eco4.txt",encoding="utf-8").read()
eco5=open("outcomes/Economic/eco5.txt",encoding="utf-8").read()
eco6=open("outcomes/Economic/eco6.txt",encoding="utf-8").read()
eco7=open("outcomes/Economic/eco7.txt",encoding="utf-8").read()
eco1v=docvec(eco1)
eco2v=docvec(eco2)
eco3v=docvec(eco3)
eco4v=docvec(eco4)
eco5v=docvec(eco5)
eco6v=docvec(eco6)
eco7v=docvec(eco7)
print(cossim(testvec,eco1v),end="\t")
print(cossim(testvec,eco2v),end="\t")
print(cossim(testvec,eco3v),end="\t")
print(cossim(testvec,eco4v),end="\t")
print(cossim(testvec,eco5v),end="\t")
print(cossim(testvec,eco6v),end="\t")
print(cossim(testvec,eco7v),end="\t")
#Scientific
print("\nScientific")
sci1=open("outcomes/Scientific/Scientific1.txt",encoding="utf-8").read()
sci2=open("outcomes/Scientific/Scientific2.txt",encoding="utf-8").read()
sci3=open("outcomes/Scientific/Scientific3.txt",encoding="utf-8").read()
sci1v=docvec(sci1)
sci2v=docvec(sci2)
sci3v=docvec(sci3)
print(cossim(testvec,sci1v),end="\t")
print(cossim(testvec,sci2v),end="\t")
print(cossim(testvec,sci3v),end="\t")
#Technical
print("\nTechnical")
tec1=open("outcomes/Technical/Technical1.txt",encoding="utf-8").read()
tec3=open("outcomes/Technical/Technical3.txt",encoding="utf-8").read()
tec4=open("outcomes/Technical/Technical4.txt",encoding="utf-8").read()
tec1v=docvec(tec1)
tec3v=docvec(tec3)
tec4v=docvec(tec4)
print(cossim(testvec,tec1v),end="\t")
print(cossim(testvec,tec3v),end="\t")
print(cossim(testvec,tec4v),end="\t")
###################################################################################################
