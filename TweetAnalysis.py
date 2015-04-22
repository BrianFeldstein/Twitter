# -*- coding: utf-8 -*-
"""
Created on Wed Apr 08 11:56:57 2015

@author: Brian
"""

from __future__ import division
import cPickle
import pickle
from collections import Counter
import numpy as np
import gc
import time
from MovieReviews3 import alphabet, sigmoid, ProcessX, CreateX, Threshold

"""
This was used as part of an ipython notebook to look at the Power Spectrum of Tweets
"""
def RevTimes(FName):

    with open("MyReviews2/" + FName + " FULL", "rb") as f:
        Name, tso = cPickle.load(f)

    TweetSet = [i['text'] for i in tso]
    TimeSet = [i['created_at'] for i in tso]
    
    del tso
    gc.collect()


    avoid_words = ["rt", "trailer", "review"]
    
    with open("ThetaRes0326", "rU") as Th, open("FeatureWords0326", "rU") as FW, open("featureMeans0326", "rU") as fM, open("featureSTD0326", "rU") as fS:        
        ThetaRes = pickle.load(Th)
        FeatureWords = pickle.load(FW)
        featureMeans = pickle.load(fM)
        featureSTD = pickle.load(fS)
    

    WordsInName = ("".join([i.lower() for i in Name if i in alphabet])).split(" ")    
    

    Words = [''.join([j.lower() for j in i if j in alphabet]) for i in TweetSet]
    WordsR = [";".join([j for j in i.split(" ") if len(j)>2 and j[0] != "@" and j[0:4] != "http" and j not in WordsInName and j != ''.join(i for i in Name if i != ' ') and j!= 'movie']) for i in Words]
    TweetCounter = Counter(WordsR)
    
    keep_list = [i for i in range(len(WordsR)) if TweetCounter[WordsR[i]] == 1 and "looks" not in WordsR[i] and "sounds" not in WordsR[i] and not [j for j in avoid_words if j in Words[i]] and not [j for j in Words[i].split() if j[0:4] == "http"]]
    
    WordsLessSpam = np.array(WordsR)[keep_list]
    TimesLessSpam = np.array(TimeSet)[keep_list]

    del WordsR
    del TimeSet
    gc.collect()


    M = [i.split(";") + [i.split(";")[j] + " " + i.split(";")[j+1] for j in range(len(i.split(";")) - 1)] for i in WordsLessSpam]    

    del WordsLessSpam
    gc.collect()    
    
    res = sigmoid( np.dot(ProcessX(CreateX(M, FeatureWords), featureMeans, featureSTD), ThetaRes))
    
    del M
    gc.collect()   
    print "almost"
    
    AllTimes = [time.mktime(time.strptime(TimesLessSpam[i], '%a %b %d %H:%M:%S +0000 %Y')) for i in range(len(TimesLessSpam))]
    PosTimes = [time.mktime(time.strptime(TimesLessSpam[i], '%a %b %d %H:%M:%S +0000 %Y')) for i in range(len(TimesLessSpam)) if res[i] > Threshold]
    NegTimes = [time.mktime(time.strptime(TimesLessSpam[i], '%a %b %d %H:%M:%S +0000 %Y')) for i in range(len(TimesLessSpam)) if res[i] < 1-Threshold]
    #print [len(res[res>Threshold]), len(res[res<1-Threshold])]
    #score = len(res[res>Threshold]) / (len(res[res<1-Threshold]) + len(res[res>Threshold]))
    #print 'error: ', score*(1/np.sqrt(len(res[res>Threshold])) - 1/np.sqrt(len(res[res<1-Threshold]) + len(res[res>Threshold])))
    return (AllTimes, PosTimes, NegTimes)

