# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 08:45:03 2015

@author: Brian
"""

from __future__ import division
import numpy as np
import pickle
from MovieData import MovieData, r, CrRating, PubRating

GoodWords = ['good', 'amazing', 'excellent', 'best', 'great', 'fun', 'exciting', 'worth', 'well', 'beautiful', 'awesome', 'enjoyable', 'interesting', 'perfect', 'wonderful', 'better', 'favorite', 'brilliant', 'strong', 'superb', 'highlyrecommended', 'verygood', 'finest', 'loved']
BadWords = ['bad', 'poor', 'terrible', 'worst', 'crap', 'boring', 'awful', 'mistake', 'fake', 'ugly', 'slow', 'tedious', 'uninteresting', 'waste', 'notworth', 'nothing', 'dull', 'worse', 'horrible', 'lame', 'annoying', 'stupid', 'weak', 'dissapointing', 'mess', 'dreadful', 'predictable', 'badly']

def TrivRev(FName):
    
    A = pickle.load(open("MyReviews\\" + FName, "rb"))
    M = [i.split(";") for i in A]
    GoodCount = 0
    BadCount = 0
    for i in M:
        for j in i:
            if j in GoodWords: GoodCount +=1
            if j in BadWords: BadCount +=1
    return GoodCount/(GoodCount+BadCount)
    
scores = []    
for i in MovieData:
    scores.append(TrivRev(i[0]))
    
        
    