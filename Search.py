# -*- coding: utf-8 -*-
"""
Created on Tue Feb 03 18:58:10 2015

@author: Brian
"""

from __future__ import division
#import numpy as np
import TwitterSearch as TSch
#from datetime import date
import pickle
import sys
import os
#from collections import Counter
scriptpath = "../TPass"
#scriptpath = "C:\Users\Brian\Desktop\CodeForGit\TPass"
sys.path.append(os.path.abspath(scriptpath))
from TwitterPass import TwitterPass

alphabet = 'QWERTYUIOPASDFGHJKLZXCVBNM qwertyuiopasdfghjklzxcvbnm'

def MakeTweetFile(Name, FName):
    print Name
    #WordsInName = ("".join([i.lower() for i in Name if i in alphabet])).split(" ")
    try:
        tso = TSch.TwitterSearchOrder() # create a TwitterSearchOrder object
        #tso.set_keywords(['#'+ ''.join(i for i in Name if i != ' '), 'movie']) # let's define all words we would like to have a look for
        tso.set_keywords([Name , 'OR',  '#'+ ''.join(i for i in Name if i != ' '), 'movie']) # let's define all words we would like to have a look for
        tso.set_language('en') # we want to see German tweets only
        #tso.set_until(date(2015, 2, 1))    
        #tso.set_include_entities(False) # and don't give us all those entity information
        print tso.create_search_url()
        # it's about time to create a TwitterSearch object with our secret tokens
        ts = TSch.TwitterSearch(
            consumer_key = TwitterPass['ck'],
            consumer_secret = TwitterPass['cs'],
            access_token = TwitterPass['at'],
            access_token_secret = TwitterPass['ats'])
        
        Tweets = []
        for i in ts.search_tweets_iterable(tso):
            Tweets.append(i)
         
        with open("MyReviews2/" + FName + " FULL", "wb") as f:
            pickle.dump([Name, Tweets], f)
 
    except TSch.TwitterSearchException as e: # take care of all those ugly errors if there are some
        print(e)
