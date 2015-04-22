# -*- coding: utf-8 -*-
"""
Created on Tue Feb 03 18:58:10 2015

@author: Brian
"""

from __future__ import division
import TwitterSearch as TSch
import pickle
import sys
import os
scriptpath = "../TPass"
sys.path.append(os.path.abspath(scriptpath))
from TwitterPass import TwitterPass

alphabet = 'QWERTYUIOPASDFGHJKLZXCVBNM qwertyuiopasdfghjklzxcvbnm'

"""
This searches for tweets including Name, as well as the word movie, and dumps
the tweet data into the file FName 
"""
def MakeTweetFile(Name, FName):
    print Name
    try:
        tso = TSch.TwitterSearchOrder() # create a TwitterSearchOrder object
        #tso.set_keywords(['#'+ ''.join(i for i in Name if i != ' '), 'movie']) # let's define all words we would like to have a look for
        tso.set_keywords([Name , 'OR',  '#'+ ''.join(i for i in Name if i != ' '), 'movie'])
        tso.set_language('en')
        print tso.create_search_url()
        
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
 
    except TSch.TwitterSearchException as e: 
        print(e)
