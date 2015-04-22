# -*- coding: utf-8 -*-
"""
Created on Sun Feb 01 03:54:40 2015

@author: Brian
"""
#Data from: http://www.cs.cornell.edu/people/pabo/movie-review-data/review_polarity.tar.gz
# and more from http://ai.stanford.edu/~amaas/data/sentiment/


from __future__ import division
import numpy as np
from collections import Counter
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from Search import MakeTweetFile
import gc
import pickle
import cPickle

"""
This is the main module, and it does two things:
MakeThetaAndFeatureWords determines the popular words to use
as features by counting words (and word pairs) from the Review
data set.  It then trains a logistic regression algoritm to classify the 
sentiment of the reviews based on the counts of those words.
The Feature Words and Regression coefficients are dumped to various files.
"""


KeepNum = 1000
Threshold = .999#.9975 #changed before 0225
lam = 2#15000 #regularization parameter for logistic regression
Popular = 12000 #Number of "popular" words to use as features from positive and negative reviews
NegFileNums = range(1, 20001)
PosFileNums = range(1, 20001)
NegFileNumsV = range(20001, 25000)
PosFileNumsV = range(20001, 25000)

alphabet = 'QWERTYUIOPASDFGHJKLZXCVBNM qwertyuiopasdfghjklzxcvbnm'


"""
Given a list of Review File numbers, this returns a list of 1D arrays
of words from each review.. either from the set of positive reviews or negative
reviews in the training set, depending on "posorneg".
"""
def WordsByReview(FileNums, posorneg):

    WordSet = [[''] for i in FileNums]
    
    for j in range(len(FileNums)):
        with open("Reviews/" + posorneg + "/" + posorneg + " (" + str(FileNums[j]) +").txt") as myfile:
            words = ("".join(line.rstrip() for line in myfile))
            words = ''.join(i.lower() for i in words if  i in 'qwertyuiopasdfghjklzxcvbnm QWERTYUIOPASDFGHJKLZXCVBNM')
            words = words.split(' ')
            words = np.array([i for i in words if len(i) > 2 and i != 'the'])
            pairwords = np.array(  [words[i] + " " + words[i+1]  for i in range(len(words)-1)]  )
        WordSet[j] = np.hstack((words, pairwords))
    return WordSet


"""
This takes a list of 1D arrays of words from reviews, and a list of "featurewords"
and returns an array where the rows correspond to each review, and the columns
are the number of appearances of each featureword, divided by the number of
words in the review. 
"""
def CreateX(Words, FeatureWords):
    Xresult = np.zeros((len(Words), len(FeatureWords)))
    for i in range(len(Words)):
        ReviewWordCounter = Counter(Words[i])
        Xresult[i, :] = np.array([ReviewWordCounter[j]/len(Words[i]) for j in FeatureWords])
    return Xresult
    
"""
This takes a feature array, and performs mean subtraction, and normalization.
"""
def ProcessX(Xtry, featureMeans, featureMax):
    XtryN = Xtry
    for i in range(XtryN.shape[1]):
        XtryN[:,i] = Xtry[:,i] - featureMeans[i]
        XtryN[:,i] = XtryN[:,i]/featureMax[i]
    del Xtry
    gc.collect()
    XtryN = np.hstack((np.ones((XtryN.shape[0], 1)), XtryN ))
    return XtryN  
    
def sigmoid(z):
    return 1/(1 + np.exp(-z))

"""
This computes the logistic regression cost function and derivatives for use in
scipy.optimize.minimize.
"""
def CostAndGrad(Theta, Xvalues, Yvalues):
    m = Xvalues.shape[0]
    predictions = sigmoid( np.dot(Xvalues, Theta))

    #regularize_weight = np.max(Xvalues, axis = 0, keepdims = True).transpose()
    #np.array([np.max(Xvalues[:,i]) for i in range(Xvalues.shape[1])]).reshape(Theta.shape[0], Theta.shape[1])

    #Cost =   (-1/m)*  np.sum( (1- Yvalues)*np.log(np.abs(1- predictions)+.0000001) + Yvalues*np.log(np.abs(predictions) + .0000001) ) + (lam/(2*m))*np.sum((Theta[1:,:]*regularize_weight[1:,:])**2) 
    #Grad = (1/m)* (np.dot(np.transpose(Xvalues),(predictions - Yvalues))) + (lam/m) * np.vstack(  (np.array([[0]]) , Theta[1:, :]*regularize_weight[1:,:]**2) )
    Cost =   (-1/m)*  np.sum( (1- Yvalues)*np.log(np.abs(1- predictions)+.0000001) + Yvalues*np.log(np.abs(predictions) + .0000001) ) + (lam/(2*m))*np.sum(Theta[1:,:]**2) 
    Grad = (1/m)* (np.dot(np.transpose(Xvalues),(predictions - Yvalues))) + (lam/m) * np.vstack(  (np.array([[0]]) , Theta[1:, :]) )  

    return (Cost, Grad)

"""
This was used for gradient descent, but is not being used anymore in favor of scipy's
optimize.minimize.
"""
def GradDesc(Cost, Grad, x0, eps = .1, Trials = 1000):
    x = x0    
    for i in range(Trials):
        x = x - eps * Grad(x)
        if i%100 ==0:print Cost(x)
    return x
        
        
"""
This trains the logistic regression algorithm.  It dumps the regression coefficients
"Theta", as well as the feature means, feature maxima, and feature words.
It also dumps a list "BestWords" of the KeepNum words with the highest regression coefficients.
If UseBest = True, the "BestWords" found in a previous run will be used
as the featureWords.  Otherwise, featureWords come from the most common
words and word pairs in the review data set.
If plot = true, a plot is made of the words with the highest regression coefficients.
"""
def MakeThetaAndFeatureWords(UseBest = False, plot = False):

    NegWords = WordsByReview(NegFileNums, "neg")
    PosWords = WordsByReview(PosFileNums, "pos")
    NegWordsV = WordsByReview(NegFileNumsV, "neg")
    PosWordsV = WordsByReview(PosFileNumsV, "pos")
    
    if not UseBest:
        NegWordsAll = np.concatenate(NegWords)    
        NegWordsUnique = np.unique(NegWordsAll)    
        NegWordCounter = Counter(NegWordsAll)
        
        NegCounts = np.array([NegWordCounter[i] for i in NegWordsUnique])
        NegWordsUnique = (NegWordsUnique[NegCounts.argsort()])[::-1]
        
        PosWordsAll = np.concatenate(PosWords)    
        PosWordsUnique = np.unique(PosWordsAll)    
        PosWordCounter = Counter(PosWordsAll)
        
        PosCounts = np.array([PosWordCounter[i] for i in PosWordsUnique])
        PosWordsUnique = (PosWordsUnique[PosCounts.argsort()])[::-1]
        
        NegWordsPop = NegWordsUnique[0:Popular]
        PosWordsPop = PosWordsUnique[0:Popular]
    
    if UseBest: FeatureWords = pickle.load(open("BestWords", "rU"))   
    else: FeatureWords = np.unique(np.concatenate([NegWordsPop, PosWordsPop]))
    print FeatureWords.shape
    
    #Training Data:
    XNeg = CreateX(NegWords, FeatureWords)
    XPos = CreateX(PosWords, FeatureWords)
    YNeg = np.zeros((len(NegFileNums),1))
    YPos = np.ones((len(PosFileNums),1))
    X = np.vstack((XNeg, XPos))
    Y = np.vstack((YNeg, YPos))

    del XNeg
    del XPos
    del YNeg
    del YPos
    del NegWords
    del PosWords

    gc.collect()
    
    #Validation Data:
    XNegV = CreateX(NegWordsV, FeatureWords)
    XPosV = CreateX(PosWordsV, FeatureWords)
    YNegV = np.zeros((len(NegFileNumsV),1))
    YPosV = np.ones((len(PosFileNumsV),1))
    XV = np.vstack((XNegV, XPosV))
    YV = np.vstack((YNegV, YPosV))

    del NegWordsV
    del PosWordsV
    del XNegV
    del XPosV
    del YNegV
    del YPosV
    gc.collect()
    
    print "normalizing"
    featureMeans = [np.mean(X[:,i]) for i in range(X.shape[1])]
    for i in range(X.shape[1]):
        X[:,i] = X[:,i] - featureMeans[i]
        XV[:,i] = XV[:,i] - featureMeans[i]
    #featureSTD = [np.std(X[:,i]) for i in range(X.shape[1])]
    #for i in range(X.shape[1]):
    #    X[:,i] = X[:,i]/featureSTD[i]
    #    XV[:,i] = XV[:,i]/featureSTD[i]
    featureMax = [np.max(X[:,i]) for i in range(X.shape[1])]
    for i in range(X.shape[1]):
        X[:,i] = X[:,i]/featureMax[i]
        XV[:,i] = XV[:,i]/featureMax[i]
    print "done normalizing"
   
    if not UseBest:    
        del NegWordsAll
        del PosWordsAll
        del NegWordsUnique
        del PosWordsUnique
        del NegWordCounter
        del PosWordCounter
        del NegCounts
        del PosCounts
        del NegWordsPop
        del PosWordsPop
    gc.collect()
    
    X = np.hstack((np.ones((X.shape[0], 1)), X ))
    XV = np.hstack((np.ones((XV.shape[0], 1)), XV ))
     
    result = minimize(fun = lambda th: CostAndGrad(th.reshape((len(th), 1)), X, Y)[0], x0 = np.random.random(X.shape[1]) - .5, method = 'TNC', jac = lambda th: CostAndGrad(th.reshape((len(th),1)), X, Y)[1], options = {'maxiter':10**9})  #TNC
    ThetaRes = (result.x).reshape((len(result.x),1))   
    #ThetaRes = GradDesc(lambda th: CostAndGrad(th, X, Y)[0], lambda th: CostAndGrad(th, X, Y)[1], np.random.random((X.shape[1],1))-.5  )
       
    predictions = sigmoid( np.dot(X, ThetaRes))
    predictionsV =  sigmoid( np.dot(XV, ThetaRes))
    
    print np.sum(np.rint(predictions))/len(Y)    
    
    SuccessRate = np.sum(np.rint(predictions) == Y)/len(Y)
    SuccessRateV = np.sum(np.rint(predictionsV) == YV)/len(YV)
        
    print(len(FeatureWords), SuccessRate, SuccessRateV)
    
    with open("ThetaRes0411", "wb") as Th, open("FeatureWords0411", "wb") as FW, open("featureMeans0411", "wb") as fM, open("featureMax0411", "wb") as fS:
        pickle.dump(ThetaRes, Th)
        pickle.dump(FeatureWords, FW)
        pickle.dump(featureMeans, fM)
        pickle.dump(featureMax, fS)
        

    ThetaOrder = np.abs(ThetaRes.flatten()[1:]).argsort()
    BestWords = np.array(FeatureWords[ThetaOrder][::-1])
    BestThetas = (ThetaRes.flatten()[1:][ThetaOrder])[::-1]
    print BestWords.shape
    print BestWords[0:KeepNum].shape
    with open("BestWords", "wb") as BW:
        pickle.dump(BestWords[0:KeepNum], BW)
    #BestWordCountsPos = [PosWordCounter[i] for i in BestWords]
    #BestWordCountsNeg = [NegWordCounter[i] for i in BestWords]
    
    if plot == True:
        NumToPlot = 40
        start = 0
        plt.figure(1)
        ind = np.array(range(NumToPlot))
        width = 0.4
        p = plt.bar(ind, BestThetas[0 + start:NumToPlot + start], width, color='b')
        #p0 = plt.bar(ind, BestWordCountsNeg[0 + start:NumToPlot + start], width, color='b')
        #p1 = plt.bar(ind, BestWordCountsPos[0 + start:NumToPlot+ start], width, color='r', bottom = BestWordCountsNeg[0 + start:NumToPlot+ start])
        plt.ylabel('Occurences in Training Set')
        plt.title('Words with Highest Regression Coefficients')
        plt.xticks(ind+width/2, list(BestWords[0 + start:NumToPlot+ start]), rotation = 'vertical' )
        #plt.legend( (p1[0], p0[0]), ('Positive Reviews', 'Negative Reviews'), loc=2)
        plt.subplots_adjust(bottom=0.25)
        plt.show()
        #print BestThetas[start: start+NumToPlot]
    
    
"""
This creatures movie review scores.
if make_new[0] is false, The tweets used to create the score come from 
the previously made file FName.  Otherwise tweets are first downloaded
and dumped to the file FName before making the review.  The movie name is 
make_new[1].  If looks is true, the words "looks" or "sounds" are required to 
appear in a tweet in order for it to be counted, otherise these words
are required to not appear (The issue being whether or not we are considering 
people who have already seen the movie or ont).
"""
def ReviewMovie(FName, make_new = (False, ""), looks = False):

    avoid_words = ["rt", "trailer", "review"]
    
    #with open("ThetaRes0411", "rU") as Th, open("FeatureWords0411", "rU") as FW, open("featureMeans0411", "rU") as fM, open("featureSTD0411", "rU") as fS:        
    #    ThetaRes = pickle.load(Th)
    #    FeatureWords = pickle.load(FW)
    #    featureMeans = pickle.load(fM)
    #    featureSTD = pickle.load(fS)

    with open("ThetaRes0411", "rU") as Th, open("FeatureWords0411", "rU") as FW, open("featureMeans0411", "rU") as fM, open("featureMax0411", "rU") as fS:        
        ThetaRes = pickle.load(Th)
        FeatureWords = pickle.load(FW)
        featureMeans = pickle.load(fM)
        featureMax = pickle.load(fS)
    
    if make_new[0]: MakeTweetFile(make_new[1], FName)
    
    with open("MyReviews2/" + FName + " FULL", "rb") as f:
        Name, tso = cPickle.load(f)
        
    WordsInName = ("".join([i.lower() for i in Name if i in alphabet])).split(" ")    
    
    TweetSet = [i['text'] for i in tso]
    del tso
    gc.collect    
    
    #TweetSet = TweetSet[0:min(4000, len(TweetSet))]
    print len(TweetSet)
    Words = [''.join([j.lower() for j in i if j in alphabet]) for i in TweetSet]
    WordsR = [";".join([j for j in i.split(" ") if len(j)>2 and j[0] != "@" and j[0:4] != "http" and j not in WordsInName and j != ''.join(i for i in Name if i != ' ') and j!= 'movie']) for i in Words]
    TweetCounter = Counter(WordsR)
    if looks:
        keep_list = [i for i in range(len(WordsR)) if TweetCounter[WordsR[i]] == 1 and "looks" in WordsR[i] or "sounds"in WordsR[i] and not [j for j in avoid_words if j in Words[i]] and not [j for j in Words[i].split() if j[0:4] == "http"]]
    else:
        keep_list = [i for i in range(len(WordsR)) if TweetCounter[WordsR[i]] == 1 and "looks" not in WordsR[i] and "sounds" not in WordsR[i] and not [j for j in avoid_words if j in Words[i]] and not [j for j in Words[i].split() if j[0:4] == "http"]]
    
    WordsLessSpam = np.array(WordsR)[keep_list] 
    print_tweets_possible = np.array(TweetSet)[keep_list]

    M = [i.split(";") + [i.split(";")[j] + " " + i.split(";")[j+1] for j in range(len(i.split(";")) - 1)] for i in WordsLessSpam]
    res = sigmoid( np.dot(ProcessX(CreateX(M, FeatureWords), featureMeans, featureMax), ThetaRes))
    print M[0:10]
    print_tweets_pos = print_tweets_possible[res[:,0] > Threshold]
    print_tweets_neg = print_tweets_possible[res[:,0] < 1-Threshold]

    if looks:
        with open("MyReviews2/" + FName + " Lpos", "wb") as fp, open("MyReviews2/" + FName + " Lneg", "wb") as fn:
            pickle.dump(print_tweets_pos, fp)
            pickle.dump(print_tweets_neg, fn)
    else:
        with open("MyReviews2/" + FName + " NoLpos", "wb") as fp, open("MyReviews2/" + FName + " NoLneg", "wb") as fn:
            pickle.dump(print_tweets_pos, fp)
            pickle.dump(print_tweets_neg, fn)

    print [len(res[res>Threshold]), len(res[res<1-Threshold])]
    score = len(res[res>Threshold]) / (len(res[res<1-Threshold]) + len(res[res>Threshold]))
    print 'error: ', score*(1/np.sqrt(len(res[res>Threshold])) - 1/np.sqrt(len(res[res<1-Threshold]) + len(res[res>Threshold])))
    return score
    

    