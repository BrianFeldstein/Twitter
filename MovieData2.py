# -*- coding: utf-8 -*-
"""
Created on Fri Feb 13 09:36:29 2015

@author: Brian
"""

from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

#Name, MyRating, RTPub, RTCritic
#At least 20 useful tweets required

MovieData = [
('Chappie 0327', .74, .64, .30),#2408
('Cinderella 0327', .89, .87, .84),#18000
('Fifty Shades Of Grey 0327', .29, .45, .25),#3854
('Focus 0327', .73, .59, .55),#5588
('Get Hard 0402', .73, .61, .29),#18000
('Home 0402', .84, .69, .47),#18000
('It Follows 0327', .53, .72, .94),#5936
('Kingsman The Secret Service 0327', .76, .87, .74),#821
('McFarland USA 0327', 1.0, .92, .79),#398
('Run All Night 0327', .92, .65, .60),#1312
('Insurgent 0327', .73, .69, .31),#16600
('The Duff 0327', .71, .74, .64),#1720
('The Gunman 0327', .53, .36, .12),#4771
('The Lazarus Effect 0327', .52, .28, .14),#312
('What We Do In The Shadows 0327', .95, .88, .96),#202
]

""" before looks:
MovieData = [
('Chappie 0326', .77, .64, .30),
('Cinderella 0326', .88, .87, .84),
('Fifty Shades Of Grey 0326', .39, .45, .25),
('Focus 0326', .78, .59, .55),
('It Follows 0326', .52, .72, .94),
('Jupiter Ascending 0326', .56, .46, .25),
('Kingsman The Secret Service 0326', .90, .87, .74),
('McFarland USA 0326', .93, .92, .79),
('Run All Night 0326', .87, .65, .60),
('The Lazarus Effect 0326', .41, .28, .14),
('The Second Best Exotic Marigold Hotel 0326', .84, .67, .63),
('What We Do In The Shadows 0326', .98, .88, .96)
]
"""

twitflicksData = [
('Adore', .75, .42, .31),
('Amour', .875, .82, .93),
('Augustine', .625 , .51, .73),
('Backwards', .75, .54, .28),
('Belle', .875, .84, .83),
('Boyhood', .875, .84, .98),
('Brave', .875, .76, .78),
('Catfish', .625, .70, .81),
('Chef', 1.0, .85, .86),
('Drive', .75, .78, .93),
('Flight', .75, .74, .78),
('Ida', 1.0, .80, .96),
('Jobs', .625, .41, .27),
('Lincoln', .625, .80, .90),
('Lucy', .625, .47, .66),
('Luv', .875, .61, .36),
('Maleficent', 1.0, .71, .49),
('OKA!', .75, .40, .71),
('Pump', .625, .87, .75),
('Rush', .875, .89, .89),
('Safe', .875, .59, .57),
('Shame', .625, .75, .79), 
('Teenage', .625, .52, .75),
('Test', .25, .61, .89),
('Titanic', .75, .69, .88)
]


MyRating = [i[1] for i in MovieData]
PubRating = [i[2] for i in MovieData]
CrRating = [i[3] for i in MovieData]

TFRating = [i[1] for i in twitflicksData]
TFPubRating  = [i[2] for i in twitflicksData]
TFCrRating  = [i[3] for i in twitflicksData]

def Predictions(Th, Ratings):
    return [i * Th[1] +Th[0] for i in Ratings]
    
def Cost(Th, Ratings, RatingsToPred):
    return (1/(2 * len(Ratings)))*np.sum((np.array(Predictions(Th, Ratings)) - np.array(RatingsToPred))**2)
    
MyThBest = minimize(fun = lambda th: Cost(th, MyRating, PubRating), x0 = np.array([0,1]), method = 'TNC').x
CrThBest = minimize(fun = lambda th: Cost(th, CrRating, PubRating), x0 = np.array([0,1]), method = 'TNC').x
MyThBestForCr = minimize(fun = lambda th: Cost(th, MyRating, CrRating), x0 = np.array([0,1]), method = 'TNC').x
CrThBestForMy = minimize(fun = lambda th: Cost(th, CrRating, MyRating), x0 = np.array([0,1]), method = 'TNC').x
TFThetaBest = minimize(fun = lambda th: Cost(th, TFRating, TFPubRating), x0 = np.array([0,1]), method = 'TNC').x



MyRatingP = Predictions(MyThBest, MyRating)
CrRatingP = Predictions(CrThBest, CrRating)
TFRatingP = Predictions(TFThetaBest, TFRating)

def r(L1, L2):
    return np.dot(np.transpose( np.array(L1) - np.mean(L1)), np.array(L2) - np.mean(L2) )/(np.sqrt(len(L1)*len(L2))*np.std(L1)*np.std(L2))    
        


if __name__ == "__main__":
    #plt.scatter(MyRating, PubRating, color = 'blue') # http://i.imgur.com/HrMtxdb.png
    plt.scatter(MyRatingP, PubRating, color = 'green') # http://i.imgur.com/LNTnT0A.png
    plt.ylabel('Rotten Tomatoes Audience Rating')
    plt.xlabel('My Twitter Rating (Rescaled)')
    
    #plt.scatter(CrRating, PubRating, color = 'red') # http://i.imgur.com/DFQZbCo.png
    #plt.scatter(CrRatingP, PubRating, color = 'turquoise') # http://i.imgur.com/JUjSqQF.png
    
    #plt.scatter(Predictions(MyThBestForCr, MyRating), CrRating, color = 'purple')
    #plt.scatter(Predictions(CrThBestForMy, CrRating), MyRating, color = 'yellow')
    
    
    #plt.scatter(TFRating, TFPubRating, color = 'brown') # http://i.imgur.com/9glvn9k.png
    #plt.scatter(TFRatingP, TFPubRating, color = 'magenta') # http://i.imgur.com/4jygktj.png
    
        
    Myr = r(MyRatingP, PubRating)
    MyPr = r(CrRatingP, PubRating)
    MyCrr = r(CrRating, MyRating)
    TFr = r(TFRatingP, TFPubRating)
    
    print Myr, MyPr, MyCrr, TFr
    
    plt.plot([0,1],[0,1])
    plt.show()
    plt.axis([0,1, 0, 1])
    
