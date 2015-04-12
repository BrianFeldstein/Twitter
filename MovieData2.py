# -*- coding: utf-8 -*-
"""
Created on Fri Feb 13 09:36:29 2015

@author: Brian
"""

from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

#Name, MyRating, RTPub, RTCritic, MCPub, MCCritic
#At least 20 useful tweets required
#.975 Threshold:
MovieData = [
('Annie 0408', .76, .62, .28, .47, .33),#3876
('Big Hero 6 0407', .89, .92, .89, .80, .74),#3015
('Birdman 0407', .59, .80, .93, .79, .88),#884
('Chappie 0327', .76, .63, .31, .76, .41),#2408
('Cinderella 0327', .86, .85, .84, .75, .67),#18000
('Dear White People 0408', .77, .64, .92, .65, .79),#203
('Fifty Shades Of Grey 0327', .34, .45, .25, .36, .46),#3854
('Foxcatcher 0407', .65, .69, .88, .71, .81),#289
('Furious 7 0407', .91, .90, .83, .70, .67),#16600
('Focus 0327', .80, .58, .55, .61, .56),#5588
('Get Hard 0402', .67, .60, .29, .47, .34),#18000
('Home 0402', .83, .69, .47, .73, .55),#18000
('Horrible Bosses 2 0408', .77, .50, .35, .63, .40),#422
('Interstellar 0406', .82, .86, .72, .84, .74), #11095
('Into The Woods 0408', .51, .52, .71, .61, .69),#1199
('It Follows 0327', .54, .67, .95, .72, .83),#5936
('John Wick 0408', .79, .80, .84, .78, .67),#1514
('Kingsman The Secret Service 0327', .81, .87, .74, .77, .58),#821
('McFarland USA 0327', .98, .92, .79, .67, .60),#398
('Ouija 0408', .21, .27, .07, .52, .38),#506
('Run All Night 0327', .90, .64, .60, .76, .59),#1312
('Insurgent 0327', .74, .67, .31, .59, .42),#16600
('Song Of The Sea 0408', .96, .93, .98, .89, .85),#157
('St Vincent 0408', .95, .80, .77, .77, .64),#262
('The Cobbler 0327', .78, .40, .09, .49, .22),#295
('The Duff 0327', .75, .74, .67, .74, .56),#1720
('The Gunman 0327', .58, .37, .13, .43, .39),#4771
('The Imitation Game 0406', .97, .92, .89, .82, .73),#1044
('The Interview 0408', .65, .50, .53, .61, .52),#3346
('The Lazarus Effect 0327', .55, .27, .14, .42, .31),#312
('The Theory Of Everything 0407', .93, .84, .79, .75, .72),#795
('Unbroken 0406', .89, .71, .51, .65, .59),#1758
('What We Do In The Shadows 0327', .95, .88, .96, .84, .75),#202
('Woman In Gold 0407', .89, .90, .49, .77, .52)#1357 MCUser is only 6 ratings
]

"""with STD, .975 Threshold:
MovieData = [
('Annie 0408', .76, .62, .28, .47, .33),#3876
('Big Hero 6 0407', .89, .92, .89, .80, .74),#3015
('Birdman 0407', .59, .80, .93, .79, .88),#884
('Chappie 0327', .76, .63, .31, .76, .41),#2408
('Cinderella 0327', .86, .85, .84, .75, .67),#18000
('Dear White People 0408', .77, .64, .92, .65, .79),#203
('Fifty Shades Of Grey 0327', .34, .45, .25, .36, .46),#3854
('Foxcatcher 0407', .65, .69, .88, .71, .81),#289
('Furious 7 0407', .91, .90, .83, .70, .67),#16600
('Focus 0327', .80, .58, .55, .61, .56),#5588
('Get Hard 0402', .67, .60, .29, .47, .34),#18000
('Home 0402', .83, .69, .47, .73, .55),#18000
('Horrible Bosses 2 0408', .77, .50, .35, .63, .40),#422
('Interstellar 0406', .82, .86, .72, .84, .74), #11095
('Into The Woods 0408', .51, .52, .71, .61, .69),#1199
('It Follows 0327', .54, .67, .95, .72, .83),#5936
('John Wick 0408', .79, .80, .84, .78, .67),#1514
('Kingsman The Secret Service 0327', .81, .87, .74, .77, .58),#821
('McFarland USA 0327', .98, .92, .79, .67, .60),#398
('Ouija 0408', .21, .27, .07, .52, .38),#506
('Run All Night 0327', .90, .64, .60, .76, .59),#1312
('Insurgent 0327', .74, .67, .31, .59, .42),#16600
('Song Of The Sea 0408', .96, .93, .98, .89, .85),#157
('St Vincent 0408', .95, .80, .77, .77, .64),#262
('The Cobbler 0327', .78, .40, .09, .49, .22),#295
('The Duff 0327', .75, .74, .67, .74, .56),#1720
('The Gunman 0327', .58, .37, .13, .43, .39),#4771
('The Imitation Game 0406', .97, .92, .89, .82, .73),#1044
('The Interview 0408', .65, .50, .53, .61, .52),#3346
('The Lazarus Effect 0327', .55, .27, .14, .42, .31),#312
('The Theory Of Everything 0407', .93, .84, .79, .75, .72),#795
('Unbroken 0406', .89, .71, .51, .65, .59),#1758
('What We Do In The Shadows 0327', .95, .88, .96, .84, .75),#202
('Woman In Gold 0407', .89, .90, .49, .77, .52)#1357 MCUser is only 6 ratings
]
"""

"""
.9975 Threshold:
MovieData = [
('Chappie 0327', .81, .63, .31),#2408
('Cinderella 0327', .90, .85, .84),#18000
('Fifty Shades Of Grey 0327', .21, .45, .25),#3854
('Focus 0327', .78, .58, .55),#5588
('Get Hard 0402', .66, .60, .29),#18000
('Home 0402', .88, .69, .47),#18000
('Interstellar 0406', .81, .86, .72), #11095
('It Follows 0327', .42, .67, .95),#5936
('Kingsman The Secret Service 0327', .84, .87, .74),#821
('McFarland USA 0327', 1.0, .92, .79),#398
('Run All Night 0327', .90, .64, .60),#1312
('Insurgent 0327', .74, .67, .31),#16600
('The Duff 0327', .74, .74, .67),#1720
('The Gunman 0327', .50, .37, .13),#4771
('The Imitation Game 0406', 1.0, .92, .89),#1044
('The Lazarus Effect 0327', .44, .27, .14),#312
('Unbroken 0406', .91, .71, .51),#1758
('What We Do In The Shadows 0327', 1.0, .88, .96)#202
]
"""
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


MyRating = np.array([i[1] for i in MovieData])
PubRating = np.array([i[2] for i in MovieData])
CrRating = np.array([i[3] for i in MovieData])

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
    #plt.scatter(MyRatingP, PubRating, color = 'green') # http://i.imgur.com/LNTnT0A.png
    #plt.ylabel('Rotten Tomatoes Audience Rating')
    #plt.xlabel('My Twitter Rating (Rescaled)')
    

    plt.scatter((CrRating+MyRating)/2, PubRating, color = 'purple')  
    plt.ylabel('Rotten Tomatoes Audience Rating')
    plt.xlabel('Average of My Rating with Critic Rating') 


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
    PAvr = r((MyRating+CrRating)/2, PubRating)
    
    print Myr, MyPr, MyCrr, TFr
    
    plt.plot([0,1],[0,1])
    plt.show()
    plt.axis([0,1, 0, 1])
    
