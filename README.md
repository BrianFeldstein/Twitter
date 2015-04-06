# Twitter

This project involves sentiment analysis to construct movie reviews based on tweets from Twitter.  The code consists of two parts:  The "MovieReviews3.py" module is used to analyse a dataset of 50,000 movie reviews, and uses logistic regression to determine the positivity or negativity of a large (10,000+) set of words appearing in the reviews.

Search.py is used to collect tweets about specific movies.  These tweets are evaluted for sentiment using the logistic regression algorithm described above.  A review score is created based on the ratio to the number of very postive to very negative tweets.
