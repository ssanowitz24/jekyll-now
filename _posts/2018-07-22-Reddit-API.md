
# Using Reddit's API for Predicting Comments
Reddit has become one of the centers of the internet.  Over 150 million people used reddit just in the last month.  This is a huge community of people looking to engage in whatever topics one can think of.   Having a trending or ‘Hot’ post on reddit can really up the views on a post.  Everyone wants to see what everyone else is seeing, no one wants to miss out on anything in this digital age.

[Follow work here](https://github.com/ssanowitz24/Reddit-API-Project)

# Executive Summary
Reddit has become one of the centers of the internet.  Over 150 million people used reddit just in the last month.  This is a huge community of people looking to engage in whatever topics one can think of.   Having a trending or ‘Hot’ post on reddit can really up the views on a post.  Everyone wants to see what everyone else is seeing, no one wants to miss out on anything in this digital age.
   
Knowing what features, whether they be numerical or categorical, that can lead to having a hot post on reddit can be a huge advantage for people and businesses looking to gain exposure.  Using the reddit api, nlp and classification models one can gain the edge they need for their post to be ‘Hot’.
   
First off, the numerical data.  The amount of time the post has been up on reddit and the number of cross-posts.  A ‘Hot’ post has an average time of being up on reddit for 9.96 hours.  If you are thinking about whether to cross-post or not, it is better to cross-post than to not cross-post. Using nlp techniques I look at words as data. When looking at the subreddits, the top 3 subreddits for a ‘Hot’ post are gaming, funny, and askreddit.  When looking at words in a post title that are in ‘Hot’ posts, the top three are, ‘need’, ‘nice’ and ‘true’. Overall the numerical data lead to better results than the nlp data.
   
These results only enhance what we can learn about what factors lead to a ‘Hot’ post on reddit.  Finding out what makes a post, ‘Hot’ can be vital information to anyone trying to make it in social media or a business trying to be the first to capitalize on a new trend. 



# Scraping Reddit
Utilizing the reddit api our goal is to predict wheather a comment is hot or not. A 'Hot' post is one that exceeds the median number of comments from the pulled reddit posts. If a post is below the median number of coments it is categorized as a 'Not'. The code below will exhibit how the posts were pulled from the reddit api and the different models utilzed to examine the posts. 



# Pulling posts
The code below was used to pull posts from the reddit api. Each time one pulls from the reddit api one gets 25 posts. This code allows someone to pull multiple times from the reddit api. If in this particular example I pulled 150 times giving me 3750 posts from reddit.





# List of dictionaries
This code makes a list of dictionaries of the chosen features from the reddit posts. Seen below are the title of the reddit post, the subbreddit the reddit post belonged to, the time the reddit post has been up on reddit, the number of comments the reddit post has, the number of crossposts the reddit post has and if the rediit post is a video or not. These features have been made into a list of dictionaries so that they can be easily made into a pandas dataframe and saved to as a csv file.







# The Binarizer
We want to make y, our target, the number of comments, into a binary variable, 'Hot or Not'. Recall that a 'Hot' post is one that's number of commets is above the median number of comments and a 'Not' post is below the median number of comments. To do this I used a sci-kit learn transformer called Binarizer to create a binary variable. Setting the Binarizer's threshold to the median number of commets transforms the number of comments feature, so that if the number of comments is above the median number of comments the Binarizer returns a 1 and if it is below returns a 0. I named this feature 'Hot or Not. 



# Fist Model - Subreddit 
This model uses the subreddit as the only feature to predict if a comment is 'Hot' or 'Not'. To make sure the model tests well on unseen data a train test split was utitlized. Using Contvectorizer as the transformer, which creates  feature columns for each subreddit and Random Forest, which uses multiple decisions trees with different features in each tree, is the model that will be employed to predict 'Hot or Not'. Without any optimzation this model returns an accuracy of 63.25%, which is higher than the baseline. Using GridSearchCV to optimize parameters returns an accuracy of 60.10% which is worse, but still higher than the baseline. The other metric I investigated was the confusion matrix to see the distribution of true negatives, false positives, true positives and false positives. Than using feature importance I look at the top 10 subreddits that contribute to a 'Hot' post.


# Second Model - Fun Features
This model uses a number of features that were engineered. The is_video feature, tells if the posts are a video or not. Utitlizing the '.astype(int) makes the trues and falses of is_video into a dummy column giving 1 for true and 0 for false. Investigating punctuation I looked into if posts' titles were questions, exclamations or if people appreciated good grammer and if the post ended with a period. The last two features are if cats were in the title and if the word when is in the title. Since all the variables have been dummied using '.astype(int)' this model just used randomforest. Without optimization the model returned an accuracy score of 55.99%, better than baseline, With GridsearchCV optimization the model returned an accuracy of 53.05%, a little worse but better than the baseline. A confusion matrix was used as a metric for this model as well as feature importance to see which of these fun features contributed to predicted 'Hot or Not'. 

# Model 3 - Many features RandomForest
This model utilzes the fun features form model 2, the subbreddits from model 1, time_elasped, number of crossposts, and the words in the title of the reddit posts. To use the subreddit and the titles I needed to countvectorize each feature. Since titles contain multiple words I used a stop_word='eglish in the countvectorizer. This would filter out words that do not transmit much information in the english language, like 'the', 'a' , and 'an' for example. Countvectorizer returns matrices for your titles and subreddits so I unpacked them into arrays and made them into pandas dataframes. Once they are dataframes I concatenated them to the orginal features I wanted and this dataframe became my features for this model. Having numerical data like number of crossposts and time elasped I used the transformer standarscaler to scale all the features as a precaution. I ran a randomforest model which when not optimized produced an accuracy of 67.03% and when optimized using GridsearchCV produced an accuracy of 68.82%. A confusion matrix was used as a metric to test the model's performance and feature importance was used to see what features contributed to predicted a 'Hot' post.

# Model 4 - Many features Logistic Regression
To examine another model besides randomforest I choose logistic regrssion for this classification problem. Utiling the same features as in model 3 I transformed the features using a standardscalar.Running the the model without optimization the accuracy score was 62.15%. Optimizing with GridsearchCV I got an accuracy score of 69.24%. GridsearchCV optimized the logistic regression to have a penalty of l1 the LASSO penalty and a low C of 0.1. I used a confusion matrix to further investigate the modles performance. I used a logistic regression coefs_ to see which features best predicted a 'Hot' post.

# Model 5 Title as only Feature
This model is alot like the first model with subreddit being the only feature, but now the words in titles are the only features. One big difference is the stop_words ='english in the CountVectorizer. This will filter out words that do not transmit vital information, such as words like,'for', 'or' and 'they' as examples. When running an model without optmization, the model returns an accuracy score of 49.84% the worst scoring model and the only model to score worse than the baseline. When running gridsearch to optimize the model the model returns a score of 55.84%. The Gridsearch CV optimized the title feature utilizing the countvectorizer's binary set to equal False, which means each word gets a count everytime that word shows up in a post's title. 


# Conclusion, Next Steps
In a society of instantaneous information everday brings a new news cycle. A more extensive analysis must be done. One where posts are scraped over a much grander time scale to account for an ever changing information scape. Numerical features were of a much greater signifigance to an 'Hot' or 'Not' post than the NLP features, so gathering more numerical features would acheive a better model. Reddit posts are ever changing and predicting comments on them are just as difficult as predicting current events. 

