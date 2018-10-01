
# Executive Summary
[Follow work here](https://github.com/ssanowitz24/Scotch-Reccomender)
The goal of this project was to build a reccomender system. The Scotch-whiskey dataset was obtained from kaggle.com. It originally consited of name of scotches, category of scotch, review points, price, currency and description. As given this dataset, with some cleaning, could be used for a reccomender system, but with some feature engineering and extraction, can be made more robust.

The first step is cleaning the data. Checking for duplicate values and null values, and dropping those rows from the dataframe. A more interesting problem was makling sure that there are no duplicate names in the name columns. A duplicate name will cause an error in the reccomender system. 

The next stpe was slicing the alchohol percentage from the name column. A function was written to extract the alchohol precentages from the name column and add them to a column called 'alc'. If the name did not contain an alchohol precentage the function were to add a Nan value then. The rows that contained Nan values were then dropped.

The next step was engineering some festures with the help of sci-kit-learn. First was taking the categories columns and transforming it using LabelEncoder which assigns a number for each category. The numerical categories were made to the column 'cat'. Next was making all the words in description into usuable numerical data. Utilizing CountVectorizer the words were made into word vectors. Stopword for english were enforced, so words like, and, or, and the, are ignored. Monograms(one word) and bi-grams(two words) are taken into account. These word vectors were made into an array and then made into dataframe. The words dataframe was than concatenated to the original dataframe. So more cleaning was then done to remove more null values.

Next was getting a dataframe of all numerical features. Before we did that the names columns was made into the index to be later used for the recommender system. Any column that did not contain numerical entries was excluded from the features in the dataframe. Now that the dataframe is all numerical a distance metric between all the vectors can be made. Utilzing cosine similarity distance metric we can see how similar the scotches are to each other. Cosine Similarity was employed on the whole dataframe and a new data frame was made with the index as the names columns and the coloumns as the name columns, making a cosine similarity matrix. Using the cosine simialrity matrix reccomendations can be made. It works on the pricipal of cosine. Similar vectors will have a small angle, closer to zero than dissimilar vectors which will have vectors with large angles between them, the largest being 90 degrees. The cosine of zero is one, so similar vectors will have a cosine value close to one. The cosine of 90 is zero so, dissimilar vectors will have a cosine value close to zero. We can see the top five cosine similarity values using the reccomender cell. 








    Ardmore Traditional
    Ardmore Traditional Cask 1998
    
    Similar Scotches
    Gordon & MacPhail (distilled at Glenlossie), 27 year old, 1978 vintage, cask #1815    0.997463
    Balvenie 1973 Vintage, 30 year old, Cask #9219                                        0.950143
    Adelphi (distilled at Glenrothes) 7 year old, 67.4%                                   0.887637
    Caol Ila Unpeated 12 year old Special Release 2011, 64%                               0.877481
    Wemyss Malts Fruit Bonbons (distilled at Glen Garioch) 1989, 66%                      0.873136
    Name: Ardmore Traditional Cask 1998, dtype: float64
    
    
    Ardmore Traditional Cask, 46%
    
    Similar Scotches
    Arran Single Island Malt, Non-chill-filtered, 46%                             0.993025
    Douglas Laing Old Particular (distilled at Blair Athol) 20 year old, 51.5%    0.992914
    Bruichladdich 3D, The Big Peat, 50%                                           0.992875
    Douglas Laing Provenance (distilled at Caol Ila) 6 year old, 46%              0.992809
    Lagavulin 1993 Islay Jazz Festival bottling (bottled 2011), 55.4%             0.992782
    Name: Ardmore Traditional Cask, 46%, dtype: float64
    
    

