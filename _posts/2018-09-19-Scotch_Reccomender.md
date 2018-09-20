
# Executive Summary
### Run last two cells to see reccomender in action
The goal of this project was to build a reccomender system. The Scotch-whiskey dataset was obtained from kaggle.com. It originally consited of name of scotches, category of scotch, review points, price, currency and description. As given this dataset, with some cleaning, could be used for a reccomender system, but with some feature engineering and extraction, can be made more robust.

The first step is cleaning the data. Checking for duplicate values and null values, and dropping those rows from the dataframe. A more interesting problem was makling sure that there are no duplicate names in the name columns. A duplicate name will cause an error in the reccomender system. 

The next stpe was slicing the alchohol percentage from the name column. A function was written to extract the alchohol precentages from the name column and add them to a column called 'alc'. If the name did not contain an alchohol precentage the function were to add a Nan value then. The rows that contained Nan values were then dropped.

The next step was engineering some festures with the help of sci-kit-learn. First was taking the categories columns and transforming it using LabelEncoder which assigns a number for each category. The numerical categories were made to the column 'cat'. Next was making all the words in description into usuable numerical data. Utilizing CountVectorizer the words were made into word vectors. Stopword for english were enforced, so words like, and, or, and the, are ignored. Monograms(one word) and bi-grams(two words) are taken into account. These word vectors were made into an array and then made into dataframe. The words dataframe was than concatenated to the original dataframe. So more cleaning was then done to remove more null values.

Next was getting a dataframe of all numerical features. Before we did that the names columns was made into the index to be later used for the recommender system. Any column that did not contain numerical entries was excluded from the features in the dataframe. Now that the dataframe is all numerical a distance metric between all the vectors can be made. Utilzing cosine similarity distance metric we can see how similar the scotches are to each other. Cosine Similarity was employed on the whole dataframe and a new data frame was made with the index as the names columns and the coloumns as the name columns, making a cosine similarity matrix. Using the cosine simialrity matrix reccomendations can be made. It works on the pricipal of cosine. Similar vectors will have a small angle, closer to zero than dissimilar vectors which will have vectors with large angles between them, the largest being 90 degrees. The cosine of zero is one, so similar vectors will have a cosine value close to one. The cosine of 90 is zero so, dissimilar vectors will have a cosine value close to zero. We can see the top five cosine similarity values using the reccomender cell. 

[Follow work here](https://github.com/ssanowitz24/Scotch-Reccomender)








```python
#import libraries
import pandas as pd
import numpy as np

```

# Initial Cleaning


```python
# read in data
df = pd.read_csv('./scotch_review.csv')
```


```python
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Unnamed: 0</th>
      <th>name</th>
      <th>category</th>
      <th>review.point</th>
      <th>price</th>
      <th>currency</th>
      <th>description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>Johnnie Walker Blue Label, 40%</td>
      <td>Blended Scotch Whisky</td>
      <td>97</td>
      <td>225</td>
      <td>$</td>
      <td>Magnificently powerful and intense. Caramels, ...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>Black Bowmore, 1964 vintage, 42 year old, 40.5%</td>
      <td>Single Malt Scotch</td>
      <td>97</td>
      <td>4500.00</td>
      <td>$</td>
      <td>What impresses me most is how this whisky evol...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>Bowmore 46 year old (distilled 1964), 42.9%</td>
      <td>Single Malt Scotch</td>
      <td>97</td>
      <td>13500.00</td>
      <td>$</td>
      <td>There have been some legendary Bowmores from t...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>Compass Box The General, 53.4%</td>
      <td>Blended Malt Scotch Whisky</td>
      <td>96</td>
      <td>325</td>
      <td>$</td>
      <td>With a name inspired by a 1926 Buster Keaton m...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>Chivas Regal Ultis, 40%</td>
      <td>Blended Malt Scotch Whisky</td>
      <td>96</td>
      <td>160</td>
      <td>$</td>
      <td>Captivating, enticing, and wonderfully charmin...</td>
    </tr>
  </tbody>
</table>
</div>




```python
#drop column Unnamed: 0
df.drop('Unnamed: 0', axis = 1, inplace =True)
```


```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 2247 entries, 0 to 2246
    Data columns (total 6 columns):
    name            2247 non-null object
    category        2247 non-null object
    review.point    2247 non-null int64
    price           2247 non-null object
    currency        2247 non-null object
    description     2247 non-null object
    dtypes: int64(1), object(5)
    memory usage: 105.4+ KB



```python
# check for duplicates
df.duplicated().sum()
```




    2




```python
#drop duplicates
df.drop_duplicates(inplace=True)
```


```python
df.isnull().sum()
```




    name            0
    category        0
    review.point    0
    price           0
    currency        0
    description     0
    dtype: int64




```python
#drop duplicates in name, is important for reccomender to work properly
df.name = df.name.drop_duplicates()
```


```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 2245 entries, 0 to 2246
    Data columns (total 6 columns):
    name            2223 non-null object
    category        2245 non-null object
    review.point    2245 non-null int64
    price           2245 non-null object
    currency        2245 non-null object
    description     2245 non-null object
    dtypes: int64(1), object(5)
    memory usage: 122.8+ KB



```python
#reset index
df.reset_index(inplace = True)
```


```python
#check for nulls
df.isnull().sum()
```




    index            0
    name            22
    category         0
    review.point     0
    price            0
    currency         0
    description      0
    dtype: int64




```python
#drop nulls
df.dropna(inplace=True)
```


```python
#reset index
df.reset_index(inplace=True)
```

# Feature Extraction


```python
#test code for integers
df.name[0][-4:-1].strip(' ').strip('%')
```




    '40'




```python
#test code for floats
df.name[1][-6:-1].strip(' ').strip('%')
```




    '40.5'




```python
# append the alchohol content from the end of the name to its own column, 
#if no alchohol content apparent append a nan value
alc = []
def extract(my_df):
    for i in range(len(my_dfdf)):
        if my_df.name[i][-4:-1].strip(' ').strip('%').isdigit():
            alc.append(float(my_df.name[i][-4:-1].strip(' ').strip('%')))
        else:
            try:
                alc.append(float(my_df.name[i][-6:-1].strip(' ').strip('%')))
            except ValueError:
                alc.append(np.nan)

        
            

    
    

     


extract(df)    
```


```python
# make alc the alchohol content column
df['alc'] = alc
```


```python
# check for nans
df.isnull().sum()
```




    level_0          0
    index            0
    name             0
    category         0
    review.point     0
    price            0
    currency         0
    description      0
    alc             44
    dtype: int64




```python
#drop nans
df.dropna(inplace=True)
```


```python
df.isnull().sum()
```


```python
# drop two columns level_0 and index
df.drop(['level_0','index'],axis=1, inplace=True)
```


```python
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>name</th>
      <th>category</th>
      <th>review.point</th>
      <th>price</th>
      <th>currency</th>
      <th>description</th>
      <th>alc</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Johnnie Walker Blue Label, 40%</td>
      <td>Blended Scotch Whisky</td>
      <td>97</td>
      <td>225</td>
      <td>$</td>
      <td>Magnificently powerful and intense. Caramels, ...</td>
      <td>40.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Black Bowmore, 1964 vintage, 42 year old, 40.5%</td>
      <td>Single Malt Scotch</td>
      <td>97</td>
      <td>4500.00</td>
      <td>$</td>
      <td>What impresses me most is how this whisky evol...</td>
      <td>40.5</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Bowmore 46 year old (distilled 1964), 42.9%</td>
      <td>Single Malt Scotch</td>
      <td>97</td>
      <td>13500.00</td>
      <td>$</td>
      <td>There have been some legendary Bowmores from t...</td>
      <td>42.9</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Compass Box The General, 53.4%</td>
      <td>Blended Malt Scotch Whisky</td>
      <td>96</td>
      <td>325</td>
      <td>$</td>
      <td>With a name inspired by a 1926 Buster Keaton m...</td>
      <td>53.4</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Chivas Regal Ultis, 40%</td>
      <td>Blended Malt Scotch Whisky</td>
      <td>96</td>
      <td>160</td>
      <td>$</td>
      <td>Captivating, enticing, and wonderfully charmin...</td>
      <td>40.0</td>
    </tr>
  </tbody>
</table>
</div>



# Feature Engineering


```python
# import libraries for feature building
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import LabelEncoder
```


```python
#call Label Encoer to be used on the categopry column to make it into numerical data
le = LabelEncoder()
```


```python
#fit transform label encoder
cat = le.fit_transform(df.category)
```


```python
# make the label encoded category the column cat in the dataframe
df['cat'] = cat
```


```python
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>name</th>
      <th>category</th>
      <th>review.point</th>
      <th>price</th>
      <th>currency</th>
      <th>description</th>
      <th>alc</th>
      <th>cat</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Johnnie Walker Blue Label, 40%</td>
      <td>Blended Scotch Whisky</td>
      <td>97</td>
      <td>225</td>
      <td>$</td>
      <td>Magnificently powerful and intense. Caramels, ...</td>
      <td>40.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Black Bowmore, 1964 vintage, 42 year old, 40.5%</td>
      <td>Single Malt Scotch</td>
      <td>97</td>
      <td>4500.00</td>
      <td>$</td>
      <td>What impresses me most is how this whisky evol...</td>
      <td>40.5</td>
      <td>4</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Bowmore 46 year old (distilled 1964), 42.9%</td>
      <td>Single Malt Scotch</td>
      <td>97</td>
      <td>13500.00</td>
      <td>$</td>
      <td>There have been some legendary Bowmores from t...</td>
      <td>42.9</td>
      <td>4</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Compass Box The General, 53.4%</td>
      <td>Blended Malt Scotch Whisky</td>
      <td>96</td>
      <td>325</td>
      <td>$</td>
      <td>With a name inspired by a 1926 Buster Keaton m...</td>
      <td>53.4</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Chivas Regal Ultis, 40%</td>
      <td>Blended Malt Scotch Whisky</td>
      <td>96</td>
      <td>160</td>
      <td>$</td>
      <td>Captivating, enticing, and wonderfully charmin...</td>
      <td>40.0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Need word vectors from the description column, usung stop words in english and 1-gram and 2-grams
cv = CountVectorizer(stop_words='english', ngram_range=(1,2))
```


```python
#fit trans form CountVecotrizer on description column into an array
words = cv.fit_transform(df.description).toarray()
```


```python
# make word vector array into a dataframe, words
words = pd.DataFrame(words, columns=cv.get_feature_names())
```


```python
# concatenate dataframe with words dataframe
df = pd.concat([df, words], axis=1)
```


```python
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>name</th>
      <th>category</th>
      <th>review.point</th>
      <th>price</th>
      <th>currency</th>
      <th>description</th>
      <th>alc</th>
      <th>cat</th>
      <th>00</th>
      <th>000</th>
      <th>...</th>
      <th>zingy yes</th>
      <th>zippy</th>
      <th>zippy acidity</th>
      <th>zippy clean</th>
      <th>zone</th>
      <th>zone moderate</th>
      <th>ìle</th>
      <th>ìle 2016</th>
      <th>ìle limited</th>
      <th>ìle release</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Johnnie Walker Blue Label, 40%</td>
      <td>Blended Scotch Whisky</td>
      <td>97.0</td>
      <td>225</td>
      <td>$</td>
      <td>Magnificently powerful and intense. Caramels, ...</td>
      <td>40.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Black Bowmore, 1964 vintage, 42 year old, 40.5%</td>
      <td>Single Malt Scotch</td>
      <td>97.0</td>
      <td>4500.00</td>
      <td>$</td>
      <td>What impresses me most is how this whisky evol...</td>
      <td>40.5</td>
      <td>4.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Bowmore 46 year old (distilled 1964), 42.9%</td>
      <td>Single Malt Scotch</td>
      <td>97.0</td>
      <td>13500.00</td>
      <td>$</td>
      <td>There have been some legendary Bowmores from t...</td>
      <td>42.9</td>
      <td>4.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Compass Box The General, 53.4%</td>
      <td>Blended Malt Scotch Whisky</td>
      <td>96.0</td>
      <td>325</td>
      <td>$</td>
      <td>With a name inspired by a 1926 Buster Keaton m...</td>
      <td>53.4</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Chivas Regal Ultis, 40%</td>
      <td>Blended Malt Scotch Whisky</td>
      <td>96.0</td>
      <td>160</td>
      <td>$</td>
      <td>Captivating, enticing, and wonderfully charmin...</td>
      <td>40.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 69419 columns</p>
</div>




```python
# make the the name column the index of the dataframe
df.index = df.name
```


```python
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>name</th>
      <th>category</th>
      <th>review.point</th>
      <th>price</th>
      <th>currency</th>
      <th>description</th>
      <th>alc</th>
      <th>cat</th>
      <th>00</th>
      <th>000</th>
      <th>...</th>
      <th>zingy yes</th>
      <th>zippy</th>
      <th>zippy acidity</th>
      <th>zippy clean</th>
      <th>zone</th>
      <th>zone moderate</th>
      <th>ìle</th>
      <th>ìle 2016</th>
      <th>ìle limited</th>
      <th>ìle release</th>
    </tr>
    <tr>
      <th>name</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Johnnie Walker Blue Label, 40%</th>
      <td>Johnnie Walker Blue Label, 40%</td>
      <td>Blended Scotch Whisky</td>
      <td>97.0</td>
      <td>225</td>
      <td>$</td>
      <td>Magnificently powerful and intense. Caramels, ...</td>
      <td>40.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>Black Bowmore, 1964 vintage, 42 year old, 40.5%</th>
      <td>Black Bowmore, 1964 vintage, 42 year old, 40.5%</td>
      <td>Single Malt Scotch</td>
      <td>97.0</td>
      <td>4500.00</td>
      <td>$</td>
      <td>What impresses me most is how this whisky evol...</td>
      <td>40.5</td>
      <td>4.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>Bowmore 46 year old (distilled 1964), 42.9%</th>
      <td>Bowmore 46 year old (distilled 1964), 42.9%</td>
      <td>Single Malt Scotch</td>
      <td>97.0</td>
      <td>13500.00</td>
      <td>$</td>
      <td>There have been some legendary Bowmores from t...</td>
      <td>42.9</td>
      <td>4.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>Compass Box The General, 53.4%</th>
      <td>Compass Box The General, 53.4%</td>
      <td>Blended Malt Scotch Whisky</td>
      <td>96.0</td>
      <td>325</td>
      <td>$</td>
      <td>With a name inspired by a 1926 Buster Keaton m...</td>
      <td>53.4</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>Chivas Regal Ultis, 40%</th>
      <td>Chivas Regal Ultis, 40%</td>
      <td>Blended Malt Scotch Whisky</td>
      <td>96.0</td>
      <td>160</td>
      <td>$</td>
      <td>Captivating, enticing, and wonderfully charmin...</td>
      <td>40.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 69419 columns</p>
</div>




```python
# get columns with numerical data only
features = [col for col in df.columns if col not in ['name','category','currency', 'description']]
```


```python
# mnake data frame of only numerical features
df = df[features]
```


```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Index: 2223 entries, Johnnie Walker Blue Label, 40% to Distillery Select 'Inchmoan' (distilled at Loch Lomond), Cask #151, 13 year old, 1992 vintage, 45%
    Columns: 69415 entries, review.point to ìle release
    dtypes: float64(69413), object(2)
    memory usage: 1.1+ GB



```python
#get rid of nay non-numerical features
df = df.select_dtypes(exclude=object)
```


```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Index: 2223 entries, Johnnie Walker Blue Label, 40% to Distillery Select 'Inchmoan' (distilled at Loch Lomond), Cask #151, 13 year old, 1992 vintage, 45%
    Columns: 69413 entries, review.point to ìle release
    dtypes: float64(69413)
    memory usage: 1.1+ GB



```python
#check for nulls
df.isnull().sum()
```




    review.point           44
    price                  44
    alc                    44
    cat                    44
    00                     44
    000                    44
    000 bottle             44
    000 bottled            44
    000 bottles            44
    000 cases              44
    000 extravagantly      44
    000 individually       44
    000 liters             44
    000 special            44
    000 strong             44
    000 think              44
    000 whisky             44
    002                    44
    002 lemonade           44
    011                    44
    011 500                44
    060                    44
    060 bottles            44
    076                    44
    076 bottles            44
    08                     44
    08 masterclass         44
    080                    44
    080 bottles            44
    090                    44
                           ..
    zing ginger            44
    zing maturity          44
    zing notes             44
    zing palate            44
    zing showing           44
    zing truly             44
    zing whisky            44
    zinginess              44
    zinginess continues    44
    zinging                44
    zinging finish         44
    zings                  44
    zings adding           44
    zingy                  44
    zingy citrus           44
    zingy ginger           44
    zingy intensity        44
    zingy notes            44
    zingy orange           44
    zingy sherbet          44
    zingy yes              44
    zippy                  44
    zippy acidity          44
    zippy clean            44
    zone                   44
    zone moderate          44
    ìle                    44
    ìle 2016               44
    ìle limited            44
    ìle release            44
    Length: 69413, dtype: int64




```python
#drop null values
df.dropna(inplace=True)
```


```python
df.isnull().sum()
```




    review.point           0
    price                  0
    alc                    0
    cat                    0
    00                     0
    000                    0
    000 bottle             0
    000 bottled            0
    000 bottles            0
    000 cases              0
    000 extravagantly      0
    000 individually       0
    000 liters             0
    000 special            0
    000 strong             0
    000 think              0
    000 whisky             0
    002                    0
    002 lemonade           0
    011                    0
    011 500                0
    060                    0
    060 bottles            0
    076                    0
    076 bottles            0
    08                     0
    08 masterclass         0
    080                    0
    080 bottles            0
    090                    0
                          ..
    zing ginger            0
    zing maturity          0
    zing notes             0
    zing palate            0
    zing showing           0
    zing truly             0
    zing whisky            0
    zinginess              0
    zinginess continues    0
    zinging                0
    zinging finish         0
    zings                  0
    zings adding           0
    zingy                  0
    zingy citrus           0
    zingy ginger           0
    zingy intensity        0
    zingy notes            0
    zingy orange           0
    zingy sherbet          0
    zingy yes              0
    zippy                  0
    zippy acidity          0
    zippy clean            0
    zone                   0
    zone moderate          0
    ìle                    0
    ìle 2016               0
    ìle limited            0
    ìle release            0
    Length: 69413, dtype: int64




```python
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>review.point</th>
      <th>price</th>
      <th>alc</th>
      <th>cat</th>
      <th>00</th>
      <th>000</th>
      <th>000 bottle</th>
      <th>000 bottled</th>
      <th>000 bottles</th>
      <th>000 cases</th>
      <th>...</th>
      <th>zingy yes</th>
      <th>zippy</th>
      <th>zippy acidity</th>
      <th>zippy clean</th>
      <th>zone</th>
      <th>zone moderate</th>
      <th>ìle</th>
      <th>ìle 2016</th>
      <th>ìle limited</th>
      <th>ìle release</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Johnnie Walker Blue Label, 40%</th>
      <td>97.0</td>
      <td>0.0</td>
      <td>40.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>Black Bowmore, 1964 vintage, 42 year old, 40.5%</th>
      <td>97.0</td>
      <td>0.0</td>
      <td>40.5</td>
      <td>4.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>Bowmore 46 year old (distilled 1964), 42.9%</th>
      <td>97.0</td>
      <td>0.0</td>
      <td>42.9</td>
      <td>4.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>Compass Box The General, 53.4%</th>
      <td>96.0</td>
      <td>0.0</td>
      <td>53.4</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>Chivas Regal Ultis, 40%</th>
      <td>96.0</td>
      <td>0.0</td>
      <td>40.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 69413 columns</p>
</div>




```python
# save this dataframe as .csv file
df.to_csv('./df.csv', index_label=False)
```


```python
# read in .csv as datframe
df = pd.read_csv('./df.csv')
```


```python
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>review.point</th>
      <th>price</th>
      <th>alc</th>
      <th>cat</th>
      <th>00</th>
      <th>000</th>
      <th>000 bottle</th>
      <th>000 bottled</th>
      <th>000 bottles</th>
      <th>000 cases</th>
      <th>...</th>
      <th>zingy yes</th>
      <th>zippy</th>
      <th>zippy acidity</th>
      <th>zippy clean</th>
      <th>zone</th>
      <th>zone moderate</th>
      <th>ìle</th>
      <th>ìle 2016</th>
      <th>ìle limited</th>
      <th>ìle release</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Johnnie Walker Blue Label, 40%</th>
      <td>97.0</td>
      <td>0.0</td>
      <td>40.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>Black Bowmore, 1964 vintage, 42 year old, 40.5%</th>
      <td>97.0</td>
      <td>0.0</td>
      <td>40.5</td>
      <td>4.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>Bowmore 46 year old (distilled 1964), 42.9%</th>
      <td>97.0</td>
      <td>0.0</td>
      <td>42.9</td>
      <td>4.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>Compass Box The General, 53.4%</th>
      <td>96.0</td>
      <td>0.0</td>
      <td>53.4</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>Chivas Regal Ultis, 40%</th>
      <td>96.0</td>
      <td>0.0</td>
      <td>40.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 69413 columns</p>
</div>



# Distance Metric - Cosine


```python
#run cosine_similarity on dataframe
cos = cosine_similarity(df)
```


```python
# create dataframe of cosine similarity values where the index is names and the columns is names
recs = pd.DataFrame(cos, index = df.index, columns = df.index)
```


```python
recs.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Johnnie Walker Blue Label, 40%</th>
      <th>Black Bowmore, 1964 vintage, 42 year old, 40.5%</th>
      <th>Bowmore 46 year old (distilled 1964), 42.9%</th>
      <th>Compass Box The General, 53.4%</th>
      <th>Chivas Regal Ultis, 40%</th>
      <th>Ardbeg Corryvreckan, 57.1%</th>
      <th>Gold Bowmore, 1964 vintage, 42.4%</th>
      <th>Bowmore, 40 year old, 44.8%</th>
      <th>The Dalmore, 50 year old, 52.8%</th>
      <th>Glenfarclas Family Casks 1954 Cask #1260, 47.2%</th>
      <th>...</th>
      <th>Islay Mist, 8 year old, 40%</th>
      <th>The Singleton of Dufftown 28 year old, 52.3%</th>
      <th>Clan Denny (distilled at Girvan) 1992 21 year old HH9451, 59.6%</th>
      <th>Douglas Laing Single Minded (distilled at Jura) 8 year old, 41.5%</th>
      <th>Single Malts of Scotland (distilled at Craigellachie) 1996, 52.7%</th>
      <th>High Commissioner, 40%</th>
      <th>The Arran Malt, 43%</th>
      <th>Bowmore, 16 year old, 1990 vintage, 53.8%</th>
      <th>Bruichladdich 'Waves', 46%</th>
      <th>Inchmurrin 15 year old, 46%</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Johnnie Walker Blue Label, 40%</th>
      <td>1.000000</td>
      <td>0.990170</td>
      <td>0.991265</td>
      <td>0.984664</td>
      <td>0.990698</td>
      <td>0.981290</td>
      <td>0.985275</td>
      <td>0.989971</td>
      <td>0.982504</td>
      <td>0.988512</td>
      <td>...</td>
      <td>0.988273</td>
      <td>0.970474</td>
      <td>0.954490</td>
      <td>0.985258</td>
      <td>0.972248</td>
      <td>0.987454</td>
      <td>0.984687</td>
      <td>0.968607</td>
      <td>0.980744</td>
      <td>0.980909</td>
    </tr>
    <tr>
      <th>Black Bowmore, 1964 vintage, 42 year old, 40.5%</th>
      <td>0.990170</td>
      <td>1.000000</td>
      <td>0.991693</td>
      <td>0.984868</td>
      <td>0.990051</td>
      <td>0.983391</td>
      <td>0.987846</td>
      <td>0.991289</td>
      <td>0.985258</td>
      <td>0.989199</td>
      <td>...</td>
      <td>0.989604</td>
      <td>0.972514</td>
      <td>0.956911</td>
      <td>0.987694</td>
      <td>0.974304</td>
      <td>0.988040</td>
      <td>0.986691</td>
      <td>0.971231</td>
      <td>0.982220</td>
      <td>0.983631</td>
    </tr>
    <tr>
      <th>Bowmore 46 year old (distilled 1964), 42.9%</th>
      <td>0.991265</td>
      <td>0.991693</td>
      <td>1.000000</td>
      <td>0.988023</td>
      <td>0.991137</td>
      <td>0.986286</td>
      <td>0.987828</td>
      <td>0.992785</td>
      <td>0.986493</td>
      <td>0.991293</td>
      <td>...</td>
      <td>0.990743</td>
      <td>0.977084</td>
      <td>0.962042</td>
      <td>0.989244</td>
      <td>0.978558</td>
      <td>0.990453</td>
      <td>0.989288</td>
      <td>0.975499</td>
      <td>0.986046</td>
      <td>0.986205</td>
    </tr>
    <tr>
      <th>Compass Box The General, 53.4%</th>
      <td>0.984664</td>
      <td>0.984868</td>
      <td>0.988023</td>
      <td>1.000000</td>
      <td>0.985396</td>
      <td>0.992465</td>
      <td>0.982754</td>
      <td>0.989678</td>
      <td>0.990245</td>
      <td>0.990552</td>
      <td>...</td>
      <td>0.991720</td>
      <td>0.987148</td>
      <td>0.977797</td>
      <td>0.990216</td>
      <td>0.988772</td>
      <td>0.991015</td>
      <td>0.991874</td>
      <td>0.987393</td>
      <td>0.991492</td>
      <td>0.991133</td>
    </tr>
    <tr>
      <th>Chivas Regal Ultis, 40%</th>
      <td>0.990698</td>
      <td>0.990051</td>
      <td>0.991137</td>
      <td>0.985396</td>
      <td>1.000000</td>
      <td>0.982043</td>
      <td>0.985321</td>
      <td>0.989912</td>
      <td>0.982702</td>
      <td>0.988357</td>
      <td>...</td>
      <td>0.988654</td>
      <td>0.970978</td>
      <td>0.955230</td>
      <td>0.985376</td>
      <td>0.972369</td>
      <td>0.988046</td>
      <td>0.984874</td>
      <td>0.969178</td>
      <td>0.981251</td>
      <td>0.980790</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 2135 columns</p>
</div>




```python
# save cosine similarity matrix as .csv
recs.to_csv('./recs.csv', index_label=False)
```


```python
# run this cell for reccomender
#read in cosine similarity matrix as .csv
import pandas as pd
recs=pd.read_csv('./recs.csv')
recs.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Johnnie Walker Blue Label, 40%</th>
      <th>Black Bowmore, 1964 vintage, 42 year old, 40.5%</th>
      <th>Bowmore 46 year old (distilled 1964), 42.9%</th>
      <th>Compass Box The General, 53.4%</th>
      <th>Chivas Regal Ultis, 40%</th>
      <th>Ardbeg Corryvreckan, 57.1%</th>
      <th>Gold Bowmore, 1964 vintage, 42.4%</th>
      <th>Bowmore, 40 year old, 44.8%</th>
      <th>The Dalmore, 50 year old, 52.8%</th>
      <th>Glenfarclas Family Casks 1954 Cask #1260, 47.2%</th>
      <th>...</th>
      <th>Islay Mist, 8 year old, 40%</th>
      <th>The Singleton of Dufftown 28 year old, 52.3%</th>
      <th>Clan Denny (distilled at Girvan) 1992 21 year old HH9451, 59.6%</th>
      <th>Douglas Laing Single Minded (distilled at Jura) 8 year old, 41.5%</th>
      <th>Single Malts of Scotland (distilled at Craigellachie) 1996, 52.7%</th>
      <th>High Commissioner, 40%</th>
      <th>The Arran Malt, 43%</th>
      <th>Bowmore, 16 year old, 1990 vintage, 53.8%</th>
      <th>Bruichladdich 'Waves', 46%</th>
      <th>Inchmurrin 15 year old, 46%</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Johnnie Walker Blue Label, 40%</th>
      <td>1.000000</td>
      <td>0.990170</td>
      <td>0.991265</td>
      <td>0.984664</td>
      <td>0.990698</td>
      <td>0.981290</td>
      <td>0.985275</td>
      <td>0.989971</td>
      <td>0.982504</td>
      <td>0.988512</td>
      <td>...</td>
      <td>0.988273</td>
      <td>0.970474</td>
      <td>0.954490</td>
      <td>0.985258</td>
      <td>0.972248</td>
      <td>0.987454</td>
      <td>0.984687</td>
      <td>0.968607</td>
      <td>0.980744</td>
      <td>0.980909</td>
    </tr>
    <tr>
      <th>Black Bowmore, 1964 vintage, 42 year old, 40.5%</th>
      <td>0.990170</td>
      <td>1.000000</td>
      <td>0.991693</td>
      <td>0.984868</td>
      <td>0.990051</td>
      <td>0.983391</td>
      <td>0.987846</td>
      <td>0.991289</td>
      <td>0.985258</td>
      <td>0.989199</td>
      <td>...</td>
      <td>0.989604</td>
      <td>0.972514</td>
      <td>0.956911</td>
      <td>0.987694</td>
      <td>0.974304</td>
      <td>0.988040</td>
      <td>0.986691</td>
      <td>0.971231</td>
      <td>0.982220</td>
      <td>0.983631</td>
    </tr>
    <tr>
      <th>Bowmore 46 year old (distilled 1964), 42.9%</th>
      <td>0.991265</td>
      <td>0.991693</td>
      <td>1.000000</td>
      <td>0.988023</td>
      <td>0.991137</td>
      <td>0.986286</td>
      <td>0.987828</td>
      <td>0.992785</td>
      <td>0.986493</td>
      <td>0.991293</td>
      <td>...</td>
      <td>0.990743</td>
      <td>0.977084</td>
      <td>0.962042</td>
      <td>0.989244</td>
      <td>0.978558</td>
      <td>0.990453</td>
      <td>0.989288</td>
      <td>0.975499</td>
      <td>0.986046</td>
      <td>0.986205</td>
    </tr>
    <tr>
      <th>Compass Box The General, 53.4%</th>
      <td>0.984664</td>
      <td>0.984868</td>
      <td>0.988023</td>
      <td>1.000000</td>
      <td>0.985396</td>
      <td>0.992465</td>
      <td>0.982754</td>
      <td>0.989678</td>
      <td>0.990245</td>
      <td>0.990552</td>
      <td>...</td>
      <td>0.991720</td>
      <td>0.987148</td>
      <td>0.977797</td>
      <td>0.990216</td>
      <td>0.988772</td>
      <td>0.991015</td>
      <td>0.991874</td>
      <td>0.987393</td>
      <td>0.991492</td>
      <td>0.991133</td>
    </tr>
    <tr>
      <th>Chivas Regal Ultis, 40%</th>
      <td>0.990698</td>
      <td>0.990051</td>
      <td>0.991137</td>
      <td>0.985396</td>
      <td>1.000000</td>
      <td>0.982043</td>
      <td>0.985321</td>
      <td>0.989912</td>
      <td>0.982702</td>
      <td>0.988357</td>
      <td>...</td>
      <td>0.988654</td>
      <td>0.970978</td>
      <td>0.955230</td>
      <td>0.985376</td>
      <td>0.972369</td>
      <td>0.988046</td>
      <td>0.984874</td>
      <td>0.969178</td>
      <td>0.981251</td>
      <td>0.980790</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 2135 columns</p>
</div>




```python
# run cell input a name of a Scotch, see the top five similar Scotches, based on cosine similarity
search = input()
if len(recs[recs.columns.str.contains(search)].index) == 0:
        print ("Not in Directory")
else:
    for sco in recs[recs.columns.str.contains(search)].index:
        print(sco)
        print("")
        print("Similar Scotches")
        print(recs[sco].sort_values(ascending=False)[1:6])
        print("")
        print("")
```

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
    
    

