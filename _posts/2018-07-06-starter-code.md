
# Using Reddit's API for Predicting Comments

# Executive Summary

   Reddit has become one of the centers of the internet.  Over 150 million people used reddit just in the last month.  This is a huge community of people looking to engage in whatever topics one can think of.   Having a trending or ‘Hot’ post on reddit can really up the views on a post.  Everyone wants to see what everyone else is seeing, no one wants to miss out on anything in this digital age.
   
   Knowing what features, whether they be numerical or categorical, that can lead to having a hot post on reddit can be a huge advantage for people and businesses looking to gain exposure.  Using the reddit api, nlp and classification models one can gain the edge they need for their post to be ‘Hot’.
   
   First off, the numerical data.  The amount of time the post has been up on reddit and the number of cross-posts.  A ‘Hot’ post has an average time of being up on reddit for 9.96 hours.  If you are thinking about whether to cross-post or not, it is better to cross-post than to not cross-post. Using nlp techniques I look at words as data. When looking at the subreddits, the top 3 subreddits for a ‘Hot’ post are gaming, funny, and askreddit.  When looking at words in a post title that are in ‘Hot’ posts, the top three are, ‘need’, ‘nice’ and ‘true’. Overall the numerical data lead to better results than the nlp data.
   
   These results only enhance what we can learn about what factors lead to a ‘Hot’ post on reddit.  Finding out what makes a post, ‘Hot’ can be vital information to anyone trying to make it in social media or a business trying to be the first to capitalize on a new trend. 



# Scraping Reddit
Utilizing the reddit api our goal is to predict wheather a comment is hot or not. A 'Hot' post is one that exceeds the median number of comments from the pulled reddit posts. If a post is below the median number of coments it is categorized as a 'Not'. The code below will exhibit how the posts were pulled from the reddit api and the different models utilzed to examine the posts. 


```python
import requests
import pandas as pd
import json

import numpy as np
```


```python
URL = "http://www.reddit.com/hot.json"
```


```python
## YOUR CODE HERE
res = requests.get(URL, headers={'User-agent': 'Scott Sanowitz Bot 0.1'})
```

#### Use `res.json()` to convert the response into a dictionary format and set this to a variable. 

```python
data = res.json()
```


```python
data = res.json()

print(len(data['data']['children']))
```

    25


#### Getting more results

By default, Reddit will give you the top 25 posts:

```python
print(len(data['data']['children']))
```

If you want more, you'll need to do two things:
1. Get the name of the last post: `data['data']['after']`
2. Use that name to hit the following url: `http://www.reddit.com/hot.json?after=THE_AFTER_FROM_STEP_1`
3. Create a loop to repeat steps 1 and 2 until you have a sufficient number of posts. 

*NOTE*: Reddit will limit the number of requests per second you're allowed to make. When you create your loop, be sure to add the following after each iteration.

```python
time.sleep(3) # sleeps 3 seconds before continuing```

This will throttle your loop and keep you within Reddit's guidelines. You'll need to import the `time` library for this to work!


```python
import time
after = data['data']['after']
print(after)
URL = 'http://www.reddit.com/hot.json?count={}&'.format(25) + after
res = requests.get(URL, headers={'User-agent': 'Scott Sanowitz Bot 0.1'})
data = res.json()

```

    t3_8o8bct



```python
type(URL)

```




    str




```python
data['data']['after']
```




    't3_8mqsdi'



# Pulling posts
The code below was used to pull posts from the reddit api. Each time one pulls from the reddit api one gets 25 posts. This code allows someone to pull multiple times from the reddit api. If in this particular example I pulled 150 times but got a ErrNo 50, meaning the network is down, at 123 pulls. Giving me 3075 posts from reddit.


```python
## YOUR CODE HERE courtesy of Riley
posts = []
after = None
headers = {'User-agent': 'Scott Sanowitz bot 0.1'}
for i in range(150):
    print(i)
    if after == None:
        params = {}
    else:
        params = {'after': after}
    url = 'https://www.reddit.com/hot.json'
    res = requests.get(url, params=params, headers=headers)
    if res.status_code == 200:
        the_json = res.json()
        posts.extend(the_json['data']['children'])
        after = the_json['data']['after']
    else:
        print(res.status_code)
        break
    time.sleep(3)

```

    0
    1
    2
    3
    4
    5
    6
    7
    8
    9
    10
    11
    12
    13
    14
    15
    16
    17
    18
    19
    20
    21
    22
    23
    24
    25
    26
    27
    28
    29
    30
    31
    32
    33
    34
    35
    36
    37
    38
    39
    40
    41
    42
    43
    44
    45
    46
    47
    48
    49
    50
    51
    52
    53
    54
    55
    56
    57
    58
    59
    60
    61
    62
    63
    64
    65
    66
    67
    68
    69
    70
    71
    72
    73
    74
    75
    76
    77
    78
    79
    80
    81
    82
    83
    84
    85
    86
    87
    88
    89
    90
    91
    92
    93
    94
    95
    96
    97
    98
    99
    100
    101
    102
    103
    104
    105
    106
    107
    108
    109
    110
    111
    112
    113
    114
    115
    116
    117
    118
    119
    120
    121
    122
    123



    ---------------------------------------------------------------------------

    OSError                                   Traceback (most recent call last)

    ~/anaconda3/lib/python3.6/site-packages/urllib3/connection.py in _new_conn(self)
        140             conn = connection.create_connection(
    --> 141                 (self.host, self.port), self.timeout, **extra_kw)
        142 


    ~/anaconda3/lib/python3.6/site-packages/urllib3/util/connection.py in create_connection(address, timeout, source_address, socket_options)
         82     if err is not None:
    ---> 83         raise err
         84 


    ~/anaconda3/lib/python3.6/site-packages/urllib3/util/connection.py in create_connection(address, timeout, source_address, socket_options)
         72                 sock.bind(source_address)
    ---> 73             sock.connect(sa)
         74             return sock


    OSError: [Errno 50] Network is down

    
    During handling of the above exception, another exception occurred:


    NewConnectionError                        Traceback (most recent call last)

    ~/anaconda3/lib/python3.6/site-packages/urllib3/connectionpool.py in urlopen(self, method, url, body, headers, retries, redirect, assert_same_host, timeout, pool_timeout, release_conn, chunked, body_pos, **response_kw)
        600                                                   body=body, headers=headers,
    --> 601                                                   chunked=chunked)
        602 


    ~/anaconda3/lib/python3.6/site-packages/urllib3/connectionpool.py in _make_request(self, conn, method, url, timeout, chunked, **httplib_request_kw)
        345         try:
    --> 346             self._validate_conn(conn)
        347         except (SocketTimeout, BaseSSLError) as e:


    ~/anaconda3/lib/python3.6/site-packages/urllib3/connectionpool.py in _validate_conn(self, conn)
        849         if not getattr(conn, 'sock', None):  # AppEngine might not have  `.sock`
    --> 850             conn.connect()
        851 


    ~/anaconda3/lib/python3.6/site-packages/urllib3/connection.py in connect(self)
        283         # Add certificate verification
    --> 284         conn = self._new_conn()
        285 


    ~/anaconda3/lib/python3.6/site-packages/urllib3/connection.py in _new_conn(self)
        149             raise NewConnectionError(
    --> 150                 self, "Failed to establish a new connection: %s" % e)
        151 


    NewConnectionError: <urllib3.connection.VerifiedHTTPSConnection object at 0x10af22da0>: Failed to establish a new connection: [Errno 50] Network is down

    
    During handling of the above exception, another exception occurred:


    MaxRetryError                             Traceback (most recent call last)

    ~/anaconda3/lib/python3.6/site-packages/requests/adapters.py in send(self, request, stream, timeout, verify, cert, proxies)
        439                     retries=self.max_retries,
    --> 440                     timeout=timeout
        441                 )


    ~/anaconda3/lib/python3.6/site-packages/urllib3/connectionpool.py in urlopen(self, method, url, body, headers, retries, redirect, assert_same_host, timeout, pool_timeout, release_conn, chunked, body_pos, **response_kw)
        638             retries = retries.increment(method, url, error=e, _pool=self,
    --> 639                                         _stacktrace=sys.exc_info()[2])
        640             retries.sleep()


    ~/anaconda3/lib/python3.6/site-packages/urllib3/util/retry.py in increment(self, method, url, response, error, _pool, _stacktrace)
        387         if new_retry.is_exhausted():
    --> 388             raise MaxRetryError(_pool, url, error or ResponseError(cause))
        389 


    MaxRetryError: HTTPSConnectionPool(host='www.reddit.com', port=443): Max retries exceeded with url: /hot.json?after=t3_8mqt55 (Caused by NewConnectionError('<urllib3.connection.VerifiedHTTPSConnection object at 0x10af22da0>: Failed to establish a new connection: [Errno 50] Network is down',))

    
    During handling of the above exception, another exception occurred:


    ConnectionError                           Traceback (most recent call last)

    <ipython-input-11-e223b14eea1b> in <module>()
         10         params = {'after': after}
         11     url = 'https://www.reddit.com/hot.json'
    ---> 12     res = requests.get(url, params=params, headers=headers)
         13     if res.status_code == 200:
         14         the_json = res.json()


    ~/anaconda3/lib/python3.6/site-packages/requests/api.py in get(url, params, **kwargs)
         70 
         71     kwargs.setdefault('allow_redirects', True)
    ---> 72     return request('get', url, params=params, **kwargs)
         73 
         74 


    ~/anaconda3/lib/python3.6/site-packages/requests/api.py in request(method, url, **kwargs)
         56     # cases, and look like a memory leak in others.
         57     with sessions.Session() as session:
    ---> 58         return session.request(method=method, url=url, **kwargs)
         59 
         60 


    ~/anaconda3/lib/python3.6/site-packages/requests/sessions.py in request(self, method, url, params, data, headers, cookies, files, auth, timeout, allow_redirects, proxies, hooks, stream, verify, cert, json)
        506         }
        507         send_kwargs.update(settings)
    --> 508         resp = self.send(prep, **send_kwargs)
        509 
        510         return resp


    ~/anaconda3/lib/python3.6/site-packages/requests/sessions.py in send(self, request, **kwargs)
        616 
        617         # Send the request
    --> 618         r = adapter.send(request, **kwargs)
        619 
        620         # Total elapsed time of the request (approximately)


    ~/anaconda3/lib/python3.6/site-packages/requests/adapters.py in send(self, request, stream, timeout, verify, cert, proxies)
        506                 raise SSLError(e, request=request)
        507 
    --> 508             raise ConnectionError(e, request=request)
        509 
        510         except ClosedPoolError as e:


    ConnectionError: HTTPSConnectionPool(host='www.reddit.com', port=443): Max retries exceeded with url: /hot.json?after=t3_8mqt55 (Caused by NewConnectionError('<urllib3.connection.VerifiedHTTPSConnection object at 0x10af22da0>: Failed to establish a new connection: [Errno 50] Network is down',))



```python
len(posts)
    


```




    3075



This code lets me know if any posts pulled were duplicates. There is evidence that some of my posts are duplicates due to a difference in number from 3075 to 2792.


```python
len(set([p['data']['name'] for p in posts]))
```




    2792




```python
posts[0]['data']['title'] # title
posts[0]['data']['subreddit'] #subreddit
time.time() - posts[0]['data']['created_utc'] #time UTC on reddit when pulled
posts[0]['data']['num_comments'] # # of comments
posts[0]['data']['is_video'] # is it a video
posts[0]['data']['num_crossposts'] # # of crossposts: same post in different subreddits
```




    1



# List of dictionaries
This code makes a list of dictionaries of the chosen features from the reddit posts. Seen below are the title of the reddit post, the subbreddit the reddit post belonged to, the time the reddit post has been up on reddit, the number of comments the reddit post has, the number of crossposts the reddit post has and if the rediit post is a video or not. These features have been made into a list of dictionaries so that they can be easily made into a pandas dataframe and saved to as a csv file.


```python
# make a list of dictionaries
infos = []
for p in posts:
    info = {}
    info['title'] = p['data']['title']
    info['subreddit'] = p['data']['subreddit']
    info['time_elasped'] = time.time() - p['data']['created_utc']
    info['num_comments'] = p['data']['num_comments']
    info['num_crossposts'] = p['data']['num_crossposts']
    info['is_video'] = p['data']['is_video']
    infos.append(info)
    
    
```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    <ipython-input-2-9391d5a4972e> in <module>()
          1 # make a list of dictionaries
          2 infos = []
    ----> 3 for p in posts:
          4     info = {}
          5     info['title'] = p['data']['title']


    NameError: name 'posts' is not defined



```python
infos
```




    [{'is_video': False,
      'num_comments': 191,
      'num_crossposts': 1,
      'subreddit': 'aww',
      'time_elasped': 20833.514323949814,
      'title': 'A bit overprotective'},
     {'is_video': False,
      'num_comments': 275,
      'num_crossposts': 0,
      'subreddit': 'pics',
      'time_elasped': 18743.514327049255,
      'title': 'Let us never forget.'},
     {'is_video': False,
      'num_comments': 1027,
      'num_crossposts': 1,
      'subreddit': 'FortNiteBR',
      'time_elasped': 18909.514329195023,
      'title': 'New updates/ rideable shopping carts'},
     {'is_video': False,
      'num_comments': 650,
      'num_crossposts': 1,
      'subreddit': 'funny',
      'time_elasped': 26945.514330863953,
      'title': 'Dave Bautista has achieved full Drax.'},
     {'is_video': False,
      'num_comments': 66,
      'num_crossposts': 1,
      'subreddit': 'AnimalsBeingBros',
      'time_elasped': 21782.51433300972,
      'title': 'Hey Human, Want a Treat?'},
     {'is_video': False,
      'num_comments': 179,
      'num_crossposts': 0,
      'subreddit': 'BlackPeopleTwitter',
      'time_elasped': 27663.51433491707,
      'title': 'Old HObits die hard...'},
     {'is_video': False,
      'num_comments': 277,
      'num_crossposts': 3,
      'subreddit': 'news',
      'time_elasped': 28833.514337062836,
      'title': '94-yr old WW II veteran gets high school diploma 74 years after dropping out to serve'},
     {'is_video': False,
      'num_comments': 699,
      'num_crossposts': 2,
      'subreddit': 'gaming',
      'time_elasped': 27438.51433801651,
      'title': 'Not even safe in Super Mario 64...'},
     {'is_video': False,
      'num_comments': 741,
      'num_crossposts': 36,
      'subreddit': 'gifs',
      'time_elasped': 31305.514340162277,
      'title': 'rapid-fire cigar box juggling (sort of looks like he has 3 hands'},
     {'is_video': False,
      'num_comments': 934,
      'num_crossposts': 1,
      'subreddit': 'worldnews',
      'time_elasped': 32248.514341831207,
      'title': 'European Union moves to ban single-use plastics.'},
     {'is_video': False,
      'num_comments': 354,
      'num_crossposts': 2,
      'subreddit': 'MovieDetails',
      'time_elasped': 32907.5143430233,
      'title': "Terminator 2 (1991) had some practical effects with the help of Linda Hamilton's twin sister. When Sarah cuts a hole in T-800 head it's a model of Schwarzenegger’s head in the foreground, the real Schwarzenegger plays his own reflection, and Linda’s twin sister mimics her moves"},
     {'is_video': False,
      'num_comments': 77,
      'num_crossposts': 2,
      'subreddit': 'TheLastAirbender',
      'time_elasped': 30418.51434493065,
      'title': 'New Yip...'},
     {'is_video': False,
      'num_comments': 418,
      'num_crossposts': 1,
      'subreddit': 'DunderMifflin',
      'time_elasped': 32719.514347076416,
      'title': 'I met a heart surgeon while shopping'},
     {'is_video': False,
      'num_comments': 364,
      'num_crossposts': 14,
      'subreddit': 'BetterEveryLoop',
      'time_elasped': 32787.514348983765,
      'title': 'This is majestic.'},
     {'is_video': False,
      'num_comments': 497,
      'num_crossposts': 1,
      'subreddit': 'todayilearned',
      'time_elasped': 33087.51435017586,
      'title': 'TIL that oregano was practically unheard of in the U.S. until American G.I.s in WWII returned from Italy with a taste for the "pizza herb"'},
     {'is_video': False,
      'num_comments': 144,
      'num_crossposts': 0,
      'subreddit': 'marvelstudios',
      'time_elasped': 20980.514352083206,
      'title': 'Masks and helmets in the MCU are on another level of awesome'},
     {'is_video': False,
      'num_comments': 2881,
      'num_crossposts': 9,
      'subreddit': 'mildlyinfuriating',
      'time_elasped': 33293.51435279846,
      'title': 'The hospital "helping"'},
     {'is_video': True,
      'num_comments': 497,
      'num_crossposts': 0,
      'subreddit': 'MMA',
      'time_elasped': 32217.51435494423,
      'title': 'An accurate summary of the Till vs. Thompson fight'},
     {'is_video': False,
      'num_comments': 2858,
      'num_crossposts': 4,
      'subreddit': 'trashy',
      'time_elasped': 31084.514355897903,
      'title': 'Thought this belonged here.'},
     {'is_video': False,
      'num_comments': 287,
      'num_crossposts': 3,
      'subreddit': 'woahdude',
      'time_elasped': 34759.51435804367,
      'title': 'Timelapse of a 3D printed iris box'},
     {'is_video': False,
      'num_comments': 443,
      'num_crossposts': 2,
      'subreddit': 'educationalgifs',
      'time_elasped': 32539.51435995102,
      'title': 'How a fire sprinkler works'},
     {'is_video': False,
      'num_comments': 268,
      'num_crossposts': 0,
      'subreddit': 'INEEEEDIT',
      'time_elasped': 27118.514361143112,
      'title': 'Pretty durable solar panel charger'},
     {'is_video': False,
      'num_comments': 70,
      'num_crossposts': 0,
      'subreddit': 'WhitePeopleTwitter',
      'time_elasped': 27931.514362812042,
      'title': 'Poor Nigel'},
     {'is_video': False,
      'num_comments': 9039,
      'num_crossposts': 0,
      'subreddit': 'movies',
      'time_elasped': 36054.514364004135,
      'title': "Box Office Week - Solo: A Star Wars Story debuts at #1 with a worrisome $83.3M domestic on an estimated budget of $250M-$300M. Worldwide it's even worse as the film debuted to a disastrous $65M international, less than what Deadpool 2 made internationally on its second weekend."},
     {'is_video': False,
      'num_comments': 249,
      'num_crossposts': 0,
      'subreddit': 'LateStageCapitalism',
      'time_elasped': 32310.514365911484,
      'title': 'truly a mystery'},
     {'is_video': False,
      'num_comments': 82,
      'num_crossposts': 3,
      'subreddit': 'nonononoyes',
      'time_elasped': 25855.51436805725,
      'title': 'That landing though'},
     {'is_video': False,
      'num_comments': 138,
      'num_crossposts': 2,
      'subreddit': 'comics',
      'time_elasped': 37862.5143699646,
      'title': 'musical plants [oc]'},
     {'is_video': False,
      'num_comments': 112,
      'num_crossposts': 0,
      'subreddit': 'blackmagicfuckery',
      'time_elasped': 29497.514372110367,
      'title': 'How?'},
     {'is_video': False,
      'num_comments': 734,
      'num_crossposts': 1,
      'subreddit': 'space',
      'time_elasped': 34898.5143737793,
      'title': "Hope that in our lifetimes and not when we're super old that we can witness the first manned Mars landing the same way the world watched a man land walk on the moon."},
     {'is_video': False,
      'num_comments': 313,
      'num_crossposts': 6,
      'subreddit': 'youseeingthisshit',
      'time_elasped': 35175.51437497139,
      'title': 'Not today'},
     {'is_video': False,
      'num_comments': 112,
      'num_crossposts': 0,
      'subreddit': 'disneyvacation',
      'time_elasped': 33264.51437687874,
      'title': "How to Cope When You're the Only One Who Realizes Gazebos Are Stupid"},
     {'is_video': False,
      'num_comments': 836,
      'num_crossposts': 1,
      'subreddit': 'Showerthoughts',
      'time_elasped': 35286.51437807083,
      'title': 'Shows with laugh tracks would be actually funny if the laugh track were replaced with the sound of just one guy laughing hysterically'},
     {'is_video': False,
      'num_comments': 163,
      'num_crossposts': 2,
      'subreddit': 'EmpireDidNothingWrong',
      'time_elasped': 34066.51437997818,
      'title': 'Its true'},
     {'is_video': False,
      'num_comments': 167,
      'num_crossposts': 1,
      'subreddit': 'FoodPorn',
      'time_elasped': 27352.514381170273,
      'title': 'Steak and Eggs wrapped with a fried cheese shell [925X960]'},
     {'is_video': False,
      'num_comments': 438,
      'num_crossposts': 2,
      'subreddit': 'ProgrammerHumor',
      'time_elasped': 36278.51438307762,
      'title': '3 Steps to enjoy life'},
     {'is_video': False,
      'num_comments': 1079,
      'num_crossposts': 2,
      'subreddit': 'pics',
      'time_elasped': 28240.51438498497,
      'title': 'Pakistan: Women Police officers escort man to jail for abusing his daughter, govt made sure only women would escort him to jail.'},
     {'is_video': False,
      'num_comments': 154,
      'num_crossposts': 0,
      'subreddit': 'NintendoSwitch',
      'time_elasped': 21723.514385938644,
      'title': "(E3 Hype) It's a long shot, but i'm hoping Animal Crossing Switch gets announced this year - so I drew some fan art!"},
     {'is_video': False,
      'num_comments': 150,
      'num_crossposts': 3,
      'subreddit': 'PandR',
      'time_elasped': 39135.51438808441,
      'title': 'Be as positive as Chris Traeger.'},
     {'is_video': False,
      'num_comments': 201,
      'num_crossposts': 4,
      'subreddit': 'oddlysatisfying',
      'time_elasped': 36327.51438999176,
      'title': 'The way her waistline aligns with the tide!'},
     {'is_video': False,
      'num_comments': 620,
      'num_crossposts': 3,
      'subreddit': 'Futurology',
      'time_elasped': 36946.514390945435,
      'title': 'Oil industry is finally starting to be affected by Norway’s rapid electric car adoption'},
     {'is_video': False,
      'num_comments': 330,
      'num_crossposts': 2,
      'subreddit': 'food',
      'time_elasped': 37435.5143930912,
      'title': '[Homemade] Seared Duck Breast'},
     {'is_video': False,
      'num_comments': 60,
      'num_crossposts': 0,
      'subreddit': 'AccidentalWesAnderson',
      'time_elasped': 36348.514394044876,
      'title': 'Hong Kong Playground by Ludwig Favre'},
     {'is_video': False,
      'num_comments': 772,
      'num_crossposts': 5,
      'subreddit': 'EarthPorn',
      'time_elasped': 40638.514395952225,
      'title': 'Just a cloud and a hill, Italy (OC)[1920x1920]'},
     {'is_video': False,
      'num_comments': 2102,
      'num_crossposts': 1,
      'subreddit': 'LifeProTips',
      'time_elasped': 40041.5143969059,
      'title': 'LPT: when your ISP raises your bill, call in and say “cancel service” to the automated operator. You’ll be sent to their retention team with no waiting on hold. They will usually take $10-20 off your monthly bill for a year. I do this once a year.'},
     {'is_video': False,
      'num_comments': 145,
      'num_crossposts': 0,
      'subreddit': 'PrequelMemes',
      'time_elasped': 36737.514399051666,
      'title': 'Is this true?'},
     {'is_video': False,
      'num_comments': 298,
      'num_crossposts': 2,
      'subreddit': 'Bitcoin',
      'time_elasped': 37151.514400959015,
      'title': 'There are 180 different scenarios where bitcoin go. If any one thing happens remember me i am the first one to predict this'},
     {'is_video': False,
      'num_comments': 62,
      'num_crossposts': 1,
      'subreddit': 'HistoryMemes',
      'time_elasped': 27313.514402151108,
      'title': '"It\'s all Germany\'s fault"'},
     {'is_video': False,
      'num_comments': 225,
      'num_crossposts': 5,
      'subreddit': 'geek',
      'time_elasped': 34856.51440405846,
      'title': 'Making a knife from Lignum Vitae wood'},
     {'is_video': False,
      'num_comments': 440,
      'num_crossposts': 4,
      'subreddit': 'HighQualityGifs',
      'time_elasped': 40337.514405965805,
      'title': 'Cows watching yoga.'},
     {'is_video': True,
      'num_comments': 102,
      'num_crossposts': 6,
      'subreddit': 'instant_regret',
      'time_elasped': 25858.51440691948,
      'title': 'He just barely touched his tail'},
     {'is_video': False,
      'num_comments': 550,
      'num_crossposts': 0,
      'subreddit': 'LiverpoolFC',
      'time_elasped': 15172.514409065247,
      'title': 'Fabinho confirmed'},
     {'is_video': False,
      'num_comments': 32,
      'num_crossposts': 1,
      'subreddit': 'Thisismylifemeow',
      'time_elasped': 31860.514410972595,
      'title': 'Catfishing'},
     {'is_video': False,
      'num_comments': 158,
      'num_crossposts': 0,
      'subreddit': 'KenM',
      'time_elasped': 38190.51441311836,
      'title': 'Ken M on the Bible'},
     {'is_video': False,
      'num_comments': 1062,
      'num_crossposts': 1,
      'subreddit': 'HumansBeingBros',
      'time_elasped': 43346.51441502571,
      'title': 'Appreciation and gratefulness.'},
     {'is_video': False,
      'num_comments': 371,
      'num_crossposts': 6,
      'subreddit': 'quityourbullshit',
      'time_elasped': 40704.51441693306,
      'title': 'The Human Watch Speaks'},
     {'is_video': False,
      'num_comments': 94,
      'num_crossposts': 1,
      'subreddit': 'surrealmemes',
      'time_elasped': 23020.514417886734,
      'title': 'THEYЯE HΞЯΣ'},
     {'is_video': False,
      'num_comments': 76,
      'num_crossposts': 1,
      'subreddit': 'cringepics',
      'time_elasped': 17404.5144200325,
      'title': 'Debbie downer on a post about a recently sold house'},
     {'is_video': False,
      'num_comments': 189,
      'num_crossposts': 6,
      'subreddit': 'AnimalsBeingDerps',
      'time_elasped': 43038.51442193985,
      'title': 'Caught in the act'},
     {'is_video': False,
      'num_comments': 21,
      'num_crossposts': 1,
      'subreddit': 'boottoobig',
      'time_elasped': 29953.514424085617,
      'title': "Roses are red, don't hate on linguine,"},
     {'is_video': False,
      'num_comments': 214,
      'num_crossposts': 5,
      'subreddit': 'mildlyinteresting',
      'time_elasped': 34015.514425992966,
      'title': 'Friend took a photo of me swinging my shirt around and it looks like i tamed a crow to perch on my hand'},
     {'is_video': False,
      'num_comments': 46,
      'num_crossposts': 0,
      'subreddit': 'PeopleFuckingDying',
      'time_elasped': 32319.51442694664,
      'title': 'TuRtLe FucKInG DemoLisHeS PeRsoNs HeaRT'},
     {'is_video': False,
      'num_comments': 310,
      'num_crossposts': 4,
      'subreddit': 'photoshopbattles',
      'time_elasped': 42464.51442885399,
      'title': 'PsBattle: Turtle eating strawberry'},
     {'is_video': False,
      'num_comments': 65,
      'num_crossposts': 0,
      'subreddit': 'funny',
      'time_elasped': 25265.514430999756,
      'title': 'Is this the best face swap or what?'},
     {'is_video': False,
      'num_comments': 526,
      'num_crossposts': 4,
      'subreddit': 'OopsDidntMeanTo',
      'time_elasped': 40637.51443195343,
      'title': 'Oopsie'},
     {'is_video': False,
      'num_comments': 821,
      'num_crossposts': 10,
      'subreddit': 'interestingasfuck',
      'time_elasped': 42085.5144340992,
      'title': 'Stitchless healing'},
     {'is_video': False,
      'num_comments': 56,
      'num_crossposts': 0,
      'subreddit': 'wholesomememes',
      'time_elasped': 38440.514436006546,
      'title': 'Very important message'},
     {'is_video': False,
      'num_comments': 828,
      'num_crossposts': 0,
      'subreddit': 'MemeEconomy',
      'time_elasped': 15628.514437198639,
      'title': 'This format can work for a variety of things. A smart invest indeed!'},
     {'is_video': False,
      'num_comments': 860,
      'num_crossposts': 3,
      'subreddit': 'FunnyandSad',
      'time_elasped': 42508.51443886757,
      'title': 'Its more true than sad but I thought it belonged here'},
     {'is_video': False,
      'num_comments': 130,
      'num_crossposts': 0,
      'subreddit': 'anime_irl',
      'time_elasped': 34172.51443982124,
      'title': 'anime_irl'},
     {'is_video': False,
      'num_comments': 614,
      'num_crossposts': 2,
      'subreddit': 'aww',
      'time_elasped': 27578.51444196701,
      'title': "Couple with Down's syndrome celebrate 22 years of marriage."},
     {'is_video': False,
      'num_comments': 41,
      'num_crossposts': 0,
      'subreddit': 'youdontsurf',
      'time_elasped': 35999.51444411278,
      'title': 'Game night ruined'},
     {'is_video': False,
      'num_comments': 50,
      'num_crossposts': 0,
      'subreddit': 'nostalgia',
      'time_elasped': 17018.514445066452,
      'title': 'The family computer'},
     {'is_video': False,
      'num_comments': 86,
      'num_crossposts': 0,
      'subreddit': 'justneckbeardthings',
      'time_elasped': 22269.51444721222,
      'title': '*tips fedora at you*'},
     {'is_video': False,
      'num_comments': 115,
      'num_crossposts': 4,
      'subreddit': 'DesignPorn',
      'time_elasped': 42207.51444888115,
      'title': 'Park exhibit'},
     {'is_video': False,
      'num_comments': 70,
      'num_crossposts': 4,
      'subreddit': 'meirl',
      'time_elasped': 41085.51444983482,
      'title': 'me_iRl'},
     {'is_video': False,
      'num_comments': 344,
      'num_crossposts': 0,
      'subreddit': 'IAmA',
      'time_elasped': 32679.51445198059,
      'title': "IAmA science journalist who has spent the past year visiting every lab that has discovered a chemical element since 1945. I've traveled 60,000 miles and I still goof about around particle accelerators. AMA."},
     {'is_video': False,
      'num_comments': 125,
      'num_crossposts': 3,
      'subreddit': 'rarepuppers',
      'time_elasped': 43582.51445412636,
      'title': 'Do(t)go'},
     {'is_video': False,
      'num_comments': 82,
      'num_crossposts': 1,
      'subreddit': 'gifs',
      'time_elasped': 20186.514456033707,
      'title': 'Nice trick'},
     {'is_video': False,
      'num_comments': 28,
      'num_crossposts': 0,
      'subreddit': 'hardcoreaww',
      'time_elasped': 35059.514458179474,
      'title': "Can I help you? If you are looking for my mom, she already knows you're here"},
     {'is_video': False,
      'num_comments': 38,
      'num_crossposts': 0,
      'subreddit': 'memes',
      'time_elasped': 19522.51445889473,
      'title': 'Numbuh 1'},
     {'is_video': False,
      'num_comments': 55,
      'num_crossposts': 5,
      'subreddit': 'BikiniBottomTwitter',
      'time_elasped': 32979.5144610405,
      'title': 'Every time man😢'},
     {'is_video': False,
      'num_comments': 210,
      'num_crossposts': 1,
      'subreddit': 'gaming',
      'time_elasped': 28383.514462947845,
      'title': 'Trying to make friends as an adult'},
     {'is_video': False,
      'num_comments': 3076,
      'num_crossposts': 3,
      'subreddit': 'technology',
      'time_elasped': 46109.51446390152,
      'title': 'Bitcoin backlash as ‘miners’ suck up electricity, stress power grids in Central Washington'},
     {'is_video': False,
      'num_comments': 196,
      'num_crossposts': 0,
      'subreddit': 'evilbuildings',
      'time_elasped': 43349.51446604729,
      'title': 'This telescope looks like it could be part of The Death Star'},
     {'is_video': False,
      'num_comments': 226,
      'num_crossposts': 0,
      'subreddit': 'JusticeServed',
      'time_elasped': 30534.514468193054,
      'title': 'Apollo 14 astronaut Ed Mitchell literally kicks the ass of a moon landing denier who uses forged History Channel credentials to enter his home, ambush him, and call him satanic in front of his kid'},
     {'is_video': False,
      'num_comments': 124,
      'num_crossposts': 2,
      'subreddit': 'creepy',
      'time_elasped': 32093.51446914673,
      'title': 'secare'},
     {'is_video': False,
      'num_comments': 1012,
      'num_crossposts': 5,
      'subreddit': 'reactiongifs',
      'time_elasped': 41745.51447081566,
      'title': 'MRW I make her squirt'},
     {'is_video': False,
      'num_comments': 143,
      'num_crossposts': 0,
      'subreddit': 'tattoos',
      'time_elasped': 39866.514472961426,
      'title': 'My new calf piece. Artist: Tony Gacci/Blu Gorilla/Charleston, SC.'},
     {'is_video': False,
      'num_comments': 123,
      'num_crossposts': 1,
      'subreddit': 'dogswithjobs',
      'time_elasped': 45587.5144739151,
      'title': 'L A Z Y B O I'},
     {'is_video': False,
      'num_comments': 33,
      'num_crossposts': 3,
      'subreddit': 'CatSlaps',
      'time_elasped': 32840.51447606087,
      'title': '"Hey! Occupied!"'},
     {'is_video': False,
      'num_comments': 597,
      'num_crossposts': 0,
      'subreddit': 'MURICA',
      'time_elasped': 39410.514477968216,
      'title': 'God bless America and all those who died to protect her'},
     {'is_video': False,
      'num_comments': 98,
      'num_crossposts': 2,
      'subreddit': 'itookapicture',
      'time_elasped': 40646.51447916031,
      'title': 'ITAP of a sunset through a window'},
     {'is_video': False,
      'num_comments': 153,
      'num_crossposts': 4,
      'subreddit': 'therewasanattempt',
      'time_elasped': 43783.51448082924,
      'title': 'to exercise with a swing'},
     {'is_video': False,
      'num_comments': 262,
      'num_crossposts': 0,
      'subreddit': 'bestof',
      'time_elasped': 39212.51448202133,
      'title': "/u/Not_A_BusDriver explains why censoring offensive history diminishes it's importance of reminding us of past atrocities (and to not repeat them)."},
     {'is_video': False,
      'num_comments': 811,
      'num_crossposts': 6,
      'subreddit': 'NatureIsFuckingLit',
      'time_elasped': 47224.51448392868,
      'title': '🔥 Camels can eat cactuses! Their cheeks are lined with keratinized cones that protect them from damage called papillae 🔥'},
     {'is_video': False,
      'num_comments': 91,
      'num_crossposts': 2,
      'subreddit': 'Damnthatsinteresting',
      'time_elasped': 40361.514484882355,
      'title': 'Dogs master the art of the mannequin.'},
     {'is_video': False,
      'num_comments': 98,
      'num_crossposts': 0,
      'subreddit': 'niceguys',
      'time_elasped': 42022.51448702812,
      'title': 'Gotta love Prequel Memes.'},
     {'is_video': False,
      'num_comments': 331,
      'num_crossposts': 0,
      'subreddit': 'books',
      'time_elasped': 21397.51448917389,
      'title': "I know I'm late to the party but Terry Pratchett is brilliant."},
     {'is_video': False,
      'num_comments': 256,
      'num_crossposts': 1,
      'subreddit': 'Art',
      'time_elasped': 43170.51449012756,
      'title': 'Wheres the coffee, digital drawing, 6600x 10200px'},
     {'is_video': False,
      'num_comments': 332,
      'num_crossposts': 0,
      'subreddit': 'Unexpected',
      'time_elasped': 41314.51449179649,
      'title': "Hey, Becky, let's go shopping and meet cute guys!"},
     {'is_video': False,
      'num_comments': 88,
      'num_crossposts': 0,
      'subreddit': 'travel',
      'time_elasped': 37661.51449394226,
      'title': 'Our honeymoon view in Santorini'},
     {'is_video': False,
      'num_comments': 58,
      'num_crossposts': 2,
      'subreddit': '2healthbars',
      'time_elasped': 35419.514494895935,
      'title': 'Unbearable'},
     {'is_video': False,
      'num_comments': 128,
      'num_crossposts': 0,
      'subreddit': 'dankchristianmemes',
      'time_elasped': 47326.5144970417,
      'title': 'Sorry momma'},
     {'is_video': False,
      'num_comments': 215,
      'num_crossposts': 0,
      'subreddit': 'nevertellmetheodds',
      'time_elasped': 42341.51449918747,
      'title': 'that curve'},
     {'is_video': False,
      'num_comments': 56,
      'num_crossposts': 2,
      'subreddit': 'BeAmazed',
      'time_elasped': 39001.514500141144,
      'title': '2D Drawing.'},
     {'is_video': False,
      'num_comments': 116,
      'num_crossposts': 1,
      'subreddit': '4PanelCringe',
      'time_elasped': 26305.514501810074,
      'title': 'Oh no'},
     {'is_video': False,
      'num_comments': 49,
      'num_crossposts': 2,
      'subreddit': 'yesyesyesyesno',
      'time_elasped': 40830.51450300217,
      'title': 'Slap Happy.'},
     {'is_video': False,
      'num_comments': 818,
      'num_crossposts': 2,
      'subreddit': 'OldSchoolCool',
      'time_elasped': 47612.514504909515,
      'title': 'Outsiders 1983'},
     {'is_video': False,
      'num_comments': 104,
      'num_crossposts': 1,
      'subreddit': 'coaxedintoasnafu',
      'time_elasped': 40726.51450705528,
      'title': 'too real...'},
     {'is_video': False,
      'num_comments': 343,
      'num_crossposts': 2,
      'subreddit': 'dataisbeautiful',
      'time_elasped': 40415.51450896263,
      'title': 'A plot of objects which were found to be "the most distant object" by astronomers over time [OC]'},
     {'is_video': False,
      'num_comments': 84,
      'num_crossposts': 1,
      'subreddit': 'DnDGreentext',
      'time_elasped': 36186.514510154724,
      'title': 'The DM copies OGLAF'},
     {'is_video': False,
      'num_comments': 127,
      'num_crossposts': 0,
      'subreddit': 'ChoosingBeggars',
      'time_elasped': 28646.514511823654,
      'title': "Never thought I'd post here but, how does 1$ a design sound?"},
     {'is_video': False,
      'num_comments': 217,
      'num_crossposts': 0,
      'subreddit': 'thatHappened',
      'time_elasped': 40356.51451396942,
      'title': 'And then the train conductor clapped...'},
     {'is_video': False,
      'num_comments': 296,
      'num_crossposts': 0,
      'subreddit': 'xboxone',
      'time_elasped': 33810.514514923096,
      'title': 'Battlefield V: "There are no more Battlepacks. Instead, players will be able to choose their rewards directly or through rank up events."'},
     {'is_video': False,
      'num_comments': 37,
      'num_crossposts': 0,
      'subreddit': 'bigboye',
      'time_elasped': 42615.51451706886,
      'title': 'Big Tank Puppy'},
     {'is_video': False,
      'num_comments': 41,
      'num_crossposts': 0,
      'subreddit': 'brooklynninenine',
      'time_elasped': 42312.51451897621,
      'title': "That's not what a vasectomy is"},
     {'is_video': False,
      'num_comments': 140,
      'num_crossposts': 1,
      'subreddit': 'tumblr',
      'time_elasped': 43743.514520168304,
      'title': 'Robin Williams, Hogwarts professor'},
     {'is_video': False,
      'num_comments': 807,
      'num_crossposts': 1,
      'subreddit': 'PoliticalHumor',
      'time_elasped': 44942.51452207565,
      'title': 'Opt To Miss Prime.'},
     {'is_video': False,
      'num_comments': 67,
      'num_crossposts': 0,
      'subreddit': 'misleadingthumbnails',
      'time_elasped': 42055.51452279091,
      'title': 'This hole right in the middle of my kitchen.'},
     {'is_video': False,
      'num_comments': 41,
      'num_crossposts': 0,
      'subreddit': 'PixelArt',
      'time_elasped': 35867.514524936676,
      'title': 'Three Bears [JFS] | artist: prettyprettypixels'},
     {'is_video': False,
      'num_comments': 167,
      'num_crossposts': 0,
      'subreddit': 'Tinder',
      'time_elasped': 44855.51452589035,
      'title': 'I Got Rekt'},
     {'is_video': False,
      'num_comments': 836,
      'num_crossposts': 2,
      'subreddit': 'Justfuckmyshitup',
      'time_elasped': 47825.51452803612,
      'title': 'Gimme that Elvis caught in a vacuum cleaner.'},
     {'is_video': True,
      'num_comments': 90,
      'num_crossposts': 1,
      'subreddit': 'Perfectfit',
      'time_elasped': 35602.514529943466,
      'title': 'The perfect gap between...!'},
     {'is_video': False,
      'num_comments': 37,
      'num_crossposts': 2,
      'subreddit': 'happycowgifs',
      'time_elasped': 31833.514532089233,
      'title': 'Baby cow and little girl snuggling together.'},
     {'is_video': False,
      'num_comments': 55,
      'num_crossposts': 2,
      'subreddit': 'crappyoffbrands',
      'time_elasped': 32559.514533042908,
      'title': 'Morge'},
     {'is_video': False,
      'num_comments': 1347,
      'num_crossposts': 4,
      'subreddit': 'videos',
      'time_elasped': 45879.514534950256,
      'title': 'How to stop your mate smoking'},
     {'is_video': True,
      'num_comments': 221,
      'num_crossposts': 1,
      'subreddit': 'holdmyredbull',
      'time_elasped': 42631.514536857605,
      'title': 'The stunts in the second half are insane'},
     {'is_video': False,
      'num_comments': 276,
      'num_crossposts': 1,
      'subreddit': 'creepyasterisks',
      'time_elasped': 47080.51453995705,
      'title': 'Does this count?'},
     {'is_video': False,
      'num_comments': 280,
      'num_crossposts': 0,
      'subreddit': 'SuddenlyGay',
      'time_elasped': 45865.514542102814,
      'title': 'And they were roomates'},
     {'is_video': False,
      'num_comments': 206,
      'num_crossposts': 2,
      'subreddit': 'ATBGE',
      'time_elasped': 46748.51454305649,
      'title': 'I think this belongs here...'},
     {'is_video': False,
      'num_comments': 542,
      'num_crossposts': 0,
      'subreddit': 'madlads',
      'time_elasped': 45388.51454496384,
      'title': 'Absolute fucking madlad!'},
     {'is_video': False,
      'num_comments': 71,
      'num_crossposts': 1,
      'subreddit': 'HistoryPorn',
      'time_elasped': 31762.514546871185,
      'title': 'A captured Wehrmacht soldier identifies an SS trooper as one those who shot US Army prisoners in Malmedy, Belgium, during the “Battle of the Bulge”. These men were captured by the 3rd US Army near Passau, Germany.1945.[1000x758]'},
     {'is_video': False,
      'num_comments': 27,
      'num_crossposts': 1,
      'subreddit': 'holdmybeer',
      'time_elasped': 29244.514549016953,
      'title': 'HMB while I limbo this barrier'},
     {'is_video': False,
      'num_comments': 29,
      'num_crossposts': 0,
      'subreddit': 'southpark',
      'time_elasped': 25731.514549970627,
      'title': '"Why don\'t we just shoot him?"'},
     {'is_video': False,
      'num_comments': 32,
      'num_crossposts': 2,
      'subreddit': 'CozyPlaces',
      'time_elasped': 42091.514552116394,
      'title': 'A cozy room in the Alps'},
     {'is_video': False,
      'num_comments': 160,
      'num_crossposts': 2,
      'subreddit': 'trippinthroughtime',
      'time_elasped': 44386.51455402374,
      'title': 'When you a 40 y/o virgin..'},
     {'is_video': False,
      'num_comments': 49,
      'num_crossposts': 0,
      'subreddit': 'wtfstockphotos',
      'time_elasped': 31187.514554977417,
      'title': 'HMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMM'},
     {'is_video': False,
      'num_comments': 35,
      'num_crossposts': 0,
      'subreddit': 'OSHA',
      'time_elasped': 23698.514556884766,
      'title': 'Using newspaper as a welding mask'},
     {'is_video': False,
      'num_comments': 807,
      'num_crossposts': 0,
      'subreddit': 'television',
      'time_elasped': 48157.51455903053,
      'title': "'True Detective' Season 3 Will be More Like the First Season"},
     {'is_video': True,
      'num_comments': 194,
      'num_crossposts': 1,
      'subreddit': 'lifehacks',
      'time_elasped': 29770.51456093788,
      'title': 'Tired of being stiffed by the paper towel dispenser? Have I got a video for you!'},
     {'is_video': True,
      'num_comments': 34,
      'num_crossposts': 2,
      'subreddit': 'breakingbad',
      'time_elasped': 26134.51456308365,
      'title': 'For those who are in a rush.'},
     {'is_video': False,
      'num_comments': 205,
      'num_crossposts': 0,
      'subreddit': 'askscience',
      'time_elasped': 24507.514565229416,
      'title': "Does radioactive decay reduce an object's mass?"},
     {'is_video': False,
      'num_comments': 181,
      'num_crossposts': 3,
      'subreddit': 'CrappyDesign',
      'time_elasped': 50235.51456594467,
      'title': 'Another crappy design.'},
     {'is_video': False,
      'num_comments': 271,
      'num_crossposts': 1,
      'subreddit': 'MadeMeSmile',
      'time_elasped': 53144.51456785202,
      'title': "She's Back!"},
     {'is_video': False,
      'num_comments': 175,
      'num_crossposts': 0,
      'subreddit': 'FortNiteBR',
      'time_elasped': 16466.514569044113,
      'title': 'At this moment RiceGum knew he fukt up. His hand is off the controller but not on the hand cam. This post is late but better late than never.'},
     {'is_video': False,
      'num_comments': 37,
      'num_crossposts': 0,
      'subreddit': 'thanosdidnothingwrong',
      'time_elasped': 30194.51457095146,
      'title': 'Ratings perfectly balanced as all things should be'},
     {'is_video': False,
      'num_comments': 135,
      'num_crossposts': 0,
      'subreddit': 'Bossfight',
      'time_elasped': 46852.51457309723,
      'title': 'Gaben, The money vanisher'},
     {'is_video': False,
      'num_comments': 349,
      'num_crossposts': 0,
      'subreddit': 'europe',
      'time_elasped': 18728.514574050903,
      'title': 'Answer to "It is nice to be here in Eastern Europe!"'},
     {'is_video': False,
      'num_comments': 339,
      'num_crossposts': 0,
      'subreddit': 'greentext',
      'time_elasped': 45089.51457595825,
      'title': 'anon lose friends'},
     {'is_video': False,
      'num_comments': 48,
      'num_crossposts': 0,
      'subreddit': 'AnimalTextGifs',
      'time_elasped': 40605.5145778656,
      'title': 'Cows watching yoga. [OC]'},
     {'is_video': False,
      'num_comments': 220,
      'num_crossposts': 0,
      'subreddit': 'calvinandhobbes',
      'time_elasped': 49862.51458001137,
      'title': "Calvin's best argument"},
     {'is_video': False,
      'num_comments': 183,
      'num_crossposts': 0,
      'subreddit': 'Marvel',
      'time_elasped': 37258.51458191872,
      'title': 'guardian of the galaxy 2 family'},
     {'is_video': False,
      'num_comments': 119,
      'num_crossposts': 1,
      'subreddit': 'GetMotivated',
      'time_elasped': 47959.514584064484,
      'title': '[Image] the hard... Is what makes it great.'},
     {'is_video': False,
      'num_comments': 175,
      'num_crossposts': 0,
      'subreddit': 'FortNiteBR',
      'time_elasped': 16466.514585018158,
      'title': 'At this moment RiceGum knew he fukt up. His hand is off the controller but not on the hand cam. This post is late but better late than never.'},
     {'is_video': False,
      'num_comments': 122,
      'num_crossposts': 9,
      'subreddit': 'Wellthatsucks',
      'time_elasped': 51966.51458692551,
      'title': 'Hmm, seems dead, let me make sure.....oh shit!'},
     {'is_video': False,
      'num_comments': 265,
      'num_crossposts': 0,
      'subreddit': 'socialism',
      'time_elasped': 35778.514588832855,
      'title': '"Socialism doesn\'t work"'},
     {'is_video': False,
      'num_comments': 349,
      'num_crossposts': 0,
      'subreddit': 'europe',
      'time_elasped': 18728.514590024948,
      'title': 'Answer to "It is nice to be here in Eastern Europe!"'},
     {'is_video': False,
      'num_comments': 288,
      'num_crossposts': 1,
      'subreddit': 'tifu',
      'time_elasped': 38597.51459312439,
      'title': 'TIFU By Powerbombing A Girl'},
     {'is_video': False,
      'num_comments': 46,
      'num_crossposts': 0,
      'subreddit': 'ShittyLifeProTips',
      'time_elasped': 36873.514594078064,
      'title': 'Keep an eye out'},
     {'is_video': False,
      'num_comments': 35,
      'num_crossposts': 0,
      'subreddit': 'sbubby',
      'time_elasped': 37020.51459598541,
      'title': 'Tony Hawk'},
     {'is_video': True,
      'num_comments': 73,
      'num_crossposts': 0,
      'subreddit': 'lego',
      'time_elasped': 37308.51459789276,
      'title': 'MOC Walking Dinosaur'},
     {'is_video': False,
      'num_comments': 355,
      'num_crossposts': 0,
      'subreddit': 'Jokes',
      'time_elasped': 51505.514598846436,
      'title': 'I applied to be a sperm donor and the nurse asked if I could masturbate in the cup...'},
     {'is_video': False,
      'num_comments': 3370,
      'num_crossposts': 11,
      'subreddit': 'pics',
      'time_elasped': 26813.514600992203,
      'title': 'Very true'},
     {'is_video': False,
      'num_comments': 508,
      'num_crossposts': 2,
      'subreddit': 'australia',
      'time_elasped': 49153.51460289955,
      'title': 'Woolworths: “We’re going plastic bag free!” - Also Woolworths...'},
     {'is_video': False,
      'num_comments': 26,
      'num_crossposts': 0,
      'subreddit': 'thalassophobia',
      'time_elasped': 22519.51460504532,
      'title': 'Imagine yourself walking out to take this photo.'},
     {'is_video': False,
      'num_comments': 243,
      'num_crossposts': 0,
      'subreddit': 'mildlyinteresting',
      'time_elasped': 32169.514605998993,
      'title': 'The plane seat in front of me was removed, giving me an abnormal amount of leg room for an economy class seat'},
     {'is_video': False,
      'num_comments': 24,
      'num_crossposts': 0,
      'subreddit': 'standupshots',
      'time_elasped': 32871.51460814476,
      'title': 'If Jesus was alive today.'},
     {'is_video': False,
      'num_comments': 232,
      'num_crossposts': 0,
      'subreddit': 'RoomPorn',
      'time_elasped': 46253.51460981369,
      'title': 'A Secret Work Studio Suspended Below a Highway Overpass in Valencai, by Fernando Abellanas [1,050 × 787]'},
     {'is_video': False,
      'num_comments': 32,
      'num_crossposts': 0,
      'subreddit': 'Blep',
      'time_elasped': 43025.51461100578,
      'title': 'A gif of my cat sleep-blepping.'},
     {'is_video': False,
      'num_comments': 393,
      'num_crossposts': 3,
      'subreddit': 'formula1',
      'time_elasped': 54848.51461291313,
      'title': "it's the sparks, sorry had to do it"},
     {'is_video': False,
      'num_comments': 358,
      'num_crossposts': 0,
      'subreddit': 'changemyview',
      'time_elasped': 18690.5146150589,
      'title': "CMV: As we celebrate Memorial Day, people don't realize that the United States military hasn't actually fought to defend our freedom since 1945, and most of our military deaths since then were sadly fighting for other reasons."},
     {'is_video': True,
      'num_comments': 73,
      'num_crossposts': 0,
      'subreddit': 'powerwashingporn',
      'time_elasped': 38744.514617204666,
      'title': 'I was given a Craftsman 2900 PSI power for free because it was "dead". It needed a new carburetor, the previous owner let gas sit in it too long. Added a fuel shutoff switch, and... this is my first run with the "dead" power washer.'},
     {'is_video': False,
      'num_comments': 83,
      'num_crossposts': 1,
      'subreddit': 'canada',
      'time_elasped': 27917.514618873596,
      'title': "Canada's House of Commons adopts motion to formally enshrine net neutrality into law"},
     {'is_video': False,
      'num_comments': 31,
      'num_crossposts': 0,
      'subreddit': '2meirl4meirl',
      'time_elasped': 45453.51461982727,
      'title': '2meirl4meirl'},
     {'is_video': True,
      'num_comments': 31,
      'num_crossposts': 1,
      'subreddit': 'CatTaps',
      'time_elasped': 38756.51462292671,
      'title': '“I need love!!” Such a good boy'},
     {'is_video': False,
      'num_comments': 31,
      'num_crossposts': 0,
      'subreddit': '2meirl4meirl',
      'time_elasped': 45453.51462507248,
      'title': '2meirl4meirl'},
     {'is_video': False,
      'num_comments': 29,
      'num_crossposts': 0,
      'subreddit': 'dogpictures',
      'time_elasped': 29105.514626979828,
      'title': "Aaaand she's 19. HAPPY BIRTHDAY!"},
     {'is_video': False,
      'num_comments': 61,
      'num_crossposts': 0,
      'subreddit': 'oddlysatisfying',
      'time_elasped': 27835.514629125595,
      'title': 'My cake...purple reflective donut.'},
     {'is_video': False,
      'num_comments': 158,
      'num_crossposts': 0,
      'subreddit': 'DiWHY',
      'time_elasped': 29776.514630794525,
      'title': 'Bathing season'},
     {'is_video': False,
      'num_comments': 19,
      'num_crossposts': 0,
      'subreddit': 'suicidebywords',
      'time_elasped': 34172.51463198662,
      'title': 'Love is something that finds you'},
     {'is_video': False,
      'num_comments': 96,
      'num_crossposts': 0,
      'subreddit': 'happy',
      'time_elasped': 35449.51463389397,
      'title': 'He told me he thought that he would never find love before he met me. Now our wedding is 18 days away💕'},
     {'is_video': False,
      'num_comments': 159,
      'num_crossposts': 0,
      'subreddit': 'starterpacks',
      'time_elasped': 45806.514636039734,
      'title': "''r/starterpacks'' starterpack"},
     {'is_video': False,
      'num_comments': 94,
      'num_crossposts': 1,
      'subreddit': 'DeepFriedMemes',
      'time_elasped': 40415.5146381855,
      'title': 'D E S P A C I T O'},
     {'is_video': False,
      'num_comments': 173,
      'num_crossposts': 3,
      'subreddit': 'ANormalDayInRussia',
      'time_elasped': 48138.514639139175,
      'title': 'Now bounce нахуй, Сука Блять!'},
     {'is_video': False,
      'num_comments': 230,
      'num_crossposts': 4,
      'subreddit': 'wholesomegifs',
      'time_elasped': 47858.514640808105,
      'title': 'Georgia police officer springing into action, using CPR to save a 2-month-old baby from choking.'},
     {'is_video': False,
      'num_comments': 43,
      'num_crossposts': 0,
      'subreddit': 'shittyreactiongifs',
      'time_elasped': 41915.51464295387,
      'title': "MRW it's my first day back to work after broken dick surgery"},
     {'is_video': False,
      'num_comments': 103,
      'num_crossposts': 0,
      'subreddit': 'WritingPrompts',
      'time_elasped': 42380.51464509964,
      'title': '[WP] The Universe™ has just run out of free trial meaning we are being downgraded from Universe™ pro to Universe™ lite.'},
     {'is_video': False,
      'num_comments': 45,
      'num_crossposts': 0,
      'subreddit': 'IASIP',
      'time_elasped': 45386.514646053314,
      'title': 'I stopped posting on r/funny when I realised that laughs are cheap. From now on...'},
     {'is_video': False,
      'num_comments': 163,
      'num_crossposts': 2,
      'subreddit': 'coolguides',
      'time_elasped': 35193.51464796066,
      'title': 'All the days in a 90 year life on a single page'},
     {'is_video': False,
      'num_comments': 133,
      'num_crossposts': 2,
      'subreddit': 'maybemaybemaybe',
      'time_elasped': 45828.51465010643,
      'title': 'Maybe Maybe Maybe'},
     {'is_video': False,
      'num_comments': 198,
      'num_crossposts': 1,
      'subreddit': 'CatastrophicFailure',
      'time_elasped': 47907.514650821686,
      'title': 'Ellicott City, Maryland Post-Flooding'},
     {'is_video': False,
      'num_comments': 90,
      'num_crossposts': 0,
      'subreddit': 'GifRecipes',
      'time_elasped': 42897.51465296745,
      'title': 'Apple Fritter Bread'},
     {'is_video': False,
      'num_comments': 48,
      'num_crossposts': 0,
      'subreddit': 'harrypotter',
      'time_elasped': 26278.5146548748,
      'title': 'Matthew Lewis got married!'},
     {'is_video': True,
      'num_comments': 48,
      'num_crossposts': 0,
      'subreddit': 'Zoomies',
      'time_elasped': 42819.514656066895,
      'title': 'Big Zoomies'},
     {'is_video': False,
      'num_comments': 937,
      'num_crossposts': 15,
      'subreddit': 'Whatcouldgowrong',
      'time_elasped': 54884.51465797424,
      'title': 'Bullying a defenceless homeless guy wcgw'},
     {'is_video': False,
      'num_comments': 1013,
      'num_crossposts': 3,
      'subreddit': 'gifs',
      'time_elasped': 33655.51466012001,
      'title': 'These two idiots on the highway'},
     {'is_video': False,
      'num_comments': 76,
      'num_crossposts': 0,
      'subreddit': 'FireEmblemHeroes',
      'time_elasped': 28974.514661073685,
      'title': 'Upvotes now give you Orbs (Sub CSS Change)'},
     {'is_video': False,
      'num_comments': 38,
      'num_crossposts': 0,
      'subreddit': 'mechanical_gifs',
      'time_elasped': 24790.514662981033,
      'title': 'Iris box'},
     {'is_video': False,
      'num_comments': 31,
      'num_crossposts': 0,
      'subreddit': 'SequelMemes',
      'time_elasped': 39061.51466488838,
      'title': 'Found on Urban Dictionary'},
     {'is_video': False,
      'num_comments': 2976,
      'num_crossposts': 2,
      'subreddit': 'worldnews',
      'time_elasped': 46674.514666080475,
      'title': 'India says it only follows U.N. sanctions, not unilateral US sanctions on Iran'},
     {'is_video': False,
      'num_comments': 86,
      'num_crossposts': 2,
      'subreddit': 'ArtefactPorn',
      'time_elasped': 45020.51466798782,
      'title': 'A Corinthian helmet was discovered in a 5th century BC grave in the Taman Peninsula, southwest Russia (full story in comment) [1600x1000]'},
     {'is_video': False,
      'num_comments': 49,
      'num_crossposts': 1,
      'subreddit': 'vexillology',
      'time_elasped': 39453.514671087265,
      'title': 'Gadsden flag for Python'},
     {'is_video': False,
      'num_comments': 30,
      'num_crossposts': 2,
      'subreddit': 'NotMyJob',
      'time_elasped': 32552.51467204094,
      'title': 'Not so smart water'},
     {'is_video': True,
      'num_comments': 65,
      'num_crossposts': 0,
      'subreddit': 'Justrolledintotheshop',
      'time_elasped': 40094.51467394829,
      'title': 'Compressor/condenser job on my mom’s CRV when I found a 10mm I lost a LONG time ago. There’s hope out there everyone, your 10mm might be closer than you think.'},
     {'is_video': False,
      'num_comments': 1538,
      'num_crossposts': 0,
      'subreddit': 'WhitePeopleTwitter',
      'time_elasped': 47182.51467585564,
      'title': 'Gotta catch em all'},
     {'is_video': False,
      'num_comments': 146,
      'num_crossposts': 0,
      'subreddit': 'aviation',
      'time_elasped': 32298.514678001404,
      'title': "Two Ospreys escort the U.S. President's helicopter over New York"},
     {'is_video': False,
      'num_comments': 142,
      'num_crossposts': 0,
      'subreddit': 'ScottishPeopleTwitter',
      'time_elasped': 52676.51467895508,
      'title': 'Jaffa Muncher'},
     {'is_video': False,
      'num_comments': 64,
      'num_crossposts': 0,
      'subreddit': 'thedonald',
      'time_elasped': 31119.514681100845,
      'title': 'The one true Donald wishes you a happy Memorial Day! 🇺🇸'},
     {'is_video': False,
      'num_comments': 15,
      'num_crossposts': 0,
      'subreddit': 'corgi',
      'time_elasped': 35155.514682769775,
      'title': 'Suns out, tongues out!'},
     {'is_video': False,
      'num_comments': 15,
      'num_crossposts': 0,
      'subreddit': 'PornhubComments',
      'time_elasped': 42906.51468491554,
      'title': 'Ah those were the days...'},
     {'is_video': False,
      'num_comments': 55,
      'num_crossposts': 0,
      'subreddit': 'freefolk',
      'time_elasped': 49527.51468586922,
      'title': 'Gods'},
     {'is_video': False,
      'num_comments': 1342,
      'num_crossposts': 0,
      'subreddit': 'AskReddit',
      'time_elasped': 29527.514688014984,
      'title': 'What did your parents allow you to do that you would never allow your own children to do?'},
     {'is_video': False,
      'num_comments': 176,
      'num_crossposts': 3,
      'subreddit': 'mildlyinfuriating',
      'time_elasped': 32130.51469016075,
      'title': 'The visible slice of bacon vs. the rest of the package.'},
     {'is_video': False,
      'num_comments': 100,
      'num_crossposts': 2,
      'subreddit': 'nononono',
      'time_elasped': 38627.514691114426,
      'title': 'Feeding the seagulls.'},
     {'is_video': False,
      'num_comments': 197,
      'num_crossposts': 1,
      'subreddit': 'gaming',
      'time_elasped': 32546.514693021774,
      'title': 'Perfect use of MineCraft'},
     {'is_video': False,
      'num_comments': 14,
      'num_crossposts': 0,
      'subreddit': 'BikiniBottomTwitter',
      'time_elasped': 26690.514694929123,
      'title': 'oh no, not again'},
     {'is_video': False,
      'num_comments': 77,
      'num_crossposts': 1,
      'subreddit': 'TheSimpsons',
      'time_elasped': 50519.51469707489,
      'title': 'Why, you and I can run this plant ourselves!'},
     {'is_video': False,
      'num_comments': 176,
      'num_crossposts': 0,
      'subreddit': 'teenagers',
      'time_elasped': 50108.514698028564,
      'title': 'Is This a Terrorism'},
     {'is_video': False,
      'num_comments': 17,
      'num_crossposts': 0,
      'subreddit': 'sadcringe',
      'time_elasped': 25958.514699935913,
      'title': 'Top review for evil apples'},
     {'is_video': False,
      'num_comments': 27,
      'num_crossposts': 0,
      'subreddit': 'yesyesyesno',
      'time_elasped': 41756.514701128006,
      'title': 'The fetch that almost was'},
     {'is_video': False,
      'num_comments': 21,
      'num_crossposts': 2,
      'subreddit': 'wholesomememes',
      'time_elasped': 25813.514703035355,
      'title': '[OC] The joy of painting.'},
     {'is_video': False,
      'num_comments': 47,
      'num_crossposts': 0,
      'subreddit': 'specializedtools',
      'time_elasped': 20954.514704942703,
      'title': 'This tool makes fake brick print'},
     {'is_video': False,
      'num_comments': 8,
      'num_crossposts': 0,
      'subreddit': 'WhatsWrongWithYourDog',
      'time_elasped': 32651.514705896378,
      'title': "This is Cricket. Sometimes she can't handle the excitement of going to the beach."},
     {'is_video': False,
      'num_comments': 358,
      'num_crossposts': 0,
      'subreddit': 'TwoXChromosomes',
      'time_elasped': 33898.514708042145,
      'title': 'I was seen as nothing but a baby maker and it was humiliating'},
     {'is_video': False,
      'num_comments': 87,
      'num_crossposts': 1,
      'subreddit': 'futurama',
      'time_elasped': 54665.51470994949,
      'title': 'Found on r/totallynotrobots'},
     {'is_video': False,
      'num_comments': 65,
      'num_crossposts': 0,
      'subreddit': 'BoJackHorseman',
      'time_elasped': 20177.514711141586,
      'title': 'From BoJacks Twitter: “damn it todd i said you can have a a few people over”'},
     {'is_video': False,
      'num_comments': 104,
      'num_crossposts': 1,
      'subreddit': 'funny',
      'time_elasped': 20764.51471400261,
      'title': 'When they ask you about the dress code.'},
     {'is_video': False,
      'num_comments': 20,
      'num_crossposts': 0,
      'subreddit': 'holdmycatnip',
      'time_elasped': 16773.514715909958,
      'title': 'HMCN while I touch this tail.'},
     {'is_video': False,
      'num_comments': 34,
      'num_crossposts': 0,
      'subreddit': 'totallynotrobots',
      'time_elasped': 20479.514718055725,
      'title': 'I THINK I WILL FINALLY BE GETTING A PET'},
     {'is_video': True,
      'num_comments': 93,
      'num_crossposts': 3,
      'subreddit': 'PenmanshipPorn',
      'time_elasped': 49912.5147190094,
      'title': 'How NOT to break in a new pen (...wait for it)'},
     {'is_video': False,
      'num_comments': 137,
      'num_crossposts': 0,
      'subreddit': 'texas',
      'time_elasped': 31308.514720916748,
      'title': 'Yup'},
     {'is_video': False,
      'num_comments': 43,
      'num_crossposts': 0,
      'subreddit': 'dank_meme',
      'time_elasped': 31261.51472210884,
      'title': 'Error 404 Flair Not Found'},
     {'is_video': False,
      'num_comments': 14,
      'num_crossposts': 0,
      'subreddit': 'socialanxiety',
      'time_elasped': 29565.51472401619,
      'title': 'I have stitched no truer words 🖤'},
     {'is_video': True,
      'num_comments': 60,
      'num_crossposts': 1,
      'subreddit': 'KerbalSpaceProgram',
      'time_elasped': 24980.51472592354,
      'title': 'Oh... okay then little guy...'},
     {'is_video': False,
      'num_comments': 64,
      'num_crossposts': 1,
      'subreddit': 'carporn',
      'time_elasped': 51167.51472783089,
      'title': 'White Horse'},
     {'is_video': False,
      'num_comments': 130,
      'num_crossposts': 0,
      'subreddit': 'PrequelMemes',
      'time_elasped': 26586.51472902298,
      'title': 'A surprise to be sure, but a welcome one'},
     {'is_video': False,
      'num_comments': 39,
      'num_crossposts': 0,
      'subreddit': 'StrangerThings',
      'time_elasped': 46545.51473093033,
      'title': 'This is first date material right here.'},
     {'is_video': False,
      'num_comments': 581,
      'num_crossposts': 10,
      'subreddit': 'hmmm',
      'time_elasped': 59229.514733076096,
      'title': 'hmmm'},
     {'is_video': False,
      'num_comments': 18,
      'num_crossposts': 0,
      'subreddit': 'notinteresting',
      'time_elasped': 33797.51473522186,
      'title': 'Star Wars Easter egg'},
     {'is_video': False,
      'num_comments': 146,
      'num_crossposts': 1,
      'subreddit': 'malefashionadvice',
      'time_elasped': 49042.51473689079,
      'title': '1 000 000 subs!'},
     {'is_video': False,
      'num_comments': 427,
      'num_crossposts': 0,
      'subreddit': 'gifsthatkeepongiving',
      'time_elasped': 59497.51473784447,
      'title': 'The tension build up'},
     {'is_video': False,
      'num_comments': 81,
      'num_crossposts': 2,
      'subreddit': 'battlestations',
      'time_elasped': 44591.514739990234,
      'title': 'His and hers battle grounds. co-op mode ofcourse.'},
     {'is_video': False,
      'num_comments': 77,
      'num_crossposts': 0,
      'subreddit': 'Nicegirls',
      'time_elasped': 30111.514741897583,
      'title': 'Trash /traSH/ noun: 1. Someone who is not lining up at the doorstep of a girl simply because she is interested in you. 2. Discarded matter'},
     {'is_video': False,
      'num_comments': 215,
      'num_crossposts': 1,
      'subreddit': 'PoliticalHumor',
      'time_elasped': 33606.51474404335,
      'title': 'Average MAGA voter'},
     {'is_video': False,
      'num_comments': 76,
      'num_crossposts': 0,
      'subreddit': 'DnD',
      'time_elasped': 41191.514744997025,
      'title': '[Art] [OC] I made a magic weapon called Silken Wile'},
     {'is_video': False,
      'num_comments': 1367,
      'num_crossposts': 5,
      'subreddit': 'todayilearned',
      'time_elasped': 53461.51474690437,
      'title': "TIL That Yao Ming's conservation campaigns has led to a 50% drop in shark fin soup consumption in China. He is now working on poaching as well."},
     {'is_video': False,
      'num_comments': 11,
      'num_crossposts': 0,
      'subreddit': 'confusing_perspective',
      'time_elasped': 20133.51474881172,
      'title': 'When you catch her looking at you'},
     {'is_video': False,
      'num_comments': 42,
      'num_crossposts': 0,
      'subreddit': 'LearnUselessTalents',
      'time_elasped': 37704.514750003815,
      'title': 'Request: How to do the more impressive chicken dance?'},
     {'is_video': False,
      'num_comments': 25,
      'num_crossposts': 0,
      'subreddit': 'unstirredpaint',
      'time_elasped': 32928.51475191116,
      'title': 'beta fish.'},
     {'is_video': False,
      'num_comments': 326,
      'num_crossposts': 0,
      'subreddit': 'DadReflexes',
      'time_elasped': 56199.51475405693,
      'title': 'Dad reflexes kicked in'},
     {'is_video': False,
      'num_comments': 279,
      'num_crossposts': 6,
      'subreddit': 'instant_regret',
      'time_elasped': 48929.5147562027,
      'title': 'Feeding the seagulls'},
     {'is_video': False,
      'num_comments': 90,
      'num_crossposts': 0,
      'subreddit': 'Naruto',
      'time_elasped': 26648.514757871628,
      'title': 'Can we all agree that this is the ugliest character in the series?'},
     {'is_video': False,
      'num_comments': 15,
      'num_crossposts': 0,
      'subreddit': 'ilikthebred',
      'time_elasped': 46067.514760017395,
      'title': 'Watchful floof'},
     {'is_video': False,
      'num_comments': 50,
      'num_crossposts': 2,
      'subreddit': 'asianpeoplegifs',
      'time_elasped': 42172.514761924744,
      'title': 'If that’s not a clear message I don’t know what is'},
     {'is_video': False,
      'num_comments': 13,
      'num_crossposts': 0,
      'subreddit': 'ChildrenFallingOver',
      'time_elasped': 18759.514763116837,
      'title': 'So Close…'},
     {'is_video': False,
      'num_comments': 41,
      'num_crossposts': 0,
      'subreddit': 'polandball',
      'time_elasped': 32076.514765024185,
      'title': 'Lost in the Jungle'},
     {'is_video': False,
      'num_comments': 45,
      'num_crossposts': 0,
      'subreddit': 'subaru',
      'time_elasped': 21037.51476597786,
      'title': 'My brother passed away and I got his car, I had to give him a shout out today!'},
     {'is_video': False,
      'num_comments': 212,
      'num_crossposts': 0,
      'subreddit': 'westworld',
      'time_elasped': 37338.51476788521,
      'title': 'Team Meave looking badass'},
     {'is_video': False,
      'num_comments': 217,
      'num_crossposts': 0,
      'subreddit': 'CatsStandingUp',
      'time_elasped': 54606.51476883888,
      'title': 'Cat.'},
     {'is_video': False,
      'num_comments': 29,
      'num_crossposts': 0,
      'subreddit': 'BokuNoHeroAcademia',
      'time_elasped': 31771.51477098465,
      'title': 'Me and my sisters decided to give our niece volume 1 for Christmas last year. A week ago she was all caught up and attended her first convention.'},
     {'is_video': False,
      'num_comments': 128,
      'num_crossposts': 0,
      'subreddit': 'LivestreamFail',
      'time_elasped': 18053.514772892,
      'title': 'Hyphonix switching to YouTube'},
     {'is_video': False,
      'num_comments': 17,
      'num_crossposts': 0,
      'subreddit': 'PewdiepieSubmissions',
      'time_elasped': 16325.514774084091,
      'title': 'REEEEEEEEEEEEEEEEEEE'},
     {'is_video': False,
      'num_comments': 89,
      'num_crossposts': 2,
      'subreddit': 'restofthefuckingowl',
      'time_elasped': 59764.51477599144,
      'title': 'Ok. This is actually funny.'},
     {'is_video': False,
      'num_comments': 37,
      'num_crossposts': 0,
      'subreddit': 'StoppedWorking',
      'time_elasped': 59187.51477813721,
      'title': 'Scratch caused fatal error.'},
     {'is_video': False,
      'num_comments': 14,
      'num_crossposts': 0,
      'subreddit': 'firstworldanarchists',
      'time_elasped': 27572.514778852463,
      'title': 'Just buy the cheese'},
     {'is_video': False,
      'num_comments': 122,
      'num_crossposts': 4,
      'subreddit': 'AnimalsBeingJerks',
      'time_elasped': 61859.51478099823,
      'title': 'SAY THAT AGAIN'},
     {'is_video': False,
      'num_comments': 60,
      'num_crossposts': 6,
      'subreddit': 'WatchandLearn',
      'time_elasped': 42243.514781951904,
      'title': 'How chains are made'},
     {'is_video': False,
      'num_comments': 55,
      'num_crossposts': 0,
      'subreddit': 'Military',
      'time_elasped': 38970.51478409767,
      'title': 'Where the hell is the Marine Corps?!'},
     {'is_video': False,
      'num_comments': 615,
      'num_crossposts': 3,
      'subreddit': 'assholedesign',
      'time_elasped': 62174.514785051346,
      'title': 'Women’s clothing designers: lol'},
     {'is_video': False,
      'num_comments': 17,
      'num_crossposts': 0,
      'subreddit': 'BeforeNAfterAdoption',
      'time_elasped': 21291.514786958694,
      'title': 'He went from dumpster kitty to king of the castle.'},
     {'is_video': False,
      'num_comments': 23,
      'num_crossposts': 0,
      'subreddit': 'shockwaveporn',
      'time_elasped': 20816.51478910446,
      'title': 'Exploding Ordnance'},
     {'is_video': False,
      'num_comments': 55,
      'num_crossposts': 0,
      'subreddit': 'DDLC',
      'time_elasped': 23483.514789819717,
      'title': 'A short DDLC comic: Existence'},
     {'is_video': False,
      'num_comments': 45,
      'num_crossposts': 3,
      'subreddit': 'whitepeoplegifs',
      'time_elasped': 42351.514791965485,
      'title': 'The Flip Master'},
     {'is_video': False,
      'num_comments': 119,
      'num_crossposts': 1,
      'subreddit': 'Art',
      'time_elasped': 39199.51479291916,
      'title': 'Summer Skin 3, Silk Screen, 22"x30"'},
     {'is_video': False,
      'num_comments': 11,
      'num_crossposts': 0,
      'subreddit': 'dontdeadopeninside',
      'time_elasped': 22431.514795064926,
      'title': 'Make Your Stop Stop the for Lunch and Dinner'},
     {'is_video': False,
      'num_comments': 82,
      'num_crossposts': 1,
      'subreddit': 'Idubbbz',
      'time_elasped': 39583.514796972275,
      'title': 'K-Dubbbz'},
     {'is_video': False,
      'num_comments': 410,
      'num_crossposts': 2,
      'subreddit': 'news',
      'time_elasped': 37982.51479911804,
      'title': 'Virginia man put ‘copyright’ on homemade child pornography, feds say'},
     {'is_video': False,
      'num_comments': 17,
      'num_crossposts': 0,
      'subreddit': 'PewdiepieSubmissions',
      'time_elasped': 16325.514800786972,
      'title': 'REEEEEEEEEEEEEEEEEEE'},
     {'is_video': False,
      'num_comments': 68,
      'num_crossposts': 1,
      'subreddit': 'greentext',
      'time_elasped': 42459.51480293274,
      'title': 'Anon sees his mom'},
     {'is_video': False,
      'num_comments': 55,
      'num_crossposts': 0,
      'subreddit': 'rupaulsdragrace',
      'time_elasped': 23111.514803886414,
      'title': 'Willam asking Ivanka the real question.'},
     {'is_video': False,
      'num_comments': 85,
      'num_crossposts': 0,
      'subreddit': 'teslamotors',
      'time_elasped': 20897.51480603218,
      'title': 'There are some things you can only do in a Tesla!'},
     {'is_video': False,
      'num_comments': 222,
      'num_crossposts': 2,
      'subreddit': 'comics',
      'time_elasped': 48419.51480817795,
      'title': 'Mini comic: family'},
     {'is_video': False,
      'num_comments': 826,
      'num_crossposts': 0,
      'subreddit': 'KidsAreFuckingStupid',
      'time_elasped': 64079.5148100853,
      'title': "I'm sure no one will know who's son he is"},
     {'is_video': True,
      'num_comments': 243,
      'num_crossposts': 3,
      'subreddit': 'FortNiteBR',
      'time_elasped': 36863.51481080055,
      'title': 'Knock before entering! 2000 IQ Door juke.'},
     {'is_video': False,
      'num_comments': 42,
      'num_crossposts': 0,
      'subreddit': 'MealPrepSunday',
      'time_elasped': 34085.51481294632,
      'title': 'Spicy ground turkey &amp; salsa recipe (macros included)'},
     {'is_video': True,
      'num_comments': 43,
      'num_crossposts': 1,
      'subreddit': 'arresteddevelopment',
      'time_elasped': 35422.51481509209,
      'title': 'NYC Time Square promo video!'},
     {'is_video': False,
      'num_comments': 118,
      'num_crossposts': 0,
      'subreddit': 'streetwear',
      'time_elasped': 40026.51481604576,
      'title': '[WDYWT] fleece szn'},
     {'is_video': False,
      'num_comments': 27,
      'num_crossposts': 0,
      'subreddit': 'YouShouldKnow',
      'time_elasped': 28495.51481795311,
      'title': 'YSK Arlo camera owners are being advised to change their account passwords immediately because some user’s credentials have been compromised by an unknown third-party.'},
     {'is_video': False,
      'num_comments': 90,
      'num_crossposts': 1,
      'subreddit': 'RedLetterMedia',
      'time_elasped': 27075.514820098877,
      'title': "Disney uses the hacks' likenesses again"},
     {'is_video': False,
      'num_comments': 35,
      'num_crossposts': 0,
      'subreddit': 'lotr',
      'time_elasped': 24876.51482105255,
      'title': 'Finally reading through properly - I grew up with the films and dipped into the books but never read end-to-end.'},
     {'is_video': False,
      'num_comments': 61,
      'num_crossposts': 0,
      'subreddit': 'blunderyears',
      'time_elasped': 27418.5148229599,
      'title': 'For my 16th birthday I asked for a Looney Tunes cake and a new game for my Gameboy.'},
     {'is_video': False,
      'num_comments': 106,
      'num_crossposts': 0,
      'subreddit': 'AbsoluteUnits',
      'time_elasped': 42628.51482486725,
      'title': 'Health minister of belgium; absolute unit'},
     {'is_video': False,
      'num_comments': 244,
      'num_crossposts': 0,
      'subreddit': 'PS4',
      'time_elasped': 39817.514827013016,
      'title': 'Nioh sells over 2M copies worldwide'},
     {'is_video': False,
      'num_comments': 72,
      'num_crossposts': 0,
      'subreddit': 'splatoon',
      'time_elasped': 42686.51482796669,
      'title': 'My sons Inkling cosplay for Momocon 2018'},
     {'is_video': False,
      'num_comments': 20,
      'num_crossposts': 0,
      'subreddit': 'tippytaps',
      'time_elasped': 38963.51483011246,
      'title': 'Brocco-taps'},
     {'is_video': False,
      'num_comments': 28,
      'num_crossposts': 0,
      'subreddit': 'outrun',
      'time_elasped': 39773.51483106613,
      'title': 'RocketLeague x Monstercat'},
     {'is_video': False,
      'num_comments': 109,
      'num_crossposts': 0,
      'subreddit': 'OldSchoolCool',
      'time_elasped': 28181.51483297348,
      'title': "my mom's middle school custodian 1980"},
     {'is_video': True,
      'num_comments': 112,
      'num_crossposts': 0,
      'subreddit': 'PUBATTLEGROUNDS',
      'time_elasped': 46870.514833927155,
      'title': 'Nobody expects the Flashbang and the Revolver'},
     {'is_video': False,
      'num_comments': 58,
      'num_crossposts': 0,
      'subreddit': 'bonehurtingjuice',
      'time_elasped': 54297.51483607292,
      'title': 'oof ouch owwie my butt magnet wheelchair'},
     {'is_video': False,
      'num_comments': 42,
      'num_crossposts': 0,
      'subreddit': 'CrazyIdeas',
      'time_elasped': 34335.51483798027,
      'title': 'To make shows with a laugh track funnier: Instead of the sound of an audience laughing, replace it with the sound of just one guy laughing hysterically.'},
     {'is_video': False,
      'num_comments': 8,
      'num_crossposts': 1,
      'subreddit': 'absolutelynotmeirl',
      'time_elasped': 35901.51484012604,
      'title': 'Absolutely not me irl'},
     {'is_video': False,
      'num_comments': 66,
      'num_crossposts': 0,
      'subreddit': 'lotrmemes',
      'time_elasped': 29321.514842033386,
      'title': 'Virgin Martin vs Chad Tolkien'},
     {'is_video': False,
      'num_comments': 9,
      'num_crossposts': 0,
      'subreddit': 'sportsarefun',
      'time_elasped': 25695.51484298706,
      'title': 'Fan gets a high five from Outfielder'},
     {'is_video': False,
      'num_comments': 84,
      'num_crossposts': 0,
      'subreddit': 'reddevils',
      'time_elasped': 30767.51484489441,
      'title': "(5 min read) Why Mourinho's staffing changes could be as important as summer transfers"},
     {'is_video': False,
      'num_comments': 77,
      'num_crossposts': 0,
      'subreddit': 'WeWantPlates',
      'time_elasped': 39042.51484704018,
      'title': 'BBQ on a Shovel'},
     {'is_video': False,
      'num_comments': 209,
      'num_crossposts': 0,
      'subreddit': 'LiverpoolFC',
      'time_elasped': 15072.514848947525,
      'title': 'Welcome Fabinho!'},
     {'is_video': False,
      'num_comments': 24,
      'num_crossposts': 0,
      'subreddit': 'OTMemes',
      'time_elasped': 21625.514850139618,
      'title': 'You vs the guy she tells you not to worry about'},
     {'is_video': False,
      'num_comments': 50,
      'num_crossposts': 1,
      'subreddit': 'forbiddensnacks',
      'time_elasped': 43043.51485204697,
      'title': 'Forbidden_popscicle'},
     {'is_video': False,
      'num_comments': 3190,
      'num_crossposts': 1,
      'subreddit': 'MemeEconomy',
      'time_elasped': 55791.51485323906,
      'title': 'Random encounter. If anyone has the template please help me and the rest in buying it. Thank you.'},
     {'is_video': False,
      'num_comments': 88,
      'num_crossposts': 1,
      'subreddit': 'MapPorn',
      'time_elasped': 37830.51485490799,
      'title': 'Tree Canopy Height of Contiguous USA in Meters'},
     {'is_video': False,
      'num_comments': 29,
      'num_crossposts': 0,
      'subreddit': 'DeepFriedMemes',
      'time_elasped': 28618.51485800743,
      'title': "Y'all heard des🅱acito 2 yet?"},
     {'is_video': False,
      'num_comments': 348,
      'num_crossposts': 1,
      'subreddit': 'forwardsfromgrandma',
      'time_elasped': 38774.514858961105,
      'title': 'Waaaa'},
     {'is_video': False,
      'num_comments': 93,
      'num_crossposts': 0,
      'subreddit': 'UpliftingNews',
      'time_elasped': 47684.51486110687,
      'title': 'Alabama student who woke up at 4.30am every morning to get a bus to his high school is gifted a car after photos of him making the trek in graduation robes went viral'},
     {'is_video': False,
      'num_comments': 172,
      'num_crossposts': 0,
      'subreddit': 'newjersey',
      'time_elasped': 37749.51486301422,
      'title': 'Gotta catch em all'},
     {'is_video': False,
      'num_comments': 289,
      'num_crossposts': 7,
      'subreddit': 'BlackPeopleTwitter',
      'time_elasped': 56647.51486492157,
      'title': 'That explains everything'},
     {'is_video': False,
      'num_comments': 287,
      'num_crossposts': 5,
      'subreddit': 'CasualUK',
      'time_elasped': 64203.514865875244,
      'title': 'Better Watch Out'},
     {'is_video': False,
      'num_comments': 30,
      'num_crossposts': 0,
      'subreddit': 'gardening',
      'time_elasped': 41461.51486802101,
      'title': 'Medinilla Magnifica: a native to the Philippines'},
     {'is_video': False,
      'num_comments': 34,
      'num_crossposts': 0,
      'subreddit': 'indianpeoplefacebook',
      'time_elasped': 38844.51486992836,
      'title': 'Sweet like sugar'},
     {'is_video': False,
      'num_comments': 30,
      'num_crossposts': 0,
      'subreddit': 'memes',
      'time_elasped': 39421.51487112045,
      'title': 'The dog all puppies aspire to be'},
     {'is_video': False,
      'num_comments': 67,
      'num_crossposts': 0,
      'subreddit': 'The_Mueller',
      'time_elasped': 44733.5148730278,
      'title': "Republican Rick Wilson: If the GOP found 'spygate' intel 'Devin Nunes would have run out like a monkey with his ass on fire'"},
     {'is_video': False,
      'num_comments': 140,
      'num_crossposts': 0,
      'subreddit': 'fakehistoryporn',
      'time_elasped': 65578.51487493515,
      'title': 'Mods asleep. Upvote literal fake history porn. (1197, Braille)'},
     {'is_video': False,
      'num_comments': 70,
      'num_crossposts': 1,
      'subreddit': 'oddlysatisfying',
      'time_elasped': 32277.514875888824,
      'title': 'Precision Manufacturing'},
     {'is_video': False,
      'num_comments': 29,
      'num_crossposts': 0,
      'subreddit': 'Superbowl',
      'time_elasped': 41649.51487803459,
      'title': 'A Barred Owl (I think?) Has Decided to Make My Yard His Home, Meet Darius'},
     {'is_video': True,
      'num_comments': 15,
      'num_crossposts': 0,
      'subreddit': 'Simulated',
      'time_elasped': 16816.51487994194,
      'title': '[OC] Blocky tornado.'},
     {'is_video': False,
      'num_comments': 66,
      'num_crossposts': 2,
      'subreddit': 'pics',
      'time_elasped': 24317.514882087708,
      'title': '"This one opened too!"'},
     {'is_video': False,
      'num_comments': 27,
      'num_crossposts': 0,
      'subreddit': 'Sneks',
      'time_elasped': 47986.514883995056,
      'title': 'Im turtlesss now 🐢'},
     {'is_video': False,
      'num_comments': 55,
      'num_crossposts': 0,
      'subreddit': 'Unexpected',
      'time_elasped': 22106.514885902405,
      'title': 'Fuel your workout'},
     {'is_video': False,
      'num_comments': 99,
      'num_crossposts': 0,
      'subreddit': 'TooMeIrlForMeIrl',
      'time_elasped': 64961.51488804817,
      'title': 'Toomeirlformeirl'},
     {'is_video': False,
      'num_comments': 73,
      'num_crossposts': 0,
      'subreddit': 'GamePhysics',
      'time_elasped': 55763.514889001846,
      'title': '[Witcher 3] Well that’s a bit unfair.'},
     {'is_video': False,
      'num_comments': 238,
      'num_crossposts': 0,
      'subreddit': 'Libertarian',
      'time_elasped': 35438.514890909195,
      'title': 'A free market allowing competitors to undercut overinflated prices? Libertarian markets allow this and more.'},
     {'is_video': False,
      'num_comments': 265,
      'num_crossposts': 0,
      'subreddit': 'FellowKids',
      'time_elasped': 50798.51489305496,
      'title': "Y'all like Fortnite??"},
     {'is_video': False,
      'num_comments': 36,
      'num_crossposts': 1,
      'subreddit': 'mildlyinteresting',
      'time_elasped': 21951.514894008636,
      'title': 'My girlfriend’s cat paces using the same steps each day'},
     {'is_video': False,
      'num_comments': 128,
      'num_crossposts': 0,
      'subreddit': 'Jokes',
      'time_elasped': 39372.514895915985,
      'title': 'Yo mama so fat'},
     {'is_video': False,
      'num_comments': 293,
      'num_crossposts': 1,
      'subreddit': 'antiMLM',
      'time_elasped': 48588.514897823334,
      'title': 'That’s...that’s not a thing that happens.'},
     {'is_video': False,
      'num_comments': 14,
      'num_crossposts': 0,
      'subreddit': 'savedyouaclick',
      'time_elasped': 17781.514899015427,
      'title': "Woman Rescues Baby Kitten – Soon Realizes It’s Not What It Seems | It's a fox"},
     {'is_video': False,
      'num_comments': 50,
      'num_crossposts': 0,
      'subreddit': 'natureismetal',
      'time_elasped': 42378.514900922775,
      'title': 'Leopard with a baboon head'},
     {'is_video': False,
      'num_comments': 75,
      'num_crossposts': 0,
      'subreddit': 'dadjokes',
      'time_elasped': 43389.51490306854,
      'title': 'What was Oman called before it officially become a nation?'},
     {'is_video': False,
      'num_comments': 138,
      'num_crossposts': 1,
      'subreddit': 'likeus',
      'time_elasped': 62471.51490402222,
      'title': 'Bear family enjoying a summer day on the beach'},
     {'is_video': False,
      'num_comments': 16,
      'num_crossposts': 0,
      'subreddit': 'The_Dennis',
      'time_elasped': 28409.514906167984,
      'title': 'The GOLDEN GOD is AWAKE and on patrol for NON-DENNIS posts'},
     {'is_video': False,
      'num_comments': 22,
      'num_crossposts': 0,
      'subreddit': 'FlashTV',
      'time_elasped': 32223.51490688324,
      'title': 'When the teacher asks you what are your goals for the school year'},
     {'is_video': False,
      'num_comments': 20,
      'num_crossposts': 0,
      'subreddit': 'Eyebleach',
      'time_elasped': 36090.51490902901,
      'title': 'Oh hi it’s so good to see you'},
     {'is_video': False,
      'num_comments': 5,
      'num_crossposts': 0,
      'subreddit': 'curledfeetsies',
      'time_elasped': 42011.514910936356,
      'title': 'Baby feetsies'},
     {'is_video': False,
      'num_comments': 62,
      'num_crossposts': 0,
      'subreddit': 'brockhampton',
      'time_elasped': 38667.51491189003,
      'title': 'Take me back to November'},
     {'is_video': False,
      'num_comments': 27,
      'num_crossposts': 0,
      'subreddit': 'doctorwho',
      'time_elasped': 34367.5149140358,
      'title': 'Tenth Doctor cosplay by my boyfriend. He truly loves this cosplay. Hoping I can convince him to take the plunge on getting a suit by Magnoli Clothiers!'},
     {'is_video': False,
      'num_comments': 8,
      'num_crossposts': 0,
      'subreddit': 'teefies',
      'time_elasped': 24228.514916181564,
      'title': 'When the nip hits'},
     {'is_video': False,
      'num_comments': 26,
      'num_crossposts': 0,
      'subreddit': '2meirl4meirl',
      'time_elasped': 36465.514917850494,
      'title': '2meirl4meirl'},
     {'is_video': False,
      'num_comments': 31,
      'num_crossposts': 0,
      'subreddit': 'Baking',
      'time_elasped': 32244.51491880417,
      'title': 'I attempted marshmallow teacakes!'},
     {'is_video': False,
      'num_comments': 69,
      'num_crossposts': 0,
      'subreddit': 'NoStupidQuestions',
      'time_elasped': 29030.514920949936,
      'title': 'After a brain surgery where the skull cap is removed, how is sensation restored to the scalp?'},
     {'is_video': False,
      'num_comments': 4,
      'num_crossposts': 0,
      'subreddit': 'combinedgifs',
      'time_elasped': 19900.51492190361,
      'title': 'Fashionista'},
     {'is_video': False,
      'num_comments': 51,
      'num_crossposts': 0,
      'subreddit': 'Kanye',
      'time_elasped': 37680.51492404938,
      'title': 'Runaway painting in Kanye’s home. 🌊 🌊'},
     {'is_video': False,
      'num_comments': 48,
      'num_crossposts': 0,
      'subreddit': 'DunderMifflin',
      'time_elasped': 39705.514926195145,
      'title': 'To those who have sacrificed their lives and all who served. Thank you.'},
     {'is_video': False,
      'num_comments': 92,
      'num_crossposts': 0,
      'subreddit': 'SweatyPalms',
      'time_elasped': 48339.51492810249,
      'title': 'How to save a kid from falling off from a balcony'},
     {'is_video': False,
      'num_comments': 5,
      'num_crossposts': 0,
      'subreddit': 'donaldglover',
      'time_elasped': 30619.514930009842,
      'title': 'The one true Donald wishes you a happy Memorial Day! 🇺🇸'},
     {'is_video': False,
      'num_comments': 261,
      'num_crossposts': 0,
      'subreddit': 'zelda',
      'time_elasped': 60771.51493191719,
      'title': 'Thought some of you might enjoy my setup.'},
     {'is_video': False,
      'num_comments': 6277,
      'num_crossposts': 2,
      'subreddit': 'AskReddit',
      'time_elasped': 57520.51493406296,
      'title': 'People of Reddit who have heard someone say their “dying words,” what were they and how did they impact you?'},
     {'is_video': False,
      'num_comments': 11,
      'num_crossposts': 0,
      'subreddit': 'trebuchetmemes',
      'time_elasped': 17324.514935970306,
      'title': 'Homemade cardboard trebuchet.'},
     {'is_video': False,
      'num_comments': 194,
      'num_crossposts': 0,
      'subreddit': 'CryptoCurrency',
      'time_elasped': 24126.514938116074,
      'title': "'The bull run will start after new year!' 'The bull run will start after Chinese new year!' 'The bull run will start after everyone pays their taxes!' 'The bull run will start after everyone gets their tax refunds!' 'The bull run will start after consensus!'"},
     {'is_video': False,
      'num_comments': 13,
      'num_crossposts': 0,
      'subreddit': 'BeAmazed',
      'time_elasped': 19338.51493883133,
      'title': 'Illuminated tunnel in Mei, Japan'},
     {'is_video': False,
      'num_comments': 41,
      'num_crossposts': 0,
      'subreddit': 'Floof',
      'time_elasped': 35019.5149409771,
      'title': 'Oh AnnPurrkins, you majestic ebony colored perfect princess of floof.'},
     {'is_video': False,
      'num_comments': 60,
      'num_crossposts': 0,
      'subreddit': 'GifRecipes',
      'time_elasped': 30229.514942884445,
      'title': 'They finally showed us how to make a Good Burger and Ed’s secret sauce from the movie'},
     {'is_video': False,
      'num_comments': 175,
      'num_crossposts': 0,
      'subreddit': 'MaliciousCompliance',
      'time_elasped': 52018.51494503021,
      'title': "You can't work holidays if you're at 40 hours for the week!"},
     {'is_video': True,
      'num_comments': 1235,
      'num_crossposts': 2,
      'subreddit': 'gifs',
      'time_elasped': 34422.51494598389,
      'title': "Dad's home."},
     {'is_video': False,
      'num_comments': 44,
      'num_crossposts': 0,
      'subreddit': 'smashbros',
      'time_elasped': 35889.514948129654,
      'title': 'I’m really excited for the Smash Switch news in a couple weeks so I drew some goofy characters. Feels like Christmas is coming up!'},
     {'is_video': False,
      'num_comments': 58,
      'num_crossposts': 1,
      'subreddit': 'facepalm',
      'time_elasped': 49882.514949798584,
      'title': 'Braille writing in my school is flat'},
     {'is_video': False,
      'num_comments': 30,
      'num_crossposts': 0,
      'subreddit': 'YouSeeComrade',
      'time_elasped': 47155.51495099068,
      'title': 'You see comrade, we anger and capitalism are one in the same'},
     {'is_video': False,
      'num_comments': 11,
      'num_crossposts': 0,
      'subreddit': 'shittyrainbow6',
      'time_elasped': 41434.514952898026,
      'title': 'Rare photo of Glaz’s parents'},
     {'is_video': False,
      'num_comments': 106,
      'num_crossposts': 0,
      'subreddit': 'theydidthemath',
      'time_elasped': 44348.51495504379,
      'title': '[Request] How fast would you say that this shirt is going?'},
     {'is_video': False,
      'num_comments': 506,
      'num_crossposts': 11,
      'subreddit': 'funny',
      'time_elasped': 36928.51495599747,
      'title': 'I may never need to post again'},
     {'is_video': False,
      'num_comments': 13,
      'num_crossposts': 0,
      'subreddit': 'ExpectationVsReality',
      'time_elasped': 30701.514958143234,
      'title': 'Exercise with a swing'},
     {'is_video': False,
      'num_comments': 33,
      'num_crossposts': 0,
      'subreddit': 'sweden',
      'time_elasped': 42427.514959812164,
      'title': 'Jag_ivl'},
     {'is_video': False,
      'num_comments': 319,
      'num_crossposts': 0,
      'subreddit': 'Music',
      'time_elasped': 63125.51496195793,
      'title': 'Violent Femmes - Blister in the Sun [Alt Rock]'},
     {'is_video': False,
      'num_comments': 52,
      'num_crossposts': 0,
      'subreddit': 'sewing',
      'time_elasped': 41916.514962911606,
      'title': "I saw this fabric on sale and fell in love. I am not very experienced but I'm pretty pleased with how it turned out!"},
     {'is_video': False,
      'num_comments': 21,
      'num_crossposts': 0,
      'subreddit': 'Cyberpunk',
      'time_elasped': 31285.514965057373,
      'title': 'Pills by Klaus Wittmann'},
     {'is_video': False,
      'num_comments': 21,
      'num_crossposts': 0,
      'subreddit': 'vaxxhappened',
      'time_elasped': 37467.51496696472,
      'title': 'Polio meme'},
     {'is_video': False,
      'num_comments': 2,
      'num_crossposts': 0,
      'subreddit': 'bertstrips',
      'time_elasped': 24743.514968156815,
      'title': 'Oh no not again.'},
     {'is_video': False,
      'num_comments': 15,
      'num_crossposts': 0,
      'subreddit': 'MEOW_IRL',
      'time_elasped': 36287.51497006416,
      'title': 'Meow🐱irl'},
     {'is_video': False,
      'num_comments': 43,
      'num_crossposts': 1,
      'subreddit': 'dashcamgifs',
      'time_elasped': 26220.514971971512,
      'title': 'Slow the fuck down when approaching a crosswalk'},
     {'is_video': False,
      'num_comments': 1174,
      'num_crossposts': 1,
      'subreddit': 'Android',
      'time_elasped': 49332.51497387886,
      'title': 'Supposed Pixel 3/3 XL screen protector'},
     {'is_video': False,
      'num_comments': 320,
      'num_crossposts': 0,
      'subreddit': 'vegan',
      'time_elasped': 44016.51497602463,
      'title': 'Some vegans who like to work out together'},
     {'is_video': False,
      'num_comments': 176,
      'num_crossposts': 0,
      'subreddit': 'StarWars',
      'time_elasped': 36130.514977931976,
      'title': 'I was so excited for The Last Jedi to come out on dvd until they inexplicably CHANGED THE POSTER. It looked so much better with Luke in the background!'},
     {'is_video': False,
      'num_comments': 18,
      'num_crossposts': 0,
      'subreddit': 'FloridaMan',
      'time_elasped': 26580.51497912407,
      'title': 'Florida man steals an unmarked police car and crashes into everything in sight'},
     {'is_video': False,
      'num_comments': 30,
      'num_crossposts': 0,
      'subreddit': 'chelseafc',
      'time_elasped': 40353.51498103142,
      'title': 'The Best Bicycle'},
     {'is_video': False,
      'num_comments': 569,
      'num_crossposts': 2,
      'subreddit': 'IdiotsInCars',
      'time_elasped': 70617.51498293877,
      'title': "I'm speechless."},
     {'is_video': True,
      'num_comments': 127,
      'num_crossposts': 0,
      'subreddit': 'TheDepthsBelow',
      'time_elasped': 51371.514984846115,
      'title': 'Two of four whale sharks swimming through in a six million gallon tank in Georgia Aquarium'},
     {'is_video': False,
      'num_comments': 37,
      'num_crossposts': 0,
      'subreddit': 'instantbarbarians',
      'time_elasped': 18100.514986991882,
      'title': 'And the crowd goes wild!'},
     {'is_video': False,
      'num_comments': 612,
      'num_crossposts': 0,
      'subreddit': 'explainlikeimfive',
      'time_elasped': 67349.51498794556,
      'title': 'ELI5:How does an ant not die when flicked full force by a human finger?'},
     {'is_video': False,
      'num_comments': 120,
      'num_crossposts': 0,
      'subreddit': 'interestingasfuck',
      'time_elasped': 47387.514990091324,
      'title': 'Huge tetrapods near High Island Reservoir Hong Kong. The structures are used to reinforce shoreline defenses and prevent coastal erosion by breaking up incoming waves, interlocking, and allowing the water to flow around them rather than against.'},
     {'is_video': False,
      'num_comments': 74,
      'num_crossposts': 0,
      'subreddit': 'trashy',
      'time_elasped': 39277.51499223709,
      'title': 'Don’t worry, he’s not joking.'},
     {'is_video': False,
      'num_comments': 16,
      'num_crossposts': 0,
      'subreddit': 'Damnthatsinteresting',
      'time_elasped': 25787.514992952347,
      'title': "The Statue of Liberty's Shadow"},
     {'is_video': False,
      'num_comments': 57,
      'num_crossposts': 0,
      'subreddit': 'instantkarma',
      'time_elasped': 42916.514994859695,
      'title': 'Good Karma for a change.'},
     {'is_video': False,
      'num_comments': 105,
      'num_crossposts': 0,
      'subreddit': 'Gamingcirclejerk',
      'time_elasped': 35558.51499700546,
      'title': "Here, I'll say it: I just fucking hate women"},
     {'is_video': True,
      'num_comments': 18,
      'num_crossposts': 0,
      'subreddit': 'barkour',
      'time_elasped': 19004.514997959137,
      'title': 'Shitty barkour'},
     {'is_video': False,
      'num_comments': 9,
      'num_crossposts': 0,
      'subreddit': 'Catloaf',
      'time_elasped': 36670.515000104904,
      'title': 'Jack the bedloaf'},
     {'is_video': False,
      'num_comments': 186,
      'num_crossposts': 1,
      'subreddit': 'SubredditDrama',
      'time_elasped': 23019.515002012253,
      'title': 'Migrant is to be granted French citizenship for rescuing a small child. r/news handles this very well.'},
     {'is_video': False,
      'num_comments': 59,
      'num_crossposts': 0,
      'subreddit': 'DungeonsAndDragons',
      'time_elasped': 45576.51500296593,
      'title': 'CHECK FOR TRAP!!'},
     {'is_video': False,
      'num_comments': 354,
      'num_crossposts': 1,
      'subreddit': 'Frugal',
      'time_elasped': 67111.51500487328,
      'title': "This is the end result of taking three gap years between high school and university, working nights stacking supermarket shelves (thank you high Australian minimum wage), and putting everything possible into savings. I was definitely addicted to saving, it probably wasn't healthy in hindsight..."},
     {'is_video': False,
      'num_comments': 29,
      'num_crossposts': 1,
      'subreddit': 'engrish',
      'time_elasped': 46233.51500701904,
      'title': 'My friend in Korea captured this beauty last night.'},
     {'is_video': False,
      'num_comments': 13,
      'num_crossposts': 0,
      'subreddit': 'PuppySmiles',
      'time_elasped': 27750.51500892639,
      'title': "She looks at me like this every time I grab her leash, she knows we're going for a hike. :)"},
     {'is_video': False,
      'num_comments': 11,
      'num_crossposts': 0,
      'subreddit': 'BobsBurgers',
      'time_elasped': 28978.51501107216,
      'title': 'These seem like some Gayle would make.'},
     {'is_video': False,
      'num_comments': 48,
      'num_crossposts': 0,
      'subreddit': 'RussiaLago',
      'time_elasped': 33698.51501202583,
      'title': "Trump's lawyer admits they coordinate lies, fabriate narratives, and manipulate the American public."},
     {'is_video': False,
      'num_comments': 29,
      'num_crossposts': 1,
      'subreddit': 'engrish',
      'time_elasped': 46233.51501393318,
      'title': 'My friend in Korea captured this beauty last night.'},
     {'is_video': False,
      'num_comments': 11,
      'num_crossposts': 0,
      'subreddit': 'BobsBurgers',
      'time_elasped': 28978.51501584053,
      'title': 'These seem like some Gayle would make.'},
     {'is_video': False,
      'num_comments': 13,
      'num_crossposts': 0,
      'subreddit': 'Moviesinthemaking',
      'time_elasped': 20377.515017986298,
      'title': "Terminator 2 (1991) had some practical effects with the help of Linda Hamilton's twin sister. When Sarah cuts a hole in T-800 head it's a model of Schwarzenegger’s head in the foreground, the real Schwarzenegger plays his own reflection, and Linda’s twin sister mimics her moves"},
     {'is_video': True,
      'num_comments': 7,
      'num_crossposts': 0,
      'subreddit': 'shittyrobots',
      'time_elasped': 26181.515020132065,
      'title': 'Minibot has a shitty character and ignores you'},
     {'is_video': False,
      'num_comments': 6,
      'num_crossposts': 0,
      'subreddit': 'spaceporn',
      'time_elasped': 35349.51502108574,
      'title': 'Stargazing at Glacier Point - Milky Way over Half Dome Yosemite August 2017 [OC] [14730 × 4892]'},
     {'is_video': False,
      'num_comments': 37,
      'num_crossposts': 0,
      'subreddit': 'AwesomeCarMods',
      'time_elasped': 40205.51502299309,
      'title': 'The legendary Lancia Stratos'},
     {'is_video': False,
      'num_comments': 29,
      'num_crossposts': 0,
      'subreddit': 'RATS',
      'time_elasped': 34939.51502490044,
      'title': "This little boyo was dumped outside my other half's work (she works in a pet store) and he's the sweetest little lad."},
     {'is_video': False,
      'num_comments': 22,
      'num_crossposts': 0,
      'subreddit': 'SovietWomble',
      'time_elasped': 34487.5150270462,
      'title': 'Found this in a public toilet in a small village called Bellingham'},
     {'is_video': False,
      'num_comments': 22,
      'num_crossposts': 0,
      'subreddit': 'furry_irl',
      'time_elasped': 26778.515028953552,
      'title': 'Furry🐧Irl'},
     {'is_video': False,
      'num_comments': 94,
      'num_crossposts': 0,
      'subreddit': 'shittyfoodporn',
      'time_elasped': 32254.515029907227,
      'title': "My mom's old church cookbook has been grossing me out my whole life"},
     {'is_video': False,
      'num_comments': 44,
      'num_crossposts': 0,
      'subreddit': 'NotKenM',
      'time_elasped': 40417.515032052994,
      'title': 'Not Ken M on Donating Blood'},
     {'is_video': False,
      'num_comments': 17,
      'num_crossposts': 0,
      'subreddit': 'Wellworn',
      'time_elasped': 21603.515033006668,
      'title': 'My 2 year old antenna topper vs new'},
     {'is_video': False,
      'num_comments': 39,
      'num_crossposts': 0,
      'subreddit': 'ArcherFX',
      'time_elasped': 40638.51503491402,
      'title': 'The newspaper in Shapiro’s hand reads “Seize Restaurant under investigation” – the restaurant from S4 E7. An easter egg that’s barely visible even on 1080p.'},
     {'is_video': False,
      'num_comments': 91,
      'num_crossposts': 2,
      'subreddit': 'cursedimages',
      'time_elasped': 60445.515036821365,
      'title': 'Cursed_Life_Hack'},
     {'is_video': False,
      'num_comments': 162,
      'num_crossposts': 1,
      'subreddit': 'ComedyCemetery',
      'time_elasped': 71558.51503801346,
      'title': 'Mods are asleep, upvote actual comedy cemeteries'},
     {'is_video': False,
      'num_comments': 72,
      'num_crossposts': 0,
      'subreddit': 'CasualConversation',
      'time_elasped': 36841.51503992081,
      'title': "I'd like a game show where teams debate a controversial topic, but aren't allowed to openly pick a side and only gain points after arguing for and against. Where the objective is to properly understand both sides."},
     {'is_video': False,
      'num_comments': 925,
      'num_crossposts': 1,
      'subreddit': 'videos',
      'time_elasped': 51787.515042066574,
      'title': 'Guy Gets A Speeding Ticket Then Speeds Off And Gets Another One'},
     {'is_video': False,
      'num_comments': 35,
      'num_crossposts': 0,
      'subreddit': 'humblebrag',
      'time_elasped': 34051.51504421234,
      'title': 'I wish I could be unhealthy but I’m just too fit!'},
     {'is_video': False,
      'num_comments': 29,
      'num_crossposts': 0,
      'subreddit': 'COMPLETEANARCHY',
      'time_elasped': 35660.515045166016,
      'title': 'Conseevatives irl'},
     {'is_video': False,
      'num_comments': 20,
      'num_crossposts': 0,
      'subreddit': 'slowcooking',
      'time_elasped': 12190.515046834946,
      'title': 'Taiwanese Beef Noodle Soup'},
     {'is_video': False,
      'num_comments': 34,
      'num_crossposts': 0,
      'subreddit': 'army',
      'time_elasped': 39609.51504898071,
      'title': 'SFC Matt Leggett, the best damn NCO I ever had the pleasure of serving with. Taught me what it meant to be a leader. Never backed down when it came to taking care of soldiers. I heard him tell a 1SG to go fuck himself downrange to get we got some downtime after pointless ops. KIA AFG Aug 20, 2016.'},
     {'is_video': False,
      'num_comments': 30,
      'num_crossposts': 1,
      'subreddit': 'environment',
      'time_elasped': 23564.51505112648,
      'title': 'Europe plans ban on plastic cutlery, straws and more'},
     {'is_video': False,
      'num_comments': 14,
      'num_crossposts': 2,
      'subreddit': 'PrequelMemes',
      'time_elasped': 25780.515052080154,
      'title': 'Ah, Victory!'},
     {'is_video': True,
      'num_comments': 19,
      'num_crossposts': 0,
      'subreddit': 'marvelstudios',
      'time_elasped': 17775.515053987503,
      'title': 'One of these speeds is not like the others...'},
     {'is_video': False,
      'num_comments': 35,
      'num_crossposts': 2,
      'subreddit': 'thisismylifenow',
      'time_elasped': 64736.515055179596,
      'title': "I'm a fruit stand now.."},
     {'is_video': False,
      'num_comments': 53,
      'num_crossposts': 0,
      'subreddit': 'lastimages',
      'time_elasped': 39517.515056848526,
      'title': 'A selfie my 19 yo brother sent to my mom while she was on vacation to say he missed her. She came home the next day and he was found dead the following morning in the same spot this picture was taken.'},
     {'is_video': False,
      'num_comments': 166,
      'num_crossposts': 0,
      'subreddit': 'IncelTears',
      'time_elasped': 26749.515058994293,
      'title': 'Smart incel understands biology very well'},
     {'is_video': False,
      'num_comments': 24,
      'num_crossposts': 0,
      'subreddit': 'paradoxplaza',
      'time_elasped': 30624.515060901642,
      'title': 'Suggestion - When paradoxplaza reaches 80,000 map-staring experts, the Imperator Platypus should become our 80k avatar'},
     {'is_video': False,
      'num_comments': 40,
      'num_crossposts': 1,
      'subreddit': 'HeavySeas',
      'time_elasped': 46874.515062093735,
      'title': 'Ride of a lifetime, Cloudbreak Fiji, 2018 - instagram.com/_taylorcurran'},
     {'is_video': False,
      'num_comments': 187,
      'num_crossposts': 2,
      'subreddit': 'wholesomememes',
      'time_elasped': 53212.51506400108,
      'title': 'Yer a flower Harry'},
     {'is_video': False,
      'num_comments': 10,
      'num_crossposts': 0,
      'subreddit': 'AlisonBrie',
      'time_elasped': 24929.515065193176,
      'title': 'Late Night Cleavage'},
     {'is_video': False,
      'num_comments': 592,
      'num_crossposts': 3,
      'subreddit': 'aww',
      'time_elasped': 42476.515066862106,
      'title': 'This Little girl turns 13 today'},
     {'is_video': False,
      'num_comments': 283,
      'num_crossposts': 2,
      'subreddit': 'iamverybadass',
      'time_elasped': 70964.51506900787,
      'title': ':)'},
     {'is_video': False,
      'num_comments': 21,
      'num_crossposts': 1,
      'subreddit': 'GoodFakeTexts',
      'time_elasped': 37670.51507091522,
      'title': 'Oops'},
     {'is_video': True,
      'num_comments': 23,
      'num_crossposts': 0,
      'subreddit': 'hitmanimals',
      'time_elasped': 27779.51507306099,
      'title': 'DIE EVIL ROBOT! Repost from r/gifs'},
     {'is_video': False,
      'num_comments': 19,
      'num_crossposts': 0,
      'subreddit': 'woooosh',
      'time_elasped': 32900.51507496834,
      'title': 'Found this on the site while doing some uhh, research. yes.'},
     {'is_video': False,
      'num_comments': 168,
      'num_crossposts': 0,
      'subreddit': 'DDLC',
      'time_elasped': 46851.51507616043,
      'title': 'A couple of sleepy girls'},
     {'is_video': False,
      'num_comments': 61,
      'num_crossposts': 0,
      'subreddit': 'DiWHY',
      'time_elasped': 24866.51507782936,
      'title': 'Hate that ugly colored couch?'},
     {'is_video': False,
      'num_comments': 237,
      'num_crossposts': 1,
      'subreddit': 'worldnews',
      'time_elasped': 50745.51507997513,
      'title': 'Eighty years after they were hunted to extinction, the successful reintroduction of a herd of wild European bison on to the dunes of the Dutch coast is paving the way for their return across the continent.'},
     {'is_video': False,
      'num_comments': 66,
      'num_crossposts': 13,
      'subreddit': 'perfectloops',
      'time_elasped': 68875.5150809288,
      'title': '[A] Everything fits perfectly'},
     {'is_video': False,
      'num_comments': 201,
      'num_crossposts': 0,
      'subreddit': 'keto',
      'time_elasped': 43649.51508307457,
      'title': "Keto isn't a diet, it's a lifehack."},
     {'is_video': False,
      'num_comments': 7,
      'num_crossposts': 0,
      'subreddit': 'SlyGifs',
      'time_elasped': 15272.515084981918,
      'title': "If you can't beat 'em..."},
     {'is_video': False,
      'num_comments': 45,
      'num_crossposts': 0,
      'subreddit': 'MechanicalKeyboards',
      'time_elasped': 30927.51508617401,
      'title': 'Apex - High-profile angled aluminum case for the Planck'},
     {'is_video': False,
      'num_comments': 16,
      'num_crossposts': 0,
      'subreddit': 'ExpandDong',
      'time_elasped': 37974.51508784294,
      'title': 'just in case sbubby rejects me'},
     {'is_video': True,
      'num_comments': 116,
      'num_crossposts': 1,
      'subreddit': 'holdmyfries',
      'time_elasped': 42641.515088796616,
      'title': 'HMF while this kid smacks my bag with a bat.'},
     {'is_video': False,
      'num_comments': 15,
      'num_crossposts': 0,
      'subreddit': 'Breath_of_the_Wild',
      'time_elasped': 43011.51509094238,
      'title': 'Cant deny that I use the Majoras all the time. If only we could get the Skull Kid clothing.'},
     {'is_video': True,
      'num_comments': 22,
      'num_crossposts': 0,
      'subreddit': 'raining',
      'time_elasped': 48350.51509308815,
      'title': 'It has been raining for 3 days straight'},
     {'is_video': False,
      'num_comments': 9,
      'num_crossposts': 1,
      'subreddit': 'goddesses',
      'time_elasped': 52655.5150949955,
      'title': 'Rachel Cook'},
     {'is_video': False,
      'num_comments': 10,
      'num_crossposts': 0,
      'subreddit': 'oldpeoplefacebook',
      'time_elasped': 18664.515095949173,
      'title': 'Another gem from Deborah'},
     {'is_video': False,
      'num_comments': 691,
      'num_crossposts': 0,
      'subreddit': 'AskMen',
      'time_elasped': 41690.51509809494,
      'title': 'If a girl invited you over for a home cooked meal, what is the one meal she could cook that would make you think you better put a ring on it?'},
     {'is_video': False,
      'num_comments': 277,
      'num_crossposts': 0,
      'subreddit': 'AskOuija',
      'time_elasped': 60201.51510000229,
      'title': 'Before I wake up, I ______'},
     {'is_video': False,
      'num_comments': 111,
      'num_crossposts': 0,
      'subreddit': 'whatisthisthing',
      'time_elasped': 52722.51510190964,
      'title': 'My mother bought me this T-shirt from the little money she saved for my birthday. I have been trying to figure out the print on the tee. Can someone help me??'},
     {'is_video': False,
      'num_comments': 17,
      'num_crossposts': 0,
      'subreddit': 'JonTron',
      'time_elasped': 37933.51510286331,
      'title': 'Half the Universe in Avengers 4'},
     {'is_video': False,
      'num_comments': 25,
      'num_crossposts': 0,
      'subreddit': 'comicbooks',
      'time_elasped': 42084.51510500908,
      'title': 'First Mary Jane Watson sketch by Jazzy Johnny Romita (1966)'},
     {'is_video': False,
      'num_comments': 1,
      'num_crossposts': 0,
      'subreddit': 'puns',
      'time_elasped': 31894.515107154846,
      'title': 'Seen in down town Las Vegas'},
     {'is_video': False,
      'num_comments': 41,
      'num_crossposts': 0,
      'subreddit': 'Watches',
      'time_elasped': 28119.515109062195,
      'title': '[Rolex Daytona] My father’s Stainless Daytona.'},
     {'is_video': False,
      'num_comments': 329,
      'num_crossposts': 7,
      'subreddit': 'gaming',
      'time_elasped': 40581.51510977745,
      'title': 'God of War team makes custom gifs'},
     {'is_video': False,
      'num_comments': 22,
      'num_crossposts': 0,
      'subreddit': 'stevenuniverse',
      'time_elasped': 28987.515111923218,
      'title': "A small Pearl for My Diamond. c':"},
     {'is_video': False,
      'num_comments': 122,
      'num_crossposts': 0,
      'subreddit': 'CrappyDesign',
      'time_elasped': 60744.515114068985,
      'title': '“why are we getting so many dislikes on our advertisement”'},
     {'is_video': False,
      'num_comments': 12,
      'num_crossposts': 0,
      'subreddit': 'Persona5',
      'time_elasped': 32045.51511502266,
      'title': "They're back at it again."},
     {'is_video': False,
      'num_comments': 7,
      'num_crossposts': 2,
      'subreddit': 'nasa',
      'time_elasped': 28269.515117168427,
      'title': 'Seven Dusty Sisters'},
     {'is_video': False,
      'num_comments': 83,
      'num_crossposts': 1,
      'subreddit': 'PewdiepieSubmissions',
      'time_elasped': 42129.515119075775,
      'title': 'Pewds need to see this OMG *clickbate*'},
     {'is_video': False,
      'num_comments': 17,
      'num_crossposts': 0,
      'subreddit': 'fountainpens',
      'time_elasped': 41222.51512002945,
      'title': 'This steel roll shows the various stages of stamping a pen nib (X-post from /r/mildlyinteresting)'},
     {'is_video': False,
      'num_comments': 358,
      'num_crossposts': 0,
      'subreddit': 'rupaulsdragrace',
      'time_elasped': 33022.5151219368,
      'title': 'Shea has something to say about Aja wanting to collab with Azalea Banks'},
     {'is_video': True,
      'num_comments': 1825,
      'num_crossposts': 4,
      'subreddit': 'europe',
      'time_elasped': 64562.515124082565,
      'title': 'The Hero : Mamoudou Gassama saving a children in Paris this weekend.'},
     {'is_video': False,
      'num_comments': 38,
      'num_crossposts': 0,
      'subreddit': 'xkcd',
      'time_elasped': 22113.51512503624,
      'title': 'xkcd 1999: Selection Effect'},
     {'is_video': False,
      'num_comments': 53,
      'num_crossposts': 0,
      'subreddit': 'CampingandHiking',
      'time_elasped': 37683.51512694359,
      'title': 'What I woke up to yesterday at Algonquin Provincial Park in Ontario, CA'},
     {'is_video': False,
      'num_comments': 65,
      'num_crossposts': 0,
      'subreddit': 'mentalhealth',
      'time_elasped': 16951.515129089355,
      'title': 'Mental Health Awareness Month: I have schizoaffective bipolar type and I just graduated with my BA in English (Magna Cum Laude) at age 28 after dropping out of high school and having intermittently relapsed into psychosis for a duration of 8 hospitalizations. With loving support success is possible'},
     {'is_video': False,
      'num_comments': 45,
      'num_crossposts': 0,
      'subreddit': 'xxfitness',
      'time_elasped': 26068.51513004303,
      'title': 'Confidence victory: Getting unsolicited advice in the gym'},
     {'is_video': False,
      'num_comments': 131,
      'num_crossposts': 1,
      'subreddit': 'de',
      'time_elasped': 62066.51513195038,
      'title': 'Die deutsche Sprache in einer Nussschale'},
     {'is_video': False,
      'num_comments': 19,
      'num_crossposts': 0,
      'subreddit': 'terriblefacebookmemes',
      'time_elasped': 26948.515133857727,
      'title': 'TIL helmets were never used in any sport, profession, or military until 1974'},
     {'is_video': False,
      'num_comments': 58,
      'num_crossposts': 0,
      'subreddit': 'gameofthrones',
      'time_elasped': 42773.515136003494,
      'title': '[no spoilers] I took a photo of my friend in a Game of Thrones costume she made herself'},
     {'is_video': False,
      'num_comments': 54,
      'num_crossposts': 0,
      'subreddit': 'GamersRiseUp',
      'time_elasped': 42014.51513695717,
      'title': 'This sub in a nutshell'},
     {'is_video': True,
      'num_comments': 17,
      'num_crossposts': 2,
      'subreddit': 'PartyParrot',
      'time_elasped': 26224.515139102936,
      'title': '“My human will never find me he... NO HUMAN”'},
     {'is_video': False,
      'num_comments': 19,
      'num_crossposts': 0,
      'subreddit': 'deathgrips',
      'time_elasped': 21382.515141010284,
      'title': 'Spongebob DG Album/EP Covers'},
     {'is_video': False,
      'num_comments': 190,
      'num_crossposts': 0,
      'subreddit': 'WatchPeopleDieInside',
      'time_elasped': 46136.51514291763,
      'title': '“I’m a good child.”'},
     {'is_video': False,
      'num_comments': 16,
      'num_crossposts': 0,
      'subreddit': 'shittymoviedetails',
      'time_elasped': 35632.5151450634,
      'title': 'In The Simpsons Movie (2007), the Simpsons live in the same house that they live in in the tv show'},
     {'is_video': False,
      'num_comments': 88,
      'num_crossposts': 0,
      'subreddit': 'Shitty_Car_Mods',
      'time_elasped': 47097.51514697075,
      'title': 'PARK BENCH (Ruined a nice car)'},
     {'is_video': False,
      'num_comments': 23,
      'num_crossposts': 2,
      'subreddit': 'Cinemagraphs',
      'time_elasped': 44993.515149116516,
      'title': 'Hmmm'},
     {'is_video': False,
      'num_comments': 23,
      'num_crossposts': 0,
      'subreddit': 'PhonesAreBad',
      'time_elasped': 26618.515151023865,
      'title': 'Wow thanks dad'},
     {'is_video': False,
      'num_comments': 74,
      'num_crossposts': 0,
      'subreddit': 'youtubehaiku',
      'time_elasped': 57100.51515197754,
      'title': '[Haiku] Man With Severe Case Of Tourette’s'},
     {'is_video': False,
      'num_comments': 24,
      'num_crossposts': 0,
      'subreddit': 'reactiongifs',
      'time_elasped': 39688.51515388489,
      'title': 'MRW I get to skip the line at Chipotle because I ordered my food through the app.'},
     {'is_video': False,
      'num_comments': 39,
      'num_crossposts': 0,
      'subreddit': 'AccidentalRacism',
      'time_elasped': 43693.51515483856,
      'title': 'Even pencil sharpeners'},
     {'is_video': False,
      'num_comments': 38,
      'num_crossposts': 0,
      'subreddit': 'holdmyfeedingtube',
      'time_elasped': 25682.51515698433,
      'title': 'HMFT after I slide down this railing at the stadium'},
     {'is_video': False,
      'num_comments': 14,
      'num_crossposts': 0,
      'subreddit': 'AmateurRoomPorn',
      'time_elasped': 34203.5151591301,
      'title': 'Our first apartment together, I miss it so much! - CA'},
     {'is_video': False,
      'num_comments': 90,
      'num_crossposts': 0,
      'subreddit': 'motorcycles',
      'time_elasped': 40328.515161037445,
      'title': 'For all you motorcycle-loving tech geeks'},
     {'is_video': False,
      'num_comments': 58,
      'num_crossposts': 0,
      'subreddit': 'DarlingInTheFranxx',
      'time_elasped': 40149.515162944794,
      'title': 'Coincidence?'},
     {'is_video': False,
      'num_comments': 27,
      'num_crossposts': 0,
      'subreddit': 'Rabbits',
      'time_elasped': 42588.51516389847,
      'title': 'Been going through a tough time lately and some days it’s hard to even get out of bed but I always have this little man and his baby sister to give me the motivation I need to start my day.'},
     {'is_video': False,
      'num_comments': 40,
      'num_crossposts': 0,
      'subreddit': 'warriors',
      'time_elasped': 21919.515166044235,
      'title': 'Today is my birthday. Last time the Warriors played on my birthday, Klay Thompson went off vs OKC in Game 6. Let’s win this series.'},
     {'is_video': False,
      'num_comments': 45,
      'num_crossposts': 0,
      'subreddit': 'ihavesex',
      'time_elasped': 50961.515167951584,
      'title': 'Youtube comment on airsoft channels video'},
     {'is_video': False,
      'num_comments': 68,
      'num_crossposts': 0,
      'subreddit': 'neoliberal',
      'time_elasped': 30042.515168905258,
      'title': 'Average Trump Voter'},
     {'is_video': False,
      'num_comments': 10,
      'num_crossposts': 0,
      'subreddit': 'firefly',
      'time_elasped': 23280.515171051025,
      'title': 'This was in my local game stores restroom awhile back!'},
     {'is_video': False,
      'num_comments': 87,
      'num_crossposts': 1,
      'subreddit': 'kpop',
      'time_elasped': 44887.5151720047,
      'title': "President of South Korea, Moon Jae-in, congratulates BTS on their album's success on the Billboard 200 albums chart"},
     {'is_video': False,
      'num_comments': 13,
      'num_crossposts': 1,
      'subreddit': 'Greekgodx',
      'time_elasped': 37610.51517391205,
      'title': 'Pablo Escobar escaping with 25 pounds of cocaine. (circa 1955)'},
     {'is_video': False,
      'num_comments': 272,
      'num_crossposts': 0,
      'subreddit': 'starterpacks',
      'time_elasped': 32474.515175819397,
      'title': '"College Age Conservative Jew" Starter Pack'},
     {'is_video': False,
      'num_comments': 143,
      'num_crossposts': 2,
      'subreddit': 'Documentaries',
      'time_elasped': 56356.515177965164,
      'title': "Fukushima Uncensored (2015) - The story of one of history's most devastating nuclear disasters as told by the workers who worked to contain it."},
     {'is_video': False,
      'num_comments': 31,
      'num_crossposts': 2,
      'subreddit': 'climbing',
      'time_elasped': 46717.51517891884,
      'title': 'Feel like showing off my new chalkbag! Girlfriend made it from an old pair of jeans'},
     {'is_video': False,
      'num_comments': 67,
      'num_crossposts': 0,
      'subreddit': 'Prematurecelebration',
      'time_elasped': 31292.515181064606,
      'title': 'The Perfect Ending!'},
     {'is_video': False,
      'num_comments': 26,
      'num_crossposts': 0,
      'subreddit': 'FanTheories',
      'time_elasped': 38671.51518321037,
      'title': '[The Office] Kelly always ate Erin’s lunch (S6E1) because Erin was still labeling her lunch as “Kelly.”'},
     {'is_video': False,
      'num_comments': 258,
      'num_crossposts': 0,
      'subreddit': 'LivestreamFail',
      'time_elasped': 39285.5151848793,
      'title': 'Aggressive stream sniper enters the RV.'},
     {'is_video': False,
      'num_comments': 1054,
      'num_crossposts': 1,
      'subreddit': 'news',
      'time_elasped': 54383.51518702507,
      'title': 'Migrant who saved young boy to be made French citizen'},
     {'is_video': False,
      'num_comments': 24,
      'num_crossposts': 0,
      'subreddit': 'smoobypost',
      'time_elasped': 52482.515187978745,
      'title': 'eye of the smoober'},
     {'is_video': False,
      'num_comments': 197,
      'num_crossposts': 3,
      'subreddit': 'Whatcouldgowrong',
      'time_elasped': 57019.51519012451,
      'title': 'Not holding a shotgun property when pulling the trigger. WCGW'},
     {'is_video': True,
      'num_comments': 25,
      'num_crossposts': 0,
      'subreddit': 'PUBGXboxOne',
      'time_elasped': 27496.51519203186,
      'title': 'When your friend tells you the patch made the game better'},
     {'is_video': False,
      'num_comments': 25,
      'num_crossposts': 0,
      'subreddit': 'IDontWorkHereLady',
      'time_elasped': 39184.515192985535,
      'title': 'I AM a student here'},
     {'is_video': False,
      'num_comments': 118,
      'num_crossposts': 0,
      'subreddit': 'Seaofthieves',
      'time_elasped': 42597.51519489288,
      'title': 'For those eagerly awaiting the Hungering Deep update tomorrow...'},
     {'is_video': False,
      'num_comments': 4,
      'num_crossposts': 0,
      'subreddit': 'pitbulls',
      'time_elasped': 31709.515195846558,
      'title': 'Otis is always mean mugging in the nicest of ways.'},
     {'is_video': False,
      'num_comments': 14,
      'num_crossposts': 1,
      'subreddit': 'redditgetsdrawn',
      'time_elasped': 31934.515197992325,
      'title': 'Friend took a photo of me swinging my shirt around and it looks like i tamed a crow to perch on my hand, would be cool to see how you can draw it!'},
     {'is_video': False,
      'num_comments': 38,
      'num_crossposts': 0,
      'subreddit': 'techsupportmacgyver',
      'time_elasped': 38948.51519989967,
      'title': 'Auxilliary cord input .'},
     {'is_video': False,
      'num_comments': 34,
      'num_crossposts': 1,
      'subreddit': 'moviescirclejerk',
      'time_elasped': 28914.51520204544,
      'title': 'BREAKING NEWS: Kim Jong Un decides to give up his nukes after discovering an even bigger bomb'},
     {'is_video': False,
      'num_comments': 164,
      'num_crossposts': 0,
      'subreddit': 'beholdthemasterrace',
      'time_elasped': 59938.515202999115,
      'title': 'White supremacy groups always have the most attractive and well dressed members...'},
     {'is_video': False,
      'num_comments': 97,
      'num_crossposts': 1,
      'subreddit': 'boottoobig',
      'time_elasped': 63652.51520514488,
      'title': 'Superman wears a cape, Gandalf has a staff'},
     {'is_video': False,
      'num_comments': 52,
      'num_crossposts': 0,
      'subreddit': 'lgbt',
      'time_elasped': 47017.51520681381,
      'title': '2015-2018 living authentically! 🙍🏻\u200d♂️&lt;💁🏻'},
     {'is_video': False,
      'num_comments': 71,
      'num_crossposts': 0,
      'subreddit': 'lewronggeneration',
      'time_elasped': 38604.51520895958,
      'title': '#NotMyEra'},
     {'is_video': False,
      'num_comments': 143,
      'num_crossposts': 0,
      'subreddit': 'Braves',
      'time_elasped': 12204.515209913254,
      'title': 'DISCOUNT DANSBY WALKED US OFF UPCHOP THREAD!'},
     {'is_video': False,
      'num_comments': 25,
      'num_crossposts': 0,
      'subreddit': 'MachinePorn',
      'time_elasped': 29795.51521205902,
      'title': 'Crafting a V8 engine for the Aston Martin Vantage'},
     {'is_video': False,
      'num_comments': 34,
      'num_crossposts': 0,
      'subreddit': 'IASIP',
      'time_elasped': 37589.51521396637,
      'title': 'Have A Great Memorial Day... ROCK, FLAG &amp; EAGLE!'},
     {'is_video': False,
      'num_comments': 17,
      'num_crossposts': 1,
      'subreddit': 'QuotesPorn',
      'time_elasped': 51168.51521515846,
      'title': '" Morning is wonderful. Its only drawback is..." - Glen Cook [1280 x 706]'},
     {'is_video': False,
      'num_comments': 20,
      'num_crossposts': 0,
      'subreddit': 'VaporwaveAesthetics',
      'time_elasped': 36263.51521682739,
      'title': 'Legend of Zelda [Vaporwave Edition]'},
     {'is_video': False,
      'num_comments': 18,
      'num_crossposts': 0,
      'subreddit': 'Badfaketexts',
      'time_elasped': 26495.515218019485,
      'title': 'DAB'},
     {'is_video': False,
      'num_comments': 32,
      'num_crossposts': 0,
      'subreddit': 'ProgrammerHumor',
      'time_elasped': 26712.515219926834,
      'title': 'Is that recursion ?!'},
     {'is_video': False,
      'num_comments': 38,
      'num_crossposts': 0,
      'subreddit': 'CasualUK',
      'time_elasped': 32079.51522088051,
      'title': 'Oxford flaunting it’s elitism'},
     {'is_video': False,
      'num_comments': 21,
      'num_crossposts': 0,
      'subreddit': 'MadeMeSmile',
      'time_elasped': 43453.515223026276,
      'title': "Couple with Down's syndrome celebrate 22 years of marriage."},
     {'is_video': False,
      'num_comments': 9,
      'num_crossposts': 1,
      'subreddit': 'TheWayWeWere',
      'time_elasped': 38863.51522517204,
      'title': 'Note the string - Taking selfies in 1937'},
     {'is_video': False,
      'num_comments': 51,
      'num_crossposts': 0,
      'subreddit': 'creepyPMs',
      'time_elasped': 64953.51522612572,
      'title': 'The nerve of some people'},
     {'is_video': False,
      'num_comments': 318,
      'num_crossposts': 0,
      'subreddit': 'TheExpanse',
      'time_elasped': 43351.51522779465,
      'title': 'The Expanse needs to be recognized for their embrace of older female leading characters.'},
     {'is_video': False,
      'num_comments': 401,
      'num_crossposts': 0,
      'subreddit': 'personalfinance',
      'time_elasped': 69636.51522994041,
      'title': "Just kicked cancers butt, now I've got a bigger problem."},
     {'is_video': False,
      'num_comments': 43,
      'num_crossposts': 0,
      'subreddit': 'FireEmblemHeroes',
      'time_elasped': 31709.51523208618,
      'title': 'Sakura as Cardcaptor Sakura ;u;'},
     {'is_video': True,
      'num_comments': 114,
      'num_crossposts': 0,
      'subreddit': 'forhonor',
      'time_elasped': 45304.51523399353,
      'title': 'This execution would be so sick for Warden... Am I the only one, who wants this realized?'},
     {'is_video': True,
      'num_comments': 49,
      'num_crossposts': 1,
      'subreddit': 'MasterReturns',
      'time_elasped': 54258.51523518562,
      'title': 'Had to put our pup down at only two years old - this is her after my wife and I returned from a 4 day trip'},
     {'is_video': False,
      'num_comments': 123,
      'num_crossposts': 2,
      'subreddit': 'gay_irl',
      'time_elasped': 49054.51523709297,
      'title': 'gay_irl'},
     {'is_video': False,
      'num_comments': 79,
      'num_crossposts': 0,
      'subreddit': 'projectcar',
      'time_elasped': 31077.51523900032,
      'title': 'I’ve seen three mechanics die in the ER this year from cars falling on them. Please use proper jacks and stands, on level hard surface with wheel chocks. Cutting corners is not worth the risks.'},
     {'is_video': False,
      'num_comments': 259,
      'num_crossposts': 4,
      'subreddit': 'AccidentalRenaissance',
      'time_elasped': 80858.515239954,
      'title': 'Saint Dad, Out Father of Newborn'},
     {'is_video': False,
      'num_comments': 16,
      'num_crossposts': 0,
      'subreddit': 'NobodyAsked',
      'time_elasped': 21893.515242099762,
      'title': 'Why mention it tho.'},
     {'is_video': False,
      'num_comments': 17,
      'num_crossposts': 0,
      'subreddit': 'Denmark',
      'time_elasped': 26560.51524400711,
      'title': 'Are you Christian?'},
     {'is_video': True,
      'num_comments': 256,
      'num_crossposts': 2,
      'subreddit': 'FortNiteBR',
      'time_elasped': 37350.515244960785,
      'title': "Website I've been working on to track all the Battle Pass challenges in one place on an interactive map"},
     {'is_video': False,
      'num_comments': 121,
      'num_crossposts': 0,
      'subreddit': 'HistoryMemes',
      'time_elasped': 53890.51524710655,
      'title': 'Back to the Future teaches history'},
     {'is_video': False,
      'num_comments': 45,
      'num_crossposts': 0,
      'subreddit': 'exmormon',
      'time_elasped': 21912.515248060226,
      'title': "TBM mom says my dress is much too short... I'm glad to be an adult who is financially independent and free of all of the churches shit."},
     {'is_video': False,
      'num_comments': 54,
      'num_crossposts': 0,
      'subreddit': 'EngineeringStudents',
      'time_elasped': 57529.515312194824,
      'title': 'Why would eˣ do this?'},
     {'is_video': True,
      'num_comments': 60,
      'num_crossposts': 0,
      'subreddit': 'woof_irl',
      'time_elasped': 76038.51531505585,
      'title': 'woof_irl'},
     {'is_video': False,
      'num_comments': 46,
      'num_crossposts': 0,
      'subreddit': 'eagles',
      'time_elasped': 35604.515316963196,
      'title': 'The one word that describes the Super Bowl trophy in Philly'},
     {'is_video': False,
      'num_comments': 17,
      'num_crossposts': 0,
      'subreddit': 'CityPorn',
      'time_elasped': 50880.51531791687,
      'title': 'Gothenburg, Sweden (OC) [5103x3436]'},
     {'is_video': False,
      'num_comments': 51,
      'num_crossposts': 0,
      'subreddit': 'OnePiece',
      'time_elasped': 44536.51532006264,
      'title': 'Awesome One Piece Art , by Eiichiro Oda for Weekly Shonen Jump 2001'},
     {'is_video': False,
      'num_comments': 13,
      'num_crossposts': 0,
      'subreddit': 'Battlecars',
      'time_elasped': 25422.51532101631,
      'title': 'Stumbled upon this beast on the way home from a RallyCross'},
     {'is_video': False,
      'num_comments': 24,
      'num_crossposts': 0,
      'subreddit': 'hiking',
      'time_elasped': 34344.51532292366,
      'title': 'I went on my first hike alone this weekend. Bald Hills trail, Jasper National Park, Alberta, Canada.'},
     {'is_video': False,
      'num_comments': 6,
      'num_crossposts': 0,
      'subreddit': 'Delightfullychubby',
      'time_elasped': 42781.51532483101,
      'title': 'A wise cat once said, “If it fits, I sits”'},
     {'is_video': False,
      'num_comments': 125,
      'num_crossposts': 2,
      'subreddit': 'Art',
      'time_elasped': 42222.5153260231,
      'title': 'Andrew (2017), Oil on canvas, 65 x 75 cm'},
     {'is_video': False,
      'num_comments': 38,
      'num_crossposts': 1,
      'subreddit': 'WhatsWrongWithYourDog',
      'time_elasped': 50095.51532793045,
      'title': "Rescue pitty isn't very photogenic but she sure is sweet!"},
     {'is_video': False,
      'num_comments': 60,
      'num_crossposts': 0,
      'subreddit': 'legaladvice',
      'time_elasped': 28186.515330076218,
      'title': 'She lied about her age'},
     {'is_video': False,
      'num_comments': 122,
      'num_crossposts': 0,
      'subreddit': 'france',
      'time_elasped': 28409.515331029892,
      'title': 'Pour toi serge'},
     {'is_video': False,
      'num_comments': 325,
      'num_crossposts': 2,
      'subreddit': 'DnD',
      'time_elasped': 60771.51533317566,
      'title': '[OC] Feel free to use my clever lever riddle!'},
     {'is_video': False,
      'num_comments': 89,
      'num_crossposts': 0,
      'subreddit': 'LiverpoolFC',
      'time_elasped': 14401.515336036682,
      'title': 'What a lean son'},
     {'is_video': False,
      'num_comments': 94,
      'num_crossposts': 0,
      'subreddit': 'OutOfTheLoop',
      'time_elasped': 47434.51533699036,
      'title': "In the last few weeks I've noticed a little hand next to some people's name on Facebook. What does it mean?"},
     {'is_video': False,
      'num_comments': 25,
      'num_crossposts': 0,
      'subreddit': 'creepy',
      'time_elasped': 16050.515338897705,
      'title': 'I drew that creepy skull that was posted here a few days ago'},
     {'is_video': False,
      'num_comments': 48,
      'num_crossposts': 0,
      'subreddit': 'noisygifs',
      'time_elasped': 33991.51534104347,
      'title': 'Could you guys turn it down a little?!'},
     {'is_video': False,
      'num_comments': 160,
      'num_crossposts': 0,
      'subreddit': 'BlackPeopleTwitter',
      'time_elasped': 50153.51534199715,
      'title': 'He put the team on his back doe'},
     {'is_video': False,
      'num_comments': 142,
      'num_crossposts': 0,
      'subreddit': 'funhaus',
      'time_elasped': 40140.515343904495,
      'title': 'SUCK A DICTATOR - Demo Disk Gameplay'},
     {'is_video': False,
      'num_comments': 5,
      'num_crossposts': 0,
      'subreddit': 'ImaginaryLeviathans',
      'time_elasped': 32706.515345811844,
      'title': 'Thunder of the Abyss by Marcus Reyno'},
     {'is_video': False,
      'num_comments': 112,
      'num_crossposts': 0,
      'subreddit': 'FellowKids',
      'time_elasped': 43772.51534700394,
      'title': 'Haha yes the president too enjoys a good memay'},
     {'is_video': False,
      'num_comments': 9,
      'num_crossposts': 0,
      'subreddit': 'Megumin',
      'time_elasped': 36442.515348911285,
      'title': "HE'S MINE."},
     {'is_video': False,
      'num_comments': 42,
      'num_crossposts': 1,
      'subreddit': 'india',
      'time_elasped': 35943.51535010338,
      'title': 'Fixing the electricity, Guwahati. The mind boggles!'},
     {'is_video': False,
      'num_comments': 14,
      'num_crossposts': 0,
      'subreddit': 'HaloOnline',
      'time_elasped': 15570.515352010727,
      'title': 'Master Chief Helmet Equip (Doom Inspired)'},
     {'is_video': False,
      'num_comments': 30,
      'num_crossposts': 0,
      'subreddit': 'thewalkingdead',
      'time_elasped': 24827.515354156494,
      'title': 'My Negan cosplay for Phoenix Comic Fest 2018!'},
     {'is_video': False,
      'num_comments': 56,
      'num_crossposts': 0,
      'subreddit': 'bostonceltics',
      'time_elasped': 30866.515355825424,
      'title': 'Aron Baynes says he just told the team that in 2013, his Spurs lost a Game 7 and it “galvanized the group” for the next season. The following year, the Spurs came back and won it all.'},
     {'is_video': True,
      'num_comments': 223,
      'num_crossposts': 0,
      'subreddit': 'Unexpected',
      'time_elasped': 55517.51535797119,
      'title': 'Disarmed'},
     {'is_video': False,
      'num_comments': 13,
      'num_crossposts': 0,
      'subreddit': 'corgi',
      'time_elasped': 47557.51536011696,
      'title': 'This little cutie will be coming home in a few weeks!'},
     {'is_video': False,
      'num_comments': 36,
      'num_crossposts': 3,
      'subreddit': 'oddlysatisfying',
      'time_elasped': 28659.515361070633,
      'title': 'These dots moving'},
     {'is_video': False,
      'num_comments': 138,
      'num_crossposts': 0,
      'subreddit': 'Bad_Cop_No_Donut',
      'time_elasped': 41958.51536297798,
      'title': 'Cops Raid School, Hold Teacher at Gun Point, Terrify Kids to Look for an Absent Student'},
     {'is_video': False,
      'num_comments': 59,
      'num_crossposts': 0,
      'subreddit': 'brasil',
      'time_elasped': 36868.515364170074,
      'title': 'Como lidar com a procrastinação'},
     {'is_video': False,
      'num_comments': 23,
      'num_crossposts': 0,
      'subreddit': 'Autos',
      'time_elasped': 43711.515365839005,
      'title': 'I saw this LaFerrari last year, it just looks and sounds amazing'},
     {'is_video': False,
      'num_comments': 80,
      'num_crossposts': 0,
      'subreddit': 'Anticonsumption',
      'time_elasped': 24244.515367031097,
      'title': '“We need to stop global warming but I refuse to make tough choices” Starter Pack'},
     {'is_video': False,
      'num_comments': 18,
      'num_crossposts': 0,
      'subreddit': 'RedditLaqueristas',
      'time_elasped': 39839.515368938446,
      'title': 'Freehand nails inspired by my dress'},
     {'is_video': False,
      'num_comments': 130,
      'num_crossposts': 0,
      'subreddit': 'cars',
      'time_elasped': 29044.515371084213,
      'title': 'Ford F150 Diesel Review: A truck that gets 34.3 MPG highway in the real world'},
     {'is_video': False,
      'num_comments': 163,
      'num_crossposts': 0,
      'subreddit': 'drunk',
      'time_elasped': 72387.51537299156,
      'title': 'I’m drunk At my friends wedding. Every upvote is a dollar I’ll put in their hair and and a congratulations I’ll say to them.'},
     {'is_video': False,
      'num_comments': 543,
      'num_crossposts': 0,
      'subreddit': 'rage',
      'time_elasped': 48792.515374183655,
      'title': 'Cops bash woman on the beach for underage drinking.'},
     {'is_video': False,
      'num_comments': 47,
      'num_crossposts': 1,
      'subreddit': 'greatawakening',
      'time_elasped': 28040.515376091003,
      'title': '#WhereAreTheChildren Greatest mindfuck of the week! #MAGA!'},
     {'is_video': False,
      'num_comments': 92,
      'num_crossposts': 0,
      'subreddit': 'malelivingspace',
      'time_elasped': 46275.51537799835,
      'title': 'My Cozy Fargo, ND Studio!'},
     {'is_video': False,
      'num_comments': 55,
      'num_crossposts': 0,
      'subreddit': 'drawing',
      'time_elasped': 41633.5153799057,
      'title': 'My son likes to draw. He’d love to hear what you think of his latest creation'},
     {'is_video': False,
      'num_comments': 17,
      'num_crossposts': 0,
      'subreddit': 'bayarea',
      'time_elasped': 37575.51538205147,
      'title': 'I was told this deserves to be posted here. Low flyover requested by our pilot as we headed to Palm Springs (circa 2016)'},
     {'is_video': False,
      'num_comments': 29,
      'num_crossposts': 0,
      'subreddit': 'uglyduckling',
      'time_elasped': 19651.515383958817,
      'title': 'Back surgery to standing tall and walking on my own!'},
     {'is_video': False,
      'num_comments': 110,
      'num_crossposts': 0,
      'subreddit': 'pettyrevenge',
      'time_elasped': 61549.51538515091,
      'title': 'ID Please'},
     {'is_video': False,
      'num_comments': 6,
      'num_crossposts': 0,
      'subreddit': 'Purrito',
      'time_elasped': 31227.51538681984,
      'title': 'First time purrito'},
     {'is_video': False,
      'num_comments': 13,
      'num_crossposts': 0,
      'subreddit': 'sexygirls',
      'time_elasped': 43722.51538896561,
      'title': 'Super Hot'},
     {'is_video': False,
      'num_comments': 99,
      'num_crossposts': 0,
      'subreddit': 'scifi',
      'time_elasped': 33127.515390872955,
      'title': 'In my opinion, Fringe was better than most people seem to think!'},
     {'is_video': False,
      'num_comments': 60,
      'num_crossposts': 0,
      'subreddit': 'GodofWar',
      'time_elasped': 36502.51539206505,
      'title': 'Most EPIC fight scene (/spoiler)'},
     {'is_video': False,
      'num_comments': 34,
      'num_crossposts': 0,
      'subreddit': 'polandball',
      'time_elasped': 17163.515393972397,
      'title': 'This Polandball was drawn with a valid license'},
     {'is_video': False,
      'num_comments': 5,
      'num_crossposts': 1,
      'subreddit': 'catsareliquid',
      'time_elasped': 28810.515396118164,
      'title': 'Our cow loves to melt in the sun'},
     {'is_video': False,
      'num_comments': 17,
      'num_crossposts': 0,
      'subreddit': 'TalesFromYourServer',
      'time_elasped': 30095.51539707184,
      'title': 'The last prank call'},
     {'is_video': False,
      'num_comments': 9,
      'num_crossposts': 0,
      'subreddit': '30ROCK',
      'time_elasped': 31471.515398979187,
      'title': 'Happy Memorial Day to both the troops and troupes out there'},
     {'is_video': False,
      'num_comments': 69,
      'num_crossposts': 0,
      'subreddit': 'NewPatriotism',
      'time_elasped': 36544.515400886536,
      'title': 'On Memorial Day, one President honors the sacrifice of fallen Americans. Another exploits fallen Americans to peddle merchandise.'},
     {'is_video': False,
      'num_comments': 66,
      'num_crossposts': 0,
      'subreddit': 'gaybros',
      'time_elasped': 33312.51540207863,
      'title': "Russian football hooligans warn gay England fans 'we'll stab you at World Cup'"},
     {'is_video': False,
      'num_comments': 333,
      'num_crossposts': 0,
      'subreddit': 'britishproblems',
      'time_elasped': 54837.51540398598,
      'title': 'Apparently we needed a poll to know that WH Smith is the worst High Street shop.'},
     {'is_video': False,
      'num_comments': 16,
      'num_crossposts': 0,
      'subreddit': 'offlineTV',
      'time_elasped': 30126.51540493965,
      'title': 'The roasts keep coming from Chris'},
     {'is_video': False,
      'num_comments': 44,
      'num_crossposts': 0,
      'subreddit': 'Seattle',
      'time_elasped': 36244.51540708542,
      'title': 'Effects of Japanese American Internment on a 2nd grade Seattle classroom 76 years ago - May 27, 1942'},
     {'is_video': False,
      'num_comments': 19,
      'num_crossposts': 0,
      'subreddit': 'holdmybeer',
      'time_elasped': 19402.515408992767,
      'title': 'HMB while I help this lady get her baby inside'},
     {'is_video': False,
      'num_comments': 99,
      'num_crossposts': 0,
      'subreddit': 'pics',
      'time_elasped': 32223.51540994644,
      'title': 'This WWII veteran was fatally shot Nov. 20, 1943 on the island of Tarawa in the Pacific. He was buried on the island in a lost cemetery. His remains were found in 2015 and he was identified in 2017. Burial today. Thank you for your service and sacrifice.'},
     {'is_video': False,
      'num_comments': 17,
      'num_crossposts': 0,
      'subreddit': 'detroitlions',
      'time_elasped': 24520.51541185379,
      'title': 'F.T.P.'},
     {'is_video': False,
      'num_comments': 29,
      'num_crossposts': 0,
      'subreddit': 'destiny2',
      'time_elasped': 34688.51541399956,
      'title': 'Destiny poster Infinity war / Guardians of the galaxy style done by me :)'},
     {'is_video': False,
      'num_comments': 49,
      'num_crossposts': 1,
      'subreddit': 'mildlyinteresting',
      'time_elasped': 43778.51541495323,
      'title': 'The rain washed away this drawing of the Cheshire Cat leaving only the smile'},
     {'is_video': False,
      'num_comments': 58,
      'num_crossposts': 1,
      'subreddit': 'Conservative',
      'time_elasped': 37559.515417099,
      'title': 'Alabama student who woke up at 4.30am every morning to get a bus to his high school is gifted a car after graduation'},
     {'is_video': False,
      'num_comments': 4,
      'num_crossposts': 0,
      'subreddit': 'arcticmonkeys',
      'time_elasped': 16257.515418767929,
      'title': 'Didn’t know Neville liked the Arctic Monkeys'},
     {'is_video': False,
      'num_comments': 51,
      'num_crossposts': 1,
      'subreddit': 'ContagiousLaughter',
      'time_elasped': 49223.515420913696,
      'title': 'So Funny Little Girl'},
     {'is_video': False,
      'num_comments': 125,
      'num_crossposts': 0,
      'subreddit': 'Blackops4',
      'time_elasped': 33911.51542305946,
      'title': 'BFV no loot boxes confirmed. This is huge. Could this change COD supply drops?'},
     {'is_video': False,
      'num_comments': 898,
      'num_crossposts': 0,
      'subreddit': 'StarWars',
      'time_elasped': 61467.51542496681,
      'title': 'Kudos to Ron Howard’s classy response after Solo’s rough weekend at the box office'},
     {'is_video': False,
      'num_comments': 9,
      'num_crossposts': 1,
      'subreddit': 'ainbow',
      'time_elasped': 26728.51542711258,
      'title': 'My mom bought me a pride outfit without even knowing.'},
     {'is_video': False,
      'num_comments': 37,
      'num_crossposts': 0,
      'subreddit': 'dankchristianmemes',
      'time_elasped': 44828.51542901993,
      'title': 'Masturbation is a sin'},
     {'is_video': False,
      'num_comments': 42,
      'num_crossposts': 0,
      'subreddit': 'teenagers',
      'time_elasped': 25926.515430927277,
      'title': 'We are teens'},
     {'is_video': False,
      'num_comments': 63,
      'num_crossposts': 0,
      'subreddit': 'SeattleWA',
      'time_elasped': 22981.51543188095,
      'title': 'Busted! Seen outside someone’s house at Charles Richey Sr Viewpoint, West Seattle'},
     {'is_video': True,
      'num_comments': 509,
      'num_crossposts': 6,
      'subreddit': 'funny',
      'time_elasped': 34024.51543402672,
      'title': 'I made dis and I hope it makes you laugh.'},
     {'is_video': False,
      'num_comments': 14,
      'num_crossposts': 0,
      'subreddit': 'KingOfTheHill',
      'time_elasped': 38974.51543593407,
      'title': 'When you were the first one who did it but no one cares'},
     {'is_video': False,
      'num_comments': 13,
      'num_crossposts': 0,
      'subreddit': 'FiftyFifty',
      'time_elasped': 12415.51543712616,
      'title': '[50/50] Raccoon riding a bike (SFW) | Man gets beaten severely and almost dies (NSFW)'},
     {'is_video': False,
      'num_comments': 31,
      'num_crossposts': 0,
      'subreddit': 'techsupportgore',
      'time_elasped': 50181.51543903351,
      'title': 'At an aquarium yesterday'},
     {'is_video': False,
      'num_comments': 8,
      'num_crossposts': 0,
      'subreddit': 'nocontextpics',
      'time_elasped': 40124.51544094086,
      'title': 'PIC'},
     {'is_video': False,
      'num_comments': 8,
      'num_crossposts': 0,
      'subreddit': 'Birbs',
      'time_elasped': 40642.51544189453,
      'title': 'LonelyBirb'},
     {'is_video': False,
      'num_comments': 80,
      'num_crossposts': 0,
      'subreddit': 'iamatotalpieceofshit',
      'time_elasped': 43172.5154440403,
      'title': 'This abusive asshole'},
     {'is_video': False,
      'num_comments': 8,
      'num_crossposts': 0,
      'subreddit': 'RetroFuturism',
      'time_elasped': 14509.515445947647,
      'title': "Those Zany 50's"},
     {'is_video': False,
      'num_comments': 109,
      'num_crossposts': 0,
      'subreddit': 'assholedesign',
      'time_elasped': 20646.515448093414,
      'title': 'Fuck this hospital'},
     {'is_video': False,
      'num_comments': 25,
      'num_crossposts': 1,
      'subreddit': 'AskHistorians',
      'time_elasped': 37920.51544904709,
      'title': "The first King of Portugal was a Medieval Erasmus kid: the son of a Burgundian knight and a Castillian noblewoman, he ruled over a population speaking Galaico-Portuguese and Arabic. Do we know what languages he spoke, and if he had an accent? Wasn't he seen as a foreigner in the eyes of the natives?"},
     {'is_video': False,
      'num_comments': 53,
      'num_crossposts': 0,
      'subreddit': 'KeanuBeingAwesome',
      'time_elasped': 74712.51545095444,
      'title': 'Keanu being the (loved) One'},
     {'is_video': False,
      'num_comments': 29,
      'num_crossposts': 0,
      'subreddit': 'reddevils',
      'time_elasped': 19638.51545190811,
      'title': 'Congrats to the newlywed Victor and Maja Lindelof!'},
     {'is_video': False,
      'num_comments': 80,
      'num_crossposts': 0,
      'subreddit': 'NotHowGirlsWork',
      'time_elasped': 39462.51545405388,
      'title': 'MGTOW think orgasms are unhealthy for women'},
     {'is_video': False,
      'num_comments': 317,
      'num_crossposts': 0,
      'subreddit': 'PUBATTLEGROUNDS',
      'time_elasped': 31474.515455007553,
      'title': 'Shroud experienced the bigged D-Sync ever!'},
     {'is_video': False,
      'num_comments': 43,
      'num_crossposts': 0,
      'subreddit': 'customhearthstone',
      'time_elasped': 36765.5154569149,
      'title': 'Now that Well Made Weekend is over, I can finally post my humerus card!'},
     {'is_video': False,
      'num_comments': 4,
      'num_crossposts': 0,
      'subreddit': 'FullmetalAlchemist',
      'time_elasped': 22445.51545906067,
      'title': 'Riza Hawkeye (by バラバババ)'},
     {'is_video': False,
      'num_comments': 11,
      'num_crossposts': 0,
      'subreddit': 'dank_meme',
      'time_elasped': 32786.51546001434,
      'title': 'Survive.'},
     {'is_video': False,
      'num_comments': 2,
      'num_crossposts': 0,
      'subreddit': 'IRLEasterEggs',
      'time_elasped': 24316.515461921692,
      'title': 'Recycle or Die'},
     {'is_video': False,
      'num_comments': 198,
      'num_crossposts': 0,
      'subreddit': 'ireland',
      'time_elasped': 59004.51546382904,
      'title': 'Irish Girls'},
     {'is_video': False,
      'num_comments': 606,
      'num_crossposts': 3,
      'subreddit': 'ShittyLifeProTips',
      'time_elasped': 73631.51546597481,
      'title': 'If you stain a shirt, you can simply outline the stain with a sharpie and give it a name. This will make it seam like you visit islands.'},
     {'is_video': False,
      'num_comments': 19,
      'num_crossposts': 0,
      'subreddit': 'PUBG',
      'time_elasped': 42891.515468120575,
      'title': 'When you’ve been quietly looting for 10 minutes and all of a sudden hear a suppressed SKS...'},
     {'is_video': False,
      'num_comments': 173,
      'num_crossposts': 0,
      'subreddit': 'DIY',
      'time_elasped': 78117.51547002792,
      'title': 'I restored a 1961 vintage tandem bicycle.'},
     {'is_video': False,
      'num_comments': 27,
      'num_crossposts': 0,
      'subreddit': 'NLSSCircleJerk',
      'time_elasped': 27796.515471935272,
      'title': 'This man is actually clinically insane'},
     {'is_video': False,
      'num_comments': 108,
      'num_crossposts': 0,
      'subreddit': 'KitchenConfidential',
      'time_elasped': 32637.51547384262,
      'title': 'Bartender absolutely destroying some burgers at a cook out yesterday.'},
     {'is_video': False,
      'num_comments': 61,
      'num_crossposts': 0,
      'subreddit': 'OldSchoolCool',
      'time_elasped': 43501.515475034714,
      'title': 'My Grandad, a proud Scot. 1940s.'},
     {'is_video': False,
      'num_comments': 96,
      'num_crossposts': 1,
      'subreddit': 'facepalm',
      'time_elasped': 33636.51547694206,
      'title': 'These two idiots on the highway'},
     {'is_video': False,
      'num_comments': 40,
      'num_crossposts': 0,
      'subreddit': 'HFY',
      'time_elasped': 14527.51547908783,
      'title': 'Oh this has not gone well - 117'},
     {'is_video': False,
      'num_comments': 5,
      'num_crossposts': 0,
      'subreddit': 'mlem',
      'time_elasped': 41385.515480041504,
      'title': 'Senior mlem'},
     {'is_video': False,
      'num_comments': 13,
      'num_crossposts': 0,
      'subreddit': 'AnimalCrossing',
      'time_elasped': 33733.51548218727,
      'title': 'Got back to playing AC Gamecube a year ago and forgot about Huggy all this time'},
     {'is_video': False,
      'num_comments': 30,
      'num_crossposts': 0,
      'subreddit': 'NotHowDrugsWork',
      'time_elasped': 16587.5154838562,
      'title': 'I love when someone who has never tripped makes a meme'},
     {'is_video': False,
      'num_comments': 185,
      'num_crossposts': 2,
      'subreddit': 'gatekeeping',
      'time_elasped': 79802.51548480988,
      'title': 'From BuzzFeed Snapchat'},
     {'is_video': True,
      'num_comments': 119,
      'num_crossposts': 1,
      'subreddit': 'sixers',
      'time_elasped': 31367.515486955643,
      'title': 'Fultz jumper'},
     {'is_video': False,
      'num_comments': 13,
      'num_crossposts': 0,
      'subreddit': 'memes',
      'time_elasped': 17762.515487909317,
      'title': 'When LEGO says 8-99 years but...'},
     {'is_video': False,
      'num_comments': 723,
      'num_crossposts': 2,
      'subreddit': 'PublicFreakout',
      'time_elasped': 55047.515490055084,
      'title': 'Cops bash woman on the beach for underage drinking. As soon as one cop says "stop resisting" the other cops starts punching her while she is restrained.'},
     {'is_video': False,
      'num_comments': 8,
      'num_crossposts': 0,
      'subreddit': 'UpliftingNews',
      'time_elasped': 40151.51549220085,
      'title': "(x-post from TIL) Yao Ming's conservation campaigns has led to a 50% drop in shark fin soup consumption in China. He is now working on poaching as well."},
     {'is_video': False,
      'num_comments': 58,
      'num_crossposts': 1,
      'subreddit': 'rockets',
      'time_elasped': 35041.51549386978,
      'title': 'The rest of the NBA to James Harden and Chris Paul right now.'},
     {'is_video': False,
      'num_comments': 40,
      'num_crossposts': 0,
      'subreddit': 'WhyWereTheyFilming',
      'time_elasped': 28173.51549601555,
      'title': 'Just climbing down a ladder into a well... oh and filming for no reason.'},
     {'is_video': False,
      'num_comments': 24,
      'num_crossposts': 0,
      'subreddit': 'Showerthoughts',
      'time_elasped': 18872.515496969223,
      'title': "Identical twins could make a ton of money if one purposely got fat while the other got buff. They'd make a killing selling before and after photos for weight loss advertisements."},
     {'is_video': False,
      'num_comments': 4,
      'num_crossposts': 0,
      'subreddit': 'ChildrenFallingOver',
      'time_elasped': 18086.51549911499,
      'title': 'Just a little wind'},
     {'is_video': False,
      'num_comments': 58,
      'num_crossposts': 0,
      'subreddit': 'LifeProTips',
      'time_elasped': 23001.515500068665,
      'title': 'LPT: If an ad on YouTube is unskippable and long, you can tap the (i) in bottom left hand corner, then "stop seeing this ad", then choose irrelevant, repetitive, or inappropriate (all work). It will end immediately and earlier than if you had watched the entire ad.'},
     {'is_video': False,
      'num_comments': 32,
      'num_crossposts': 0,
      'subreddit': 'democrats',
      'time_elasped': 40128.51550197601,
      'title': 'Trump is blaming Democrats for separating migrant families at the border. That is a LIE. He and his administration are enforcing his OWN policy.'},
     {'is_video': False,
      'num_comments': 22,
      'num_crossposts': 0,
      'subreddit': 'Trumpgret',
      'time_elasped': 18143.515503168106,
      'title': 'Melania Trump Trumpgret? Twitter Changes To ‘New York’ Location, Missing FLOTUS Has Moved Back To City, New Rumor Claims'},
     {'is_video': False,
      'num_comments': 5,
      'num_crossposts': 0,
      'subreddit': 'mildlypenis',
      'time_elasped': 43599.515504837036,
      'title': 'Solar Powered Penis'},
     {'is_video': False,
      'num_comments': 54,
      'num_crossposts': 0,
      'subreddit': 'exjw',
      'time_elasped': 24487.515506982803,
      'title': 'Once upon a time, two fifteen year old kids met in high school and fell in love, only to have a religious cult split them apart. Six years later, the cult is gone and in some strange way, they ended up back together again :) Never give up guys. The universe works in mysterious ways.'},
     {'is_video': False,
      'num_comments': 20,
      'num_crossposts': 0,
      'subreddit': 'okbuddyretard',
      'time_elasped': 36347.51550889015,
      'title': 'it really be like that sometimes...'},
     {'is_video': False,
      'num_comments': 10,
      'num_crossposts': 0,
      'subreddit': 'Punny',
      'time_elasped': 40746.51551103592,
      'title': "Don't forget to wipe your beehive"},
     {'is_video': False,
      'num_comments': 14,
      'num_crossposts': 0,
      'subreddit': 'bois',
      'time_elasped': 33846.51551198959,
      'title': 'Bedhead and pussy'},
     {'is_video': False,
      'num_comments': 7,
      'num_crossposts': 0,
      'subreddit': 'ImaginaryMonsters',
      'time_elasped': 15961.51551413536,
      'title': 'It’s just a wendigo...but it wasn’t supposed to turn out cute...'},
     {'is_video': False,
      'num_comments': 29,
      'num_crossposts': 0,
      'subreddit': 'HadToHurt',
      'time_elasped': 48042.51551580429,
      'title': 'Hyper-Extended Knee'},
     {'is_video': False,
      'num_comments': 209,
      'num_crossposts': 0,
      'subreddit': 'bicycling',
      'time_elasped': 58475.515516996384,
      'title': 'My longest solo ride, ride through the night'},
     {'is_video': False,
      'num_comments': 8,
      'num_crossposts': 0,
      'subreddit': 'Konosuba',
      'time_elasped': 32883.51551890373,
      'title': 'Darkness About to Flex on You'},
     {'is_video': False,
      'num_comments': 4,
      'num_crossposts': 0,
      'subreddit': 'wholesomegreentext',
      'time_elasped': 29354.515520095825,
      'title': 'Anon sees his mom'},
     {'is_video': False,
      'num_comments': 3,
      'num_crossposts': 0,
      'subreddit': 'NamFlashbacks',
      'time_elasped': 27158.515522003174,
      'title': 'I had to. I’m sorry, John.'},
     {'is_video': False,
      'num_comments': 42,
      'num_crossposts': 0,
      'subreddit': 'CCW',
      'time_elasped': 23439.51552414894,
      'title': "Little Caesar's Employee uses CCW to save himself in a late night ambush"},
     {'is_video': False,
      'num_comments': 14,
      'num_crossposts': 0,
      'subreddit': 'KendrickLamar',
      'time_elasped': 36211.515525102615,
      'title': 'Fanart of my favorite scene from Humble!'},
     {'is_video': False,
      'num_comments': 9,
      'num_crossposts': 0,
      'subreddit': 'tiltshift',
      'time_elasped': 18045.515527009964,
      'title': 'York, England (aka the regular apple)'},
     {'is_video': False,
      'num_comments': 99,
      'num_crossposts': 0,
      'subreddit': 'MaliciousCompliance',
      'time_elasped': 61072.51552796364,
      'title': 'We knock off at 4 (long)'},
     {'is_video': False,
      'num_comments': 32,
      'num_crossposts': 0,
      'subreddit': 'iamverysmart',
      'time_elasped': 23820.515529870987,
      'title': 'Brains over looks😤'},
     {'is_video': False,
      'num_comments': 80,
      'num_crossposts': 0,
      'subreddit': 'CFB',
      'time_elasped': 37456.515532016754,
      'title': 'In Memoriam'},
     {'is_video': True,
      'num_comments': 23,
      'num_crossposts': 1,
      'subreddit': 'camping',
      'time_elasped': 47340.51553297043,
      'title': 'Nothing better then the early morning silence, campfire and of course coffee. Commence my summer baptism!'},
     {'is_video': False,
      'num_comments': 26,
      'num_crossposts': 0,
      'subreddit': 'Porsche',
      'time_elasped': 28581.515535116196,
      'title': 'Heard you guy might like this, a RUF Turbo R Limited that stopped in at work the other day. 1 of 7 produced, 620hp, US$600k.'},
     {'is_video': False,
      'num_comments': 10,
      'num_crossposts': 0,
      'subreddit': 'gardening',
      'time_elasped': 23138.515536785126,
      'title': 'My blueberry harvest was out of control - we had to split it three ways!'},
     {'is_video': False,
      'num_comments': 201,
      'num_crossposts': 4,
      'subreddit': 'Eyebleach',
      'time_elasped': 76209.51553797722,
      'title': 'Silly puppers is happy no matter what'},
     {'is_video': False,
      'num_comments': 16,
      'num_crossposts': 0,
      'subreddit': 'interestingasfuck',
      'time_elasped': 25386.515539884567,
      'title': 'rapid-fire cigar box juggling (sort of looks like he has 3 hands'},
     {'is_video': False,
      'num_comments': 20,
      'num_crossposts': 0,
      'subreddit': 'greentext',
      'time_elasped': 36386.515542030334,
      'title': 'Anon throws a party'},
     {'is_video': False,
      'num_comments': 5,
      'num_crossposts': 0,
      'subreddit': 'ImaginaryLandscapes',
      'time_elasped': 49339.51554298401,
      'title': 'New Bridge by Pablo Dominguez'},
     {'is_video': False,
      'num_comments': 37,
      'num_crossposts': 0,
      'subreddit': 'EarthPorn',
      'time_elasped': 47057.515545129776,
      'title': 'Oh Canada how you mesmerize me. Banff, Alberta [3213x2409][OC]'},
     {'is_video': False,
      'num_comments': 60,
      'num_crossposts': 0,
      'subreddit': 'Kappa',
      'time_elasped': 33166.515546798706,
      'title': 'Best part of combobreaker'},
     {'is_video': False,
      'num_comments': 93,
      'num_crossposts': 2,
      'subreddit': 'ofcoursethatsathing',
      'time_elasped': 72826.51554894447,
      'title': 'Pre-stained underwear for hiding valuables'},
     {'is_video': False,
      'num_comments': 10,
      'num_crossposts': 0,
      'subreddit': 'technicallythetruth',
      'time_elasped': 23499.51555109024,
      'title': "Well, he's right"},
     {'is_video': False,
      'num_comments': 193,
      'num_crossposts': 0,
      'subreddit': 'insanepeoplefacebook',
      'time_elasped': 66829.51555299759,
      'title': 'Deranged man uses Memorial Day to advertise cheap political trinkets'},
     {'is_video': False,
      'num_comments': 21,
      'num_crossposts': 0,
      'subreddit': 'Grimdank',
      'time_elasped': 31163.515553951263,
      'title': 'Even the Blood Axes have their principles'},
     {'is_video': False,
      'num_comments': 8,
      'num_crossposts': 0,
      'subreddit': 'tumblr',
      'time_elasped': 31133.515557050705,
      'title': 'Beautiful'},
     {'is_video': False,
      'num_comments': 3,
      'num_crossposts': 0,
      'subreddit': 'goldenretrievers',
      'time_elasped': 32707.515558958054,
      'title': "This is Cricket. Sometimes she can't handle the excitement of going to the beach."},
     {'is_video': False,
      'num_comments': 37,
      'num_crossposts': 0,
      'subreddit': 'sanfrancisco',
      'time_elasped': 27375.515560865402,
      'title': 'Please be good to Groot'},
     {'is_video': False,
      'num_comments': 14,
      'num_crossposts': 0,
      'subreddit': 'curlyhair',
      'time_elasped': 36702.515562057495,
      'title': 'Curly Rush, perfect depiction of my curl pattern consistency...none.'},
     {'is_video': False,
      'num_comments': 349,
      'num_crossposts': 2,
      'subreddit': 'PrequelMemes',
      'time_elasped': 50743.515563964844,
      'title': 'Is this okay to post here?'},
     {'is_video': False,
      'num_comments': 33,
      'num_crossposts': 0,
      'subreddit': 'softwaregore',
      'time_elasped': 32101.51556611061,
      'title': 'Raspberry the Movie'},
     {'is_video': False,
      'num_comments': 10,
      'num_crossposts': 0,
      'subreddit': 'ramen',
      'time_elasped': 51962.515567064285,
      'title': 'Google doesn’t know shit'},
     {'is_video': False,
      'num_comments': 23,
      'num_crossposts': 0,
      'subreddit': 'MrRobot',
      'time_elasped': 48603.515568971634,
      'title': 'How I spent one day in New York'},
     {'is_video': False,
      'num_comments': 13,
      'num_crossposts': 0,
      'subreddit': 'RebornDollCringe',
      'time_elasped': 28014.51556992531,
      'title': 'An accurate portrayal of me lying on the couch for hours looking for something to watch on Netflix'},
     {'is_video': False,
      'num_comments': 23,
      'num_crossposts': 0,
      'subreddit': 'BMW',
      'time_elasped': 35797.515572071075,
      'title': 'Snagged a ‘16 Azurite Black M3'},
     {'is_video': False,
      'num_comments': 27,
      'num_crossposts': 0,
      'subreddit': 'GunPorn',
      'time_elasped': 24806.51557302475,
      'title': '[OC] 7.5" Aero Precision AR-15 pistol I built myself.'},
     {'is_video': False,
      'num_comments': 21,
      'num_crossposts': 0,
      'subreddit': 'houston',
      'time_elasped': 29982.5155749321,
      'title': 'Texas sun loves Houston trees.'},
     {'is_video': False,
      'num_comments': 18,
      'num_crossposts': 0,
      'subreddit': 'itsaunixsystem',
      'time_elasped': 48307.515577077866,
      'title': '[Captured old commercial] Giant blue screens get the point across'},
     {'is_video': False,
      'num_comments': 9,
      'num_crossposts': 0,
      'subreddit': 'listentothis',
      'time_elasped': 40839.51557803154,
      'title': "20 Years of Age (ft. Yu Jein) - Let's Walk, I am Waiting In Front of your House [Korean acoustic pop] (2015)"},
     {'is_video': False,
      'num_comments': 19,
      'num_crossposts': 1,
      'subreddit': 'food',
      'time_elasped': 39529.51558089256,
      'title': '[I Ate] A Fully Loaded Crepe'},
     {'is_video': False,
      'num_comments': 8,
      'num_crossposts': 0,
      'subreddit': 'CrossStitch',
      'time_elasped': 20740.515581846237,
      'title': '[FO] Pinup cat'},
     {'is_video': False,
      'num_comments': 24,
      'num_crossposts': 0,
      'subreddit': 'CowChop',
      'time_elasped': 39798.515583992004,
      'title': 'It started as a CowChop hot sauce logo. The Japanese Kanji under the name says "chop sauce" &amp; the one near the knife says "cow". The milk rating system is the scale for how hot it is. Hope you guys like it! :)'},
     {'is_video': False,
      'num_comments': 23,
      'num_crossposts': 0,
      'subreddit': 'solotravel',
      'time_elasped': 20328.51558613777,
      'title': 'Making new friends in Bogotá, Colombia - my first solo travel experience!'},
     {'is_video': False,
      'num_comments': 39,
      'num_crossposts': 0,
      'subreddit': 'woooosh',
      'time_elasped': 44717.515587091446,
      'title': 'Found this on potato by darudesandstorm'},
     {'is_video': False,
      'num_comments': 193,
      'num_crossposts': 0,
      'subreddit': 'patientgamers',
      'time_elasped': 49130.51558923721,
      'title': 'Am I the only one that doesnt give half a shit about the absetergo story in Assassins Creed?'},
     {'is_video': False,
      'num_comments': 10,
      'num_crossposts': 0,
      'subreddit': 'BlueMidterm2018',
      'time_elasped': 23276.515590906143,
      'title': "California Dairies Inc. donates $20,000 to Devin Nunes (R-CA-22), showing it's willingness to abet conspiracy against the government in exchange for tax cuts and gutting the USPS."},
     {'is_video': False,
      'num_comments': 48,
      'num_crossposts': 0,
      'subreddit': 'gamegrumps',
      'time_elasped': 62682.51559185982,
      'title': 'Delightfully devilish, Arin... [OC] [From Guts and Glory - Part 12]'},
     {'is_video': True,
      'num_comments': 74,
      'num_crossposts': 0,
      'subreddit': 'Breath_of_the_Wild',
      'time_elasped': 61214.515594005585,
      'title': 'The best pirate in Hyrule'},
     {'is_video': False,
      'num_comments': 3,
      'num_crossposts': 0,
      'subreddit': 'AnimalsBeingJerks',
      'time_elasped': 17471.515595912933,
      'title': 'Making the most out of a bad situation'},
     {'is_video': False,
      'num_comments': 154,
      'num_crossposts': 0,
      'subreddit': 'CollegeBasketball',
      'time_elasped': 76660.5155980587,
      'title': "Don't let the Celtics blowing a 2-0 series lead distract from the fact that No. 16 UMBC beat No. 1 Virginia"},
     {'is_video': False,
      'num_comments': 95,
      'num_crossposts': 0,
      'subreddit': 'Bless',
      'time_elasped': 15272.515599012375,
      'title': 'So far, So good.'},
     {'is_video': False,
      'num_comments': 18,
      'num_crossposts': 0,
      'subreddit': 'HumansBeingBros',
      'time_elasped': 37556.51560091972,
      'title': 'This is Jake. At the Indy 500 Snake Pit (a concert in the middle of the track) on Sunday it was 92 degrees and Jake repeatedly would go out of his way to get dehydrated kids some cold water. He also did it with a smile on his face the whole time. Thanks Jake.'},
     {'is_video': False,
      'num_comments': 68,
      'num_crossposts': 0,
      'subreddit': 'APStudents',
      'time_elasped': 40237.5156018734,
      'title': 'Taking 38 AP courses next year, is that enough?'},
     {'is_video': False,
      'num_comments': 32,
      'num_crossposts': 0,
      'subreddit': 'rupaulsdragrace',
      'time_elasped': 25735.515604019165,
      'title': 'The perks of working at a movie theatre'},
     {'is_video': False,
      'num_comments': 72,
      'num_crossposts': 0,
      'subreddit': 'Memes_Of_The_Dank',
      'time_elasped': 52372.51560497284,
      'title': 'She got yeeted on'},
     {'is_video': False,
      'num_comments': 6,
      'num_crossposts': 0,
      'subreddit': 'WildernessBackpacking',
      'time_elasped': 28462.515607118607,
      'title': 'Upper Cataract Lake. Eagles Nest, CO'},
     {'is_video': False,
      'num_comments': 274,
      'num_crossposts': 0,
      'subreddit': 'programming',
      'time_elasped': 38953.515609025955,
      'title': 'Bjarne Stroustroup - Remeber the Vasa [critique of modern C++ direction]'},
     {'is_video': True,
      'num_comments': 56,
      'num_crossposts': 0,
      'subreddit': 'WWE',
      'time_elasped': 33439.515610933304,
      'title': 'Happy Birthday Seth Rollins!!'},
     {'is_video': False,
      'num_comments': 22,
      'num_crossposts': 0,
      'subreddit': 'dataisbeautiful',
      'time_elasped': 31328.515611886978,
      'title': 'Fourier transform of a square wave visualised [OC]'},
     {'is_video': False,
      'num_comments': 8,
      'num_crossposts': 1,
      'subreddit': 'wallpaper',
      'time_elasped': 48100.515614032745,
      'title': 'Your dream island. it will change you. [3840×2160]'},
     {'is_video': False,
      'num_comments': 124,
      'num_crossposts': 0,
      'subreddit': 'Justrolledintotheshop',
      'time_elasped': 33614.51561498642,
      'title': "96 Avalon with 900k. Going for a million miles on original motor. In for routine maintenance, thought I'd share"},
     {'is_video': False,
      'num_comments': 18,
      'num_crossposts': 0,
      'subreddit': 'billwurtzmemes',
      'time_elasped': 39493.51561713219,
      'title': 'I have a crush on a Mexican man but he looks really young and I was wondering'},
     {'is_video': False,
      'num_comments': 23,
      'num_crossposts': 0,
      'subreddit': 'DeepFriedMemes',
      'time_elasped': 37834.515619039536,
      'title': 'Who did this😂 😂 😂'},
     {'is_video': False,
      'num_comments': 1526,
      'num_crossposts': 2,
      'subreddit': 'gaming',
      'time_elasped': 39502.51561999321,
      'title': 'I overcame a gambling addiction by starting a PS4 fund. Anytime I felt like gambling, I would put some money into the fund. This is the result of just two weeks of not gambling.'},
     {'is_video': False,
      'num_comments': 16,
      'num_crossposts': 0,
      'subreddit': 'bulletjournal',
      'time_elasped': 29779.51562190056,
      'title': 'Week two of my over he garden wall themed June!'},
     {'is_video': False,
      'num_comments': 4,
      'num_crossposts': 0,
      'subreddit': 'cursedimages',
      'time_elasped': 28959.515625,
      'title': 'cursed_spa'},
     {'is_video': True,
      'num_comments': 106,
      'num_crossposts': 1,
      'subreddit': 'thanosdidnothingwrong',
      'time_elasped': 59212.515625953674,
      'title': 'So my friend sent this to me...'},
     {'is_video': False,
      'num_comments': 18,
      'num_crossposts': 0,
      'subreddit': 'showerbeer',
      'time_elasped': 18392.51562809944,
      'title': "It's hot af in Copenhagen. Cooldown after a run is now serious business"},
     {'is_video': False,
      'num_comments': 17,
      'num_crossposts': 0,
      'subreddit': 'disney',
      'time_elasped': 45126.51563000679,
      'title': 'My second fantasy pin. Inspired by Finding Dory.'},
     {'is_video': False,
      'num_comments': 32,
      'num_crossposts': 0,
      'subreddit': 'runescape',
      'time_elasped': 47303.51563119888,
      'title': 'three rules'},
     {'is_video': True,
      'num_comments': 17,
      'num_crossposts': 0,
      'subreddit': 'PeopleFuckingDying',
      'time_elasped': 32610.515632867813,
      'title': 'Man getS shReDdeD TO pIEcEs bY tRAIN'},
     {'is_video': False,
      'num_comments': 63,
      'num_crossposts': 0,
      'subreddit': 'justlegbeardthings',
      'time_elasped': 30763.51563501358,
      'title': 'Heterosexual relationships are socially acceptable prostitution'},
     {'is_video': False,
      'num_comments': 31,
      'num_crossposts': 0,
      'subreddit': 'Psychonaut',
      'time_elasped': 24675.51563692093,
      'title': 'Old hippy saved me when I took too much'},
     {'is_video': False,
      'num_comments': 48,
      'num_crossposts': 2,
      'subreddit': 'shittyreactiongifs',
      'time_elasped': 70405.51563811302,
      'title': "When I'm in the mood for sex but my girlfriend isn't"},
     {'is_video': False,
      'num_comments': 13,
      'num_crossposts': 0,
      'subreddit': 'barstoolsports',
      'time_elasped': 28582.51564002037,
      'title': 'We Are All Witnesses.'},
     {'is_video': False,
      'num_comments': 39,
      'num_crossposts': 0,
      'subreddit': 'Aquariums',
      'time_elasped': 31676.515642166138,
      'title': "He's or she's getting big. :)"},
     {'is_video': False,
      'num_comments': 35,
      'num_crossposts': 0,
      'subreddit': 'AgainstHateSubreddits',
      'time_elasped': 29319.515643835068,
      'title': 'Nimble navigator from the_donald proposes mass murder with poison gas in response to people on welfare existing.'},
     {'is_video': False,
      'num_comments': 9,
      'num_crossposts': 0,
      'subreddit': 'submechanophobia',
      'time_elasped': 13411.515645980835,
      'title': 'Bikes in a Napoleon-era Paris canal after its most recent quindecennial drainage'},
     {'is_video': False,
      'num_comments': 30,
      'num_crossposts': 0,
      'subreddit': 'catpictures',
      'time_elasped': 34340.51564693451,
      'title': 'found a stray kitty. gave him food but he really just want attention. more than happy to give him that.'},
     {'is_video': False,
      'num_comments': 14,
      'num_crossposts': 0,
      'subreddit': 'factorio',
      'time_elasped': 20396.515649080276,
      'title': 'Factorio diorama [Cinema4D] WIP'},
     {'is_video': True,
      'num_comments': 59,
      'num_crossposts': 0,
      'subreddit': '3Dprinting',
      'time_elasped': 48548.51565003395,
      'title': 'Iris Box Timelapse'},
     {'is_video': False,
      'num_comments': 46,
      'num_crossposts': 0,
      'subreddit': 'Liberal',
      'time_elasped': 35248.51565217972,
      'title': 'Black Defendants Get Longer Sentences From Republican-Appointed Judges, Study Finds'},
     {'is_video': True,
      'num_comments': 336,
      'num_crossposts': 7,
      'subreddit': 'aww',
      'time_elasped': 39628.51565384865,
      'title': 'Trying Coke for the first time'},
     {'is_video': False,
      'num_comments': 7,
      'num_crossposts': 0,
      'subreddit': 'Outdoors',
      'time_elasped': 19930.515655994415,
      'title': 'I had the path to myself this morning. Lake Tahoe, CA.'},
     {'is_video': False,
      'num_comments': 1,
      'num_crossposts': 0,
      'subreddit': 'MildlyVandalised',
      'time_elasped': 24777.51565694809,
      'title': 'Danger, go trespassing'},
     {'is_video': False,
      'num_comments': 27,
      'num_crossposts': 0,
      'subreddit': 'Kanye',
      'time_elasped': 32845.51565909386,
      'title': 'r/kanye in May starterpack'},
     {'is_video': False,
      'num_comments': 111,
      'num_crossposts': 0,
      'subreddit': 'shittyfoodporn',
      'time_elasped': 26614.51566004753,
      'title': "Boyfriend's burger, a snack bar in NL"},
     {'is_video': False,
      'num_comments': 58,
      'num_crossposts': 0,
      'subreddit': 'IsTodayFridayThe13th',
      'time_elasped': 32439.5156621933,
      'title': 'Is Today Friday the 13th?'},
     {'is_video': False,
      'num_comments': 28,
      'num_crossposts': 1,
      'subreddit': 'Marvel',
      'time_elasped': 34927.51566386223,
      'title': 'What am I?'},
     {'is_video': False,
      'num_comments': 6,
      'num_crossposts': 0,
      'subreddit': 'unexpectedhogwarts',
      'time_elasped': 25714.515664815903,
      'title': 'NSFW! Harry did you put your seed in the goblet?!'},
     {'is_video': False,
      'num_comments': 54,
      'num_crossposts': 0,
      'subreddit': 'dragonballfighterz',
      'time_elasped': 35809.51566696167,
      'title': 'Me on the 31st'},
     {'is_video': False,
      'num_comments': 24,
      'num_crossposts': 0,
      'subreddit': 'gaybrosgonemild',
      'time_elasped': 25829.515669107437,
      'title': 'Shaved off my beard, now I look like a gq model'},
     {'is_video': False,
      'num_comments': 12,
      'num_crossposts': 0,
      'subreddit': 'starbucks',
      'time_elasped': 28621.515671014786,
      'title': "When a customer calls to ask if we're open on Memorial Day:"},
     {'is_video': False,
      'num_comments': 17,
      'num_crossposts': 0,
      'subreddit': 'Miniworlds',
      'time_elasped': 41194.51567196846,
      'title': 'Mini cliff'},
     {'is_video': False,
      'num_comments': 168,
      'num_crossposts': 0,
      'subreddit': 'videos',
      'time_elasped': 29086.515674114227,
      'title': '$1.3Million Paving Machine sits idle in a Los Angeles lot while a homeless man uses it for shelter.'},
     {'is_video': False,
      'num_comments': 11,
      'num_crossposts': 0,
      'subreddit': 'hmmm',
      'time_elasped': 33750.515676021576,
      'title': 'hmmm'},
     {'is_video': False,
      'num_comments': 10,
      'num_crossposts': 0,
      'subreddit': 'GalGadot',
      'time_elasped': 29697.515677928925,
      'title': 'Tank Top - GIF'},
     {'is_video': False,
      'num_comments': 143,
      'num_crossposts': 3,
      'subreddit': 'natureismetal',
      'time_elasped': 73006.51568007469,
      'title': 'Eagle bones (West coast British Columbia)'},
     {'is_video': False,
      'num_comments': 49,
      'num_crossposts': 0,
      'subreddit': 'ArtisanVideos',
      'time_elasped': 35126.51568198204,
      'title': 'A passionate man mows his lawn to perfection'},
     {'is_video': False,
      'num_comments': 19,
      'num_crossposts': 0,
      'subreddit': 'golf',
      'time_elasped': 36369.51568412781,
      'title': 'I just got a hole in one'},
     {'is_video': False,
      'num_comments': 7,
      'num_crossposts': 0,
      'subreddit': 'pic',
      'time_elasped': 22583.515685081482,
      'title': 'Frogs piggybacking on a caiman'},
     {'is_video': False,
      'num_comments': 303,
      'num_crossposts': 4,
      'subreddit': 'Damnthatsinteresting',
      'time_elasped': 63937.51568698883,
      'title': 'The 008004 size capacitor on a fingertip'},
     {'is_video': False,
      'num_comments': 26,
      'num_crossposts': 1,
      'subreddit': 'copypasta',
      'time_elasped': 29847.51568889618,
      'title': 'All Star translated through every language in google translate'},
     {'is_video': False,
      'num_comments': 42,
      'num_crossposts': 0,
      'subreddit': 'science',
      'time_elasped': 25549.515690088272,
      'title': 'A new DNA study found that nine out of 10 species on Earth today, including humans, came into being 100,000 to 200,000 years ago. In analysing DNA barcodes across 100,000 species, researchers found a telltale sign showing that almost all the animals emerged about the same time as humans.'},
     {'is_video': False,
      'num_comments': 196,
      'num_crossposts': 5,
      'subreddit': 'surrealmemes',
      'time_elasped': 73969.51569199562,
      'title': 'https://resist.jpg'},
     {'is_video': False,
      'num_comments': 12,
      'num_crossposts': 0,
      'subreddit': 'HumanPorn',
      'time_elasped': 26625.515694141388,
      'title': 'Father Camille Folliet, a French Roman Catholic priest, lends his support and advises the French Resistance behind a barricade during the Battle for Paris [1168 x 1281]'},
     {'is_video': False,
      'num_comments': 968,
      'num_crossposts': 0,
      'subreddit': 'marvelstudios',
      'time_elasped': 66174.51569604874,
      'title': 'This scene set the tone immediately for Ragnarok and Thor’s character. I thought it was an amazing intro to the new Thor that they’re building.'},
     {'is_video': False,
      'num_comments': 2,
      'num_crossposts': 0,
      'subreddit': 'WackyTicTacs',
      'time_elasped': 29817.51569700241,
      'title': 'LETS DO IT KAREN'},
     {'is_video': False,
      'num_comments': 227,
      'num_crossposts': 1,
      'subreddit': 'confession',
      'time_elasped': 74160.51569890976,
      'title': 'I’m still alive because I just want my dog to have a good life'},
     {'is_video': False,
      'num_comments': 2,
      'num_crossposts': 0,
      'subreddit': 'Pareidolia',
      'time_elasped': 22482.515699863434,
      'title': 'I am Bag-Groot'},
     {'is_video': False,
      'num_comments': 66,
      'num_crossposts': 0,
      'subreddit': 'todayilearned',
      'time_elasped': 34972.5157020092,
      'title': 'TIL that after surviving a car accident and receiving a settlement, Paul Dennis Reid used the money to get a plastic surgery, and then pursued a career as a country singer, under the stage name Justin Parks. When that failed, he became a serial killer.'},
     {'is_video': False,
      'num_comments': 28,
      'num_crossposts': 0,
      'subreddit': 'radiohead',
      'time_elasped': 34792.51570415497,
      'title': 'Radiohead Re-Releases ‘Kid A’ With Remastered Original Skits'},
     {'is_video': False,
      'num_comments': 106,
      'num_crossposts': 2,
      'subreddit': 'photoshopbattles',
      'time_elasped': 76663.51570510864,
      'title': 'PsBattle: Thumbs up cat'},
     {'is_video': False,
      'num_comments': 27,
      'num_crossposts': 3,
      'subreddit': 'gifsthatendtoosoon',
      'time_elasped': 38944.51570677757,
      'title': 'Wind resistance.'},
     {'is_video': False,
      'num_comments': 19,
      'num_crossposts': 0,
      'subreddit': 'philadelphia',
      'time_elasped': 34948.515707969666,
      'title': 'Wawa goose spotted on Fishtown bathroom wall.'},
     {'is_video': False,
      'num_comments': 5,
      'num_crossposts': 0,
      'subreddit': 'foxes',
      'time_elasped': 35106.515709877014,
      'title': 'Woke up and looked out the window this morning to see these little guys'},
     {'is_video': False,
      'num_comments': 19,
      'num_crossposts': 0,
      'subreddit': 'parrots',
      'time_elasped': 33784.51571202278,
      'title': "My boring gray cockatiel. Today she finally made kissy noises instead of just screaming :')"},
     {'is_video': False,
      'num_comments': 33,
      'num_crossposts': 0,
      'subreddit': 'psychology',
      'time_elasped': 39624.51571393013,
      'title': 'Being torn about which personal goals to pursue is associated with symptoms of psychological distress, suggests new research based on more than 200 young adults, which found that goal conflict and ambivalence were independently associated with anxious and depressive symptoms.'},
     {'is_video': False,
      'num_comments': 147,
      'num_crossposts': 1,
      'subreddit': 'teslamotors',
      'time_elasped': 37430.51571512222,
      'title': 'When getting out of your model 3. Don’t push on the plastic right by the seat.'},
     {'is_video': False,
      'num_comments': 23,
      'num_crossposts': 0,
      'subreddit': 'worldbuilding',
      'time_elasped': 51348.51571702957,
      'title': 'Aunt Maddie - Entanglement with a Carnivorous Fungi'},
     {'is_video': True,
      'num_comments': 464,
      'num_crossposts': 6,
      'subreddit': 'gifs',
      'time_elasped': 40931.51571893692,
      'title': 'Moose Attacks a Lawnmower'},
     {'is_video': False,
      'num_comments': 82,
      'num_crossposts': 0,
      'subreddit': 'warriors',
      'time_elasped': 29475.51572084427,
      'title': '#WARRIORS ARE #BLESSED FOR TODAYS #NBA GAME !!! - Lil B'},
     {'is_video': False,
      'num_comments': 23,
      'num_crossposts': 0,
      'subreddit': 'disneyvacation',
      'time_elasped': 42704.515722990036,
      'title': 'How to bully your dog appropriately'},
     {'is_video': False,
      'num_comments': 12,
      'num_crossposts': 0,
      'subreddit': 'fantanoforever',
      'time_elasped': 31131.515725135803,
      'title': 'When you feel a 10'},
     {'is_video': False,
      'num_comments': 11,
      'num_crossposts': 0,
      'subreddit': 'AntiJokes',
      'time_elasped': 35067.51572608948,
      'title': 'A horse walks into a bar.'},
     {'is_video': False,
      'num_comments': 10,
      'num_crossposts': 0,
      'subreddit': 'WholesomeComics',
      'time_elasped': 28086.515727758408,
      'title': 'Mini comic: family'},
     {'is_video': False,
      'num_comments': 5,
      'num_crossposts': 0,
      'subreddit': 'ThingsCutInHalfPorn',
      'time_elasped': 27932.515729904175,
      'title': 'Cross Section of a Pratt and Whitney R2800 Double Wasp [3264 x 2448]'},
     {'is_video': False,
      'num_comments': 23,
      'num_crossposts': 0,
      'subreddit': 'relationship_advice',
      'time_elasped': 20591.51573085785,
      'title': 'Found my girlfriends suicide notes'},
     {'is_video': False,
      'num_comments': 29,
      'num_crossposts': 0,
      'subreddit': 'raimimemes',
      'time_elasped': 60001.515733003616,
      'title': 'When you shake hands with someone who says Homecoming is the best Spiderman film'},
     {'is_video': False,
      'num_comments': 14,
      'num_crossposts': 0,
      'subreddit': 'backpacking',
      'time_elasped': 35717.51573514938,
      'title': 'Found this hidden gem in Dominguez Canyon, CO.'},
     {'is_video': False,
      'num_comments': 26,
      'num_crossposts': 1,
      'subreddit': 'youtubehaiku',
      'time_elasped': 20277.515737056732,
      'title': '[Poetry] [NSFW] A day to remember'},
     {'is_video': False,
      'num_comments': 6,
      'num_crossposts': 0,
      'subreddit': 'japanpics',
      'time_elasped': 34120.51573801041,
      'title': '[OC] Akihabara on a Rainy Night'},
     {'is_video': False,
      'num_comments': 2,
      'num_crossposts': 0,
      'subreddit': 'marvelmemes',
      'time_elasped': 41107.515739917755,
      'title': 'Oh.'},
     {'is_video': False,
      'num_comments': 12,
      'num_crossposts': 0,
      'subreddit': 'ExpectationVsReality',
      'time_elasped': 34755.51574206352,
      'title': 'Almost got it'},
     {'is_video': False,
      'num_comments': 146,
      'num_crossposts': 0,
      'subreddit': 'unpopularopinion',
      'time_elasped': 43786.5157430172,
      'title': "I think it's weird to name your son after yourself"},
     {'is_video': False,
      'num_comments': 25,
      'num_crossposts': 0,
      'subreddit': 'anthologymemes',
      'time_elasped': 26732.515744924545,
      'title': "When you watch Solo with someone who hasn't seen Clone Wars"},
     {'is_video': False,
      'num_comments': 14,
      'num_crossposts': 0,
      'subreddit': 'hamiltonmusical',
      'time_elasped': 23979.51574611664,
      'title': 'It may be the nosebleed, but after two years of savin’ I’m past patiently waitin’. I made it to the room where it happens.'},
     {'is_video': False,
      'num_comments': 44,
      'num_crossposts': 0,
      'subreddit': 'TheRedPill',
      'time_elasped': 21880.515748023987,
      'title': 'Deleted Instagram; Banged Instagram Model'},
     {'is_video': False,
      'num_comments': 47,
      'num_crossposts': 0,
      'subreddit': 'starterpacks',
      'time_elasped': 32377.515749931335,
      'title': 'r/deepfriedmemes Starter Pack'},
     {'is_video': False,
      'num_comments': 22,
      'num_crossposts': 0,
      'subreddit': 'LivestreamFail',
      'time_elasped': 17444.51575088501,
      'title': 'Daddy Bjorn teaching Hampton Brandon a lesson'},
     {'is_video': True,
      'num_comments': 58,
      'num_crossposts': 1,
      'subreddit': 'robotics',
      'time_elasped': 35951.51575303078,
      'title': "1st public demo of Wandercraft's autonomous exoskeleton to help disabled people walk again with France president Macron. We're hiring."},
     {'is_video': False,
      'num_comments': 6,
      'num_crossposts': 0,
      'subreddit': 'Moonmoon',
      'time_elasped': 34432.515754938126,
      'title': "Watcha drinkin'?"},
     {'is_video': False,
      'num_comments': 230,
      'num_crossposts': 0,
      'subreddit': 'cringe',
      'time_elasped': 57609.51575708389,
      'title': 'Picking up girls at bars 101: sniff their drinks'},
     {'is_video': False,
      'num_comments': 7,
      'num_crossposts': 0,
      'subreddit': 'howyoudoin',
      'time_elasped': 26289.51575899124,
      'title': 'MRW after all this time I learn that palaeontology isn’t the study of dinosaurs, but of fossils in general'},
     {'is_video': False,
      'num_comments': 43,
      'num_crossposts': 0,
      'subreddit': 'Izlam',
      'time_elasped': 42919.515760183334,
      'title': 'The struggle on reddit is real...'},
     {'is_video': False,
      'num_comments': 23,
      'num_crossposts': 0,
      'subreddit': 'awfuleyebrows',
      'time_elasped': 40083.515761852264,
      'title': "I'm not sure why she does her eyebrows like this :("},
     {'is_video': False,
      'num_comments': 160,
      'num_crossposts': 0,
      'subreddit': 'sysadmin',
      'time_elasped': 77417.51576399803,
      'title': 'Failure is always an option'},
     {'is_video': False,
      'num_comments': 124,
      'num_crossposts': 0,
      'subreddit': 'MovieDetails',
      'time_elasped': 65422.51576590538,
      'title': 'In Aladdin (1992) Genie is NOT the only character with 4 fingers, despite what the recent post said.'},
     {'is_video': False,
      'num_comments': 35,
      'num_crossposts': 0,
      'subreddit': 'absolutelynotme_irl',
      'time_elasped': 83056.51576709747,
      'title': 'absolutelynotme_irl'},
     {'is_video': False,
      'num_comments': 11,
      'num_crossposts': 3,
      'subreddit': 'LateStageCapitalism',
      'time_elasped': 39135.51576900482,
      'title': "'Murica"},
     {'is_video': False,
      'num_comments': 29,
      'num_crossposts': 0,
      'subreddit': 'apple',
      'time_elasped': 20972.515770196915,
      'title': '2019 iPhone may bring stereoscopic vision &amp; 3X optical zoom via 12MP triple-lens rear camera'},
     {'is_video': False,
      'num_comments': 21,
      'num_crossposts': 0,
      'subreddit': 'Torontobluejays',
      'time_elasped': 28509.515771865845,
      'title': 'How Gibby decides where Martin will play...'},
     {'is_video': False,
      'num_comments': 9,
      'num_crossposts': 0,
      'subreddit': 'masterforgiveme',
      'time_elasped': 24013.515774011612,
      'title': 'He’s not clowning around'},
     {'is_video': False,
      'num_comments': 34,
      'num_crossposts': 0,
      'subreddit': 'supremeclothing',
      'time_elasped': 34196.51577591896,
      'title': 'Just pickled cucumbers in Supreme jars'},
     {'is_video': False,
      'num_comments': 35,
      'num_crossposts': 0,
      'subreddit': 'blackcats',
      'time_elasped': 75394.51577711105,
      'title': 'I caught James in mid meow!'},
     {'is_video': False,
      'num_comments': 27,
      'num_crossposts': 0,
      'subreddit': 'blessedimages',
      'time_elasped': 34490.5157790184,
      'title': 'blessed_date'},
     {'is_video': False,
      'num_comments': 50,
      'num_crossposts': 1,
      'subreddit': 'ImaginaryTechnology',
      'time_elasped': 49078.51578116417,
      'title': 'untitled by Eddie Del Rio'},
     {'is_video': False,
      'num_comments': 16,
      'num_crossposts': 0,
      'subreddit': 'theocho',
      'time_elasped': 27294.5157828331,
      'title': 'Adorably unique'},
     {'is_video': False,
      'num_comments': 10,
      'num_crossposts': 0,
      'subreddit': 'KnightsOfPineapple',
      'time_elasped': 23839.515784025192,
      'title': "Happy Memorial Day! Grillin' the gold. Please remember our lost veterans."},
     {'is_video': False,
      'num_comments': 58,
      'num_crossposts': 0,
      'subreddit': 'CanadaPolitics',
      'time_elasped': 26349.51578593254,
      'title': 'PARKIN: Almost everywhere, conservatives have become the party of fiscal recklessness'},
     {'is_video': False,
      'num_comments': 18,
      'num_crossposts': 0,
      'subreddit': 'survivor',
      'time_elasped': 30730.515788078308,
      'title': 'All I see amidst the memes'},
     {'is_video': False,
      'num_comments': 76,
      'num_crossposts': 0,
      'subreddit': 'CringeAnarchy',
      'time_elasped': 31063.515789985657,
      'title': 'Race hustler btfo'},
     {'is_video': False,
      'num_comments': 149,
      'num_crossposts': 3,
      'subreddit': 'nononono',
      'time_elasped': 66358.515791893,
      'title': 'This Must Be The Passing Lane.'},
     {'is_video': False,
      'num_comments': 23,
      'num_crossposts': 3,
      'subreddit': 'BeAmazed',
      'time_elasped': 27274.51579284668,
      'title': 'Rapid fire cigar box juggling'},
     {'is_video': False,
      'num_comments': 18,
      'num_crossposts': 0,
      'subreddit': 'lifeisstrange',
      'time_elasped': 42190.51579499245,
      'title': '[NO SPOILERS] Chloe cosplay by Ariderion'},
     {'is_video': False,
      'num_comments': 30,
      'num_crossposts': 0,
      'subreddit': 'FrankOcean',
      'time_elasped': 48710.515796899796,
      'title': 'HOW MANY PEOPLE LISTEN TO THIS?'},
     {'is_video': True,
      'num_comments': 33,
      'num_crossposts': 0,
      'subreddit': 'scriptedasiangifs',
      'time_elasped': 29725.51579809189,
      'title': 'Stalker'},
     {'is_video': False,
      'num_comments': 14,
      'num_crossposts': 0,
      'subreddit': 'eatsandwiches',
      'time_elasped': 26028.515799999237,
      'title': 'Steak sandwich'},
     {'is_video': True,
      'num_comments': 69,
      'num_crossposts': 1,
      'subreddit': 'instant_regret',
      'time_elasped': 43160.51580119133,
      'title': 'Just taking a selfie'},
     {'is_video': False,
      'num_comments': 2,
      'num_crossposts': 0,
      'subreddit': 'cutelittlefangs',
      'time_elasped': 16157.51580286026,
      'title': '*Angry wolf noises* [Shingeki no bahamut/Granblue fantasy]'},
     {'is_video': False,
      'num_comments': 4,
      'num_crossposts': 0,
      'subreddit': 'urbanexploration',
      'time_elasped': 23786.515805006027,
      'title': 'Abandoned Church after an Arson Attack'},
     {'is_video': False,
      'num_comments': 5,
      'num_crossposts': 0,
      'subreddit': 'The_Congress',
      'time_elasped': 29476.515806913376,
      'title': 'Obama administration was corrupt, more and more evidence reveals daily'},
     {'is_video': False,
      'num_comments': 24,
      'num_crossposts': 0,
      'subreddit': 'awfuleverything',
      'time_elasped': 34028.51580905914,
      'title': 'This look'},
     {'is_video': False,
      'num_comments': 2248,
      'num_crossposts': 0,
      'subreddit': 'AskReddit',
      'time_elasped': 34690.51581096649,
      'title': 'What names are often tied to a specific personality?'},
     {'is_video': False,
      'num_comments': 24,
      'num_crossposts': 0,
      'subreddit': 'Gundam',
      'time_elasped': 28876.51581311226,
      'title': 'My friends tell me I might have a problem'},
     {'is_video': False,
      'num_comments': 58,
      'num_crossposts': 0,
      'subreddit': 'MealPrepSunday',
      'time_elasped': 32547.515815019608,
      'title': 'Gotta start somewhere'},
     {'is_video': False,
      'num_comments': 19,
      'num_crossposts': 0,
      'subreddit': 'keming',
      'time_elasped': 43779.51581597328,
      'title': 'Life is beautif ul.'},
     {'is_video': False,
      'num_comments': 26,
      'num_crossposts': 1,
      'subreddit': 'reallifedoodles',
      'time_elasped': 79742.51581788063,
      'title': 'Slap'},
     {'is_video': False,
      'num_comments': 29,
      'num_crossposts': 0,
      'subreddit': 'northernlion',
      'time_elasped': 38619.5158200264,
      'title': "In episode 680 of the binding of isaac afterbirth plsu NL says the the battle of hasting's happened in 1055 which is incorrect."},
     {'is_video': False,
      'num_comments': 82,
      'num_crossposts': 0,
      'subreddit': 'TrueOffMyChest',
      'time_elasped': 38737.515822172165,
      'title': 'My 2 year old son is a psychopath'},
     {'is_video': False,
      'num_comments': 21,
      'num_crossposts': 0,
      'subreddit': 'GolfGTI',
      'time_elasped': 31756.51582312584,
      'title': 'She’s an old girl but she’s in good shape.'},
     {'is_video': False,
      'num_comments': 174,
      'num_crossposts': 0,
      'subreddit': 'movies',
      'time_elasped': 27296.51582479477,
      'title': "Woody Harrelson Confirms Casting in 'Venom' &amp; Its Sequel"},
     {'is_video': False,
      'num_comments': 33,
      'num_crossposts': 0,
      'subreddit': 'Frugal_Jerk',
      'time_elasped': 34197.51582694054,
      'title': 'All for free! Livin like a king!'},
     {'is_video': True,
      'num_comments': 9,
      'num_crossposts': 0,
      'subreddit': 'whatcouldgoright',
      'time_elasped': 16582.51582789421,
      'title': '4 yr old dangling on side of 4-story building, WCGR?'},
     {'is_video': False,
      'num_comments': 0,
      'num_crossposts': 0,
      'subreddit': 'blop',
      'time_elasped': 27953.515830039978,
      'title': "Jim's 'I love sticks and they make me so happy' blop"},
     {'is_video': False,
      'num_comments': 11,
      'num_crossposts': 0,
      'subreddit': 'OreGairuSNAFU',
      'time_elasped': 18021.515831947327,
      'title': 'I drew Yukino again.'},
     {'is_video': False,
      'num_comments': 16,
      'num_crossposts': 0,
      'subreddit': 'dataisugly',
      'time_elasped': 19077.51583313942,
      'title': 'The marvel that is 3D Stacked Scatter Pie Columns.'},
     {'is_video': False,
      'num_comments': 44,
      'num_crossposts': 1,
      'subreddit': 'tuckedinkitties',
      'time_elasped': 88593.51583480835,
      'title': "Dreaming of all the ankles she's going to attack when she wakes up."},
     {'is_video': False,
      'num_comments': 7,
      'num_crossposts': 0,
      'subreddit': 'ShitCosmoSays',
      'time_elasped': 17845.515836954117,
      'title': '“Real clothes”'},
     {'is_video': False,
      'num_comments': 37,
      'num_crossposts': 1,
      'subreddit': 'italy',
      'time_elasped': 24972.51583790779,
      'title': "Strade romane d'Italia (grafico moderno)"},
     {'is_video': False,
      'num_comments': 9,
      'num_crossposts': 0,
      'subreddit': 'Xenoblade_Chronicles',
      'time_elasped': 16023.515840053558,
      'title': 'Poppi QT Pi cosplay progress'},
     {'is_video': False,
      'num_comments': 16,
      'num_crossposts': 0,
      'subreddit': 'Catswithjobs',
      'time_elasped': 55340.51584196091,
      'title': 'Hardware store kitty greets me every time I go in'},
     {'is_video': False,
      'num_comments': 60,
      'num_crossposts': 0,
      'subreddit': 'lego',
      'time_elasped': 42583.515843153,
      'title': 'Penny, my amazing assistant, passed away unexpectedly this week. She loved to "help me" build. Let your cats help you, because you may never know when they may not around to lend a paw.'},
     {'is_video': False,
      'num_comments': 1,
      'num_crossposts': 0,
      'subreddit': 'ProperAnimalNames',
      'time_elasped': 20244.51584506035,
      'title': 'Seafluff'},
     {'is_video': False,
      'num_comments': 9,
      'num_crossposts': 0,
      'subreddit': 'OldManDog',
      'time_elasped': 38715.5158469677,
      'title': 'Still away at school for 3 weeks, and missing my girl a little extra today. Here is the lovely Holly [15]'},
     {'is_video': False,
      'num_comments': 109,
      'num_crossposts': 0,
      'subreddit': 'trashy',
      'time_elasped': 56082.515848875046,
      'title': 'Is anyone else tired of the “literally trashy” posts of littering and such? Can we all agree that trash on the floor/sidewalk is trashy and not post it here? It’s neither interesting nor amusing.'},
     {'is_video': False,
      'num_comments': 22,
      'num_crossposts': 0,
      'subreddit': 'MonsterHunter',
      'time_elasped': 37899.51585102081,
      'title': 'Finished the downvote button pendant + some other things'},
     {'is_video': False,
      'num_comments': 63,
      'num_crossposts': 0,
      'subreddit': 'oculus',
      'time_elasped': 25670.51585316658,
      'title': 'Beat Saber sold 100,000 copies in less than a month'},
     {'is_video': False,
      'num_comments': 4,
      'num_crossposts': 1,
      'subreddit': 'DeathProTips',
      'time_elasped': 33847.515854120255,
      'title': 'Tired of your child draining your funds?'},
     {'is_video': False,
      'num_comments': 84,
      'num_crossposts': 0,
      'subreddit': 'ontario',
      'time_elasped': 20095.515855789185,
      'title': 'Ipos (May 25-27): PC 37 / NDP 34 / LIB 22'},
     {'is_video': False,
      'num_comments': 1,
      'num_crossposts': 0,
      'subreddit': 'LatinoPeopleTwitter',
      'time_elasped': 39742.51585793495,
      'title': 'La mañana después de llevarla a bailar'},
     {'is_video': False,
      'num_comments': 16,
      'num_crossposts': 0,
      'subreddit': 'meirl',
      'time_elasped': 37724.51586008072,
      'title': 'me irl'},
     {'is_video': False,
      'num_comments': 23,
      'num_crossposts': 0,
      'subreddit': 'Gunpla',
      'time_elasped': 23128.515861034393,
      'title': 'GM Sniper II Ver. Thunderbolt (album in comments)'},
     {'is_video': False,
      'num_comments': 93,
      'num_crossposts': 0,
      'subreddit': 'blunderyears',
      'time_elasped': 60900.51586294174,
      'title': 'Me looking fly as hell before church camp. Age 12'},
     {'is_video': False,
      'num_comments': 9,
      'num_crossposts': 0,
      'subreddit': 'virginvschad',
      'time_elasped': 23623.515864133835,
      'title': 'Virgin Ren VS Chad Maul'},
     {'is_video': False,
      'num_comments': 3,
      'num_crossposts': 0,
      'subreddit': 'Wholesome4chan',
      'time_elasped': 38474.51586604118,
      'title': 'Anon makes a face [x-post from r/Greentext]'},
     {'is_video': False,
      'num_comments': 68,
      'num_crossposts': 0,
      'subreddit': 'povertyfinance',
      'time_elasped': 18456.515867948532,
      'title': "I mentioned not being able to afford tick stuff for my dogs and got a lot of flack for it and didn't even think to properly explain my situation."},
     {'is_video': False,
      'num_comments': 3,
      'num_crossposts': 0,
      'subreddit': 'ProtectAndServe',
      'time_elasped': 45643.51586890221,
      'title': '[MEME] L A Z Y B O I'},
     {'is_video': False,
      'num_comments': 9,
      'num_crossposts': 0,
      'subreddit': 'SupermodelCats',
      'time_elasped': 45558.51587104797,
      'title': 'Cat the Maine Coon by the name of Vavilon'},
     {'is_video': False,
      'num_comments': 15,
      'num_crossposts': 0,
      'subreddit': 'Economics',
      'time_elasped': 35962.51587295532,
      'title': 'Wall Street regulations need a facelift, not a minor Dodd-Frank makeover'},
     {'is_video': False,
      'num_comments': 621,
      'num_crossposts': 0,
      'subreddit': 'TokyoGhoul',
      'time_elasped': 36735.51587510109,
      'title': 'Tokyo Ghoul:re Chapter 174 - Links and Discussion'},
     {'is_video': False,
      'num_comments': 825,
      'num_crossposts': 0,
      'subreddit': 'canada',
      'time_elasped': 49345.51587700844,
      'title': 'The public funding of Catholic schools in Ontario is unstable and unprincipled: Opinion'},
     {'is_video': False,
      'num_comments': 23,
      'num_crossposts': 0,
      'subreddit': 'Gameboy',
      'time_elasped': 36274.51587796211,
      'title': 'After posting about an old pet photo on my GB camera, a kind Redditor loaned me his GB printer so I could print it out!'},
     {'is_video': False,
      'num_comments': 22,
      'num_crossposts': 0,
      'subreddit': 'OffensiveMemes',
      'time_elasped': 45702.51587986946,
      'title': 'An interesting title'},
     {'is_video': False,
      'num_comments': 27,
      'num_crossposts': 0,
      'subreddit': 'CatsAreAssholes',
      'time_elasped': 56052.51588201523,
      'title': 'Get cats they said, cats sleep all the time they said...'},
     {'is_video': False,
      'num_comments': 306,
      'num_crossposts': 0,
      'subreddit': 'dndnext',
      'time_elasped': 38609.51588392258,
      'title': 'Mike Mearls: "Think I found the last piece for psionics"'},
     {'is_video': False,
      'num_comments': 21,
      'num_crossposts': 0,
      'subreddit': '4x4',
      'time_elasped': 30217.51588511467,
      'title': 'The new rear suspension'},
     {'is_video': False,
      'num_comments': 24,
      'num_crossposts': 1,
      'subreddit': 'whitepeoplegifs',
      'time_elasped': 25730.51588702202,
      'title': 'Prom proposal'},
     {'is_video': False,
      'num_comments': 32,
      'num_crossposts': 0,
      'subreddit': 'retrogaming',
      'time_elasped': 22357.51588821411,
      'title': 'Just need Zelda and Tetris for the full set!'},
     {'is_video': True,
      'num_comments': 27,
      'num_crossposts': 0,
      'subreddit': 'PSVR',
      'time_elasped': 25162.51588988304,
      'title': 'Making dreams come true! Grandpa playing Driveclub VR with T80 racing wheel.'},
     {'is_video': False,
      'num_comments': 29,
      'num_crossposts': 0,
      'subreddit': '2meirl4meirl',
      'time_elasped': 59697.51589202881,
      'title': '2meirl4meirl'},
     {'is_video': False,
      'num_comments': 10,
      'num_crossposts': 0,
      'subreddit': 'LittleWitchAcademia',
      'time_elasped': 22328.515893936157,
      'title': 'Real life trio'},
     {'is_video': True,
      'num_comments': 7,
      'num_crossposts': 0,
      'subreddit': 'chemicalreactiongifs',
      'time_elasped': 12856.515896081924,
      'title': 'Smoke disappear instantly with alcohol and fire'},
     {'is_video': False,
      'num_comments': 15,
      'num_crossposts': 0,
      'subreddit': 'antiMLM',
      'time_elasped': 17916.5158970356,
      'title': 'Why do they target us?'},
     {'is_video': False,
      'num_comments': 8,
      'num_crossposts': 0,
      'subreddit': 'wholesomeprequelmemes',
      'time_elasped': 42045.51589894295,
      'title': 'When SOLO is much better than anticipated.'},
     {'is_video': False,
      'num_comments': 79,
      'num_crossposts': 1,
      'subreddit': 'onguardforthee',
      'time_elasped': 39161.515900850296,
      'title': 'Canada scores top rank in basic science knowledge.'},
     {'is_video': False,
      'num_comments': 2,
      'num_crossposts': 0,
      'subreddit': 'germanshepherds',
      'time_elasped': 32287.51590204239,
      'title': 'Worn out after a fun filled day at the dog park.'},
     {'is_video': False,
      'num_comments': 4,
      'num_crossposts': 0,
      'subreddit': 'buffy',
      'time_elasped': 14394.515903949738,
      'title': 'Newest cross-stitch, I think it turned out pretty well!'},
     {'is_video': False,
      'num_comments': 10,
      'num_crossposts': 0,
      'subreddit': 'Mcat',
      'time_elasped': 34731.51590514183,
      'title': 'Upvote for Good Luck'},
     {'is_video': False,
      'num_comments': 23,
      'num_crossposts': 0,
      'subreddit': 'fatestaynight',
      'time_elasped': 32254.51590704918,
      'title': 'Scáthach'},
     {'is_video': False,
      'num_comments': 11,
      'num_crossposts': 0,
      'subreddit': 'husky',
      'time_elasped': 40958.51590800285,
      'title': 'There are 10 huskies in this truck. All of them just stayed right there while people passed by to snap photos. Good dogs! :)'},
     {'is_video': False,
      'num_comments': 39,
      'num_crossposts': 2,
      'subreddit': 'HumansAreMetal',
      'time_elasped': 42116.5159099102,
      'title': 'Man with shovel rescues woman from pack of dogs'},
     {'is_video': False,
      'num_comments': 260,
      'num_crossposts': 0,
      'subreddit': 'AMA',
      'time_elasped': 27604.51591181755,
      'title': 'I was “kidnapped” overnight and taken to a wilderness camp. AMA'},
     {'is_video': False,
      'num_comments': 9,
      'num_crossposts': 0,
      'subreddit': 'goldenknights',
      'time_elasped': 19997.515913009644,
      'title': 'MGM Lion Dressed And Ready To Roar For Tonights Game. Pic Sharing From FB'},
     {'is_video': False,
      'num_comments': 9,
      'num_crossposts': 0,
      'subreddit': 'Anarchism',
      'time_elasped': 23181.515914916992,
      'title': 'Design for a poster I came up with recently'},
     {'is_video': False,
      'num_comments': 78,
      'num_crossposts': 0,
      'subreddit': 'VaporwaveAesthetics',
      'time_elasped': 62571.51591706276,
      'title': 'I never thought LEGO can be vaporwave, then I found out they released this series called "paradisa" in the 90\'s'},
     {'is_video': False,
      'num_comments': 9,
      'num_crossposts': 0,
      'subreddit': 'cirkeltrek',
      'time_elasped': 36463.515918016434,
      'title': 'Een interessante titel'},
     {'is_video': False,
      'num_comments': 7,
      'num_crossposts': 0,
      'subreddit': 'MildlyStartledCats',
      'time_elasped': 33143.51591992378,
      'title': 'My GF rescued this little guy from under a semi truck. He always has this mildly startled look on his face. We named him Lucky.'},
     {'is_video': False,
      'num_comments': 29,
      'num_crossposts': 0,
      'subreddit': 'coys',
      'time_elasped': 32001.51592183113,
      'title': '@TalkingTHFC: “The Telegraph understand Christian Eriksen is close to signing a new Tottenham Hotspur contract worth over £100,000-a-week plus bonuses. #COYS”'},
     {'is_video': False,
      'num_comments': 20,
      'num_crossposts': 0,
      'subreddit': 'educationalgifs',
      'time_elasped': 31886.515923023224,
      'title': 'Making a knife from Lignum Vitae wood'},
     {'is_video': False,
      'num_comments': 19,
      'num_crossposts': 0,
      'subreddit': 'forhonor',
      'time_elasped': 17839.515924930573,
      'title': "Warden's getting desperate"},
     {'is_video': False,
      'num_comments': 5,
      'num_crossposts': 0,
      'subreddit': 'learnart',
      'time_elasped': 33874.515926122665,
      'title': 'Went to my first art lesson and had a fun time attempting a Klimt reconstruction.'},
     {'is_video': False,
      'num_comments': 8,
      'num_crossposts': 0,
      'subreddit': 'GeometryIsNeat',
      'time_elasped': 28292.515928030014,
      'title': 'Geometric Knot I made'},
     {'is_video': False,
      'num_comments': 9,
      'num_crossposts': 0,
      'subreddit': 'shittybattlestations',
      'time_elasped': 27049.51593017578,
      'title': 'Found on r/gaming'},
     {'is_video': False,
      'num_comments': 652,
      'num_crossposts': 10,
      'subreddit': 'nevertellmetheodds',
      'time_elasped': 81428.51593184471,
      'title': 'you could not calculate the odds even if i wanted to know them'},
     {'is_video': False,
      'num_comments': 37,
      'num_crossposts': 0,
      'subreddit': 'TwoXChromosomes',
      'time_elasped': 15115.515933990479,
      'title': 'I made an app to help my girlfriend be safer and thought you guys might find it useful.'},
     {'is_video': False,
      'num_comments': 15,
      'num_crossposts': 0,
      'subreddit': 'Scotland',
      'time_elasped': 22037.515935897827,
      'title': 'Aye, pretty pleasant today...'},
     {'is_video': False,
      'num_comments': 2,
      'num_crossposts': 0,
      'subreddit': 'beards',
      'time_elasped': 19532.515938043594,
      'title': 'The Longer the Beard, The Better.'},
     {'is_video': False,
      'num_comments': 29,
      'num_crossposts': 0,
      'subreddit': 'JoeRogan',
      'time_elasped': 30569.51594018936,
      'title': 'Edgy Brah after one elk meat from Rogan'},
     {'is_video': False,
      'num_comments': 4,
      'num_crossposts': 0,
      'subreddit': 'dndmemes',
      'time_elasped': 18642.515941143036,
      'title': 'I deal 7 slashing, plus 86 radiant damage.'},
     {'is_video': False,
      'num_comments': 7,
      'num_crossposts': 1,
      'subreddit': 'ShokugekiNoSoma',
      'time_elasped': 31166.515942811966,
      'title': 'Meme Monday - Azameme'},
     {'is_video': False,
      'num_comments': 212,
      'num_crossposts': 1,
      'subreddit': 'FortNiteBR',
      'time_elasped': 41400.51594495773,
      'title': 'Assert dominance'},
     {'is_video': False,
      'num_comments': 22,
      'num_crossposts': 0,
      'subreddit': 'BlackPeopleTwitter',
      'time_elasped': 23189.5159471035,
      'title': "Y'all really just leveled up"},
     {'is_video': False,
      'num_comments': 69,
      'num_crossposts': 0,
      'subreddit': 'Braincels',
      'time_elasped': 28395.51594901085,
      'title': 'Just go to the gym bro'},
     {'is_video': False,
      'num_comments': 6,
      'num_crossposts': 0,
      'subreddit': 'brockhampton',
      'time_elasped': 25410.515949964523,
      'title': 'NEW BROCKHAMPTON MEMBER'},
     {'is_video': False,
      'num_comments': 2,
      'num_crossposts': 0,
      'subreddit': 'GlitchInTheMatrix',
      'time_elasped': 20536.51595211029,
      'title': 'Dogs master the art of the mannequin.'},
     {'is_video': False,
      'num_comments': 4,
      'num_crossposts': 0,
      'subreddit': 'Colorado',
      'time_elasped': 34372.51595401764,
      'title': '🦊 pups in Breck this weekend'},
     {'is_video': False,
      'num_comments': 14,
      'num_crossposts': 0,
      'subreddit': 'PrettyGirlsUglyFaces',
      'time_elasped': 25951.515955924988,
      'title': 'I didn’t even have to try to look like a space demon!'},
     {'is_video': False,
      'num_comments': 301,
      'num_crossposts': 2,
      'subreddit': 'CozyPlaces',
      'time_elasped': 81403.51595687866,
      'title': 'Hyder, Alaska looks pretty cozy'},
     {'is_video': False,
      'num_comments': 9,
      'num_crossposts': 0,
      'subreddit': 'castles',
      'time_elasped': 59050.51595902443,
      'title': 'The Royal Castle of Olite is a former castle of the kings of Navarre, situated in northern Spain. The silhouette of the castle, is not only imposing, it is also unique for its architectural chaos.'},
     {'is_video': True,
      'num_comments': 7,
      'num_crossposts': 0,
      'subreddit': 'gif',
      'time_elasped': 15722.515961170197,
      'title': 'Parisian Spiderman grabs a 4 yr old that was dangling on side of 4-story building.'},
     {'is_video': False,
      'num_comments': 15,
      'num_crossposts': 0,
      'subreddit': 'FireEmblemHeroes',
      'time_elasped': 25965.515962839127,
      'title': 'Best change to the sub since forever'},
     {'is_video': False,
      'num_comments': 6,
      'num_crossposts': 0,
      'subreddit': 'Breadit',
      'time_elasped': 19909.5159637928,
      'title': "Sandwich buns for tonight's cookout!"},
     {'is_video': False,
      'num_comments': 33,
      'num_crossposts': 0,
      'subreddit': 'TopMindsOfReddit',
      'time_elasped': 31588.515965938568,
      'title': '"We don\'t hate Jews in this subreddit," says Top Mind of /r/conspiracy who in the same comment accused the Jews of controlling the media, said they were behind 9/11, and endorsed the pseudoscientific and antisemitic book "The Culture of Critique."'},
     {'is_video': False,
      'num_comments': 4,
      'num_crossposts': 0,
      'subreddit': 'simpsonsshitposting',
      'time_elasped': 35587.51596689224,
      'title': '🎶 🎷'},
     {'is_video': False,
      'num_comments': 6,
      'num_crossposts': 0,
      'subreddit': 'ABoringDystopia',
      'time_elasped': 21857.51596903801,
      'title': 'Deranged man uses Memorial Day to advertise cheap political trinkets'},
     {'is_video': False,
      'num_comments': 39,
      'num_crossposts': 1,
      'subreddit': 'comedyhomicide',
      'time_elasped': 47111.51597118378,
      'title': 'Screaming 😂😂😂'},
     {'is_video': False,
      'num_comments': 55,
      'num_crossposts': 0,
      'subreddit': 'PKA',
      'time_elasped': 20895.51597213745,
      'title': "Wings' Surgery lies exposed: My story of being approved for a gastric sleeve procedure with Dr. Fernando Garcia. (warning, a bit long)"},
     {'is_video': False,
      'num_comments': 5,
      'num_crossposts': 0,
      'subreddit': 'GirlsMirin',
      'time_elasped': 27357.51597380638,
      'title': 'Tracy Caldwell Dyson mirin Earth from the ISS Cupola, 2010'},
     {'is_video': False,
      'num_comments': 16,
      'num_crossposts': 0,
      'subreddit': 'malefashion',
      'time_elasped': 21695.515974998474,
      'title': 'what I wore a while back'},
     {'is_video': False,
      'num_comments': 15,
      'num_crossposts': 0,
      'subreddit': 'Disneyland',
      'time_elasped': 27155.515976905823,
      'title': '..DCA is Magical'},
     {'is_video': False,
      'num_comments': 7,
      'num_crossposts': 0,
      'subreddit': 'whatintarnation',
      'time_elasped': 39830.51597905159,
      'title': 'Wot in illustrations??!?!?'},
     {'is_video': False,
      'num_comments': 56,
      'num_crossposts': 0,
      'subreddit': 'MapPorn',
      'time_elasped': 32747.51598095894,
      'title': 'Territories lost by China after the fall of the Qing dynasty'},
     {'is_video': False,
      'num_comments': 105,
      'num_crossposts': 2,
      'subreddit': 'Whatcouldgowrong',
      'time_elasped': 39820.51598381996,
      'title': 'WCGW asking a girl to prom on your motorbike'},
     {'is_video': False,
      'num_comments': 9,
      'num_crossposts': 0,
      'subreddit': 'wedding',
      'time_elasped': 17739.51598596573,
      'title': 'She said YES! My name is G... and she\'s R... So I did this "logo" to symbolize us: RG. What you guys think?'},
     {'is_video': False,
      'num_comments': 108,
      'num_crossposts': 0,
      'subreddit': 'Vive',
      'time_elasped': 43271.5159869194,
      'title': 'Budget Cuts Release Trailer'},
     {'is_video': False,
      'num_comments': 97,
      'num_crossposts': 0,
      'subreddit': 'MilitaryPorn',
      'time_elasped': 67465.51598906517,
      'title': "A U. S. Army paratrooper with the 82nd Airborne Division's 1st Brigade Combat Team fires his M4 carbine at insurgents during a firefight June 30, 2012, Ghazni province, Afghanistan [2100x1395]"},
     {'is_video': False,
      'num_comments': 5,
      'num_crossposts': 0,
      'subreddit': 'ArtFundamentals',
      'time_elasped': 15821.515990018845,
      'title': 'Tried to apply perspective thing in my drawings. Very happy how it turned out!'},
     {'is_video': False,
      'num_comments': 7,
      'num_crossposts': 0,
      'subreddit': 'redneckengineering',
      'time_elasped': 27816.51599216461,
      'title': 'High up windows? No problem.'},
     {'is_video': False,
      'num_comments': 35,
      'num_crossposts': 0,
      'subreddit': 'DuelLinks',
      'time_elasped': 31139.515993118286,
      'title': '[Meme] It think its final time we get this card too'},
     {'is_video': False,
      'num_comments': 10,
      'num_crossposts': 0,
      'subreddit': 'LoveLive',
      'time_elasped': 31821.515994787216,
      'title': "Kotori's Cuteness!"},
     {'is_video': False,
      'num_comments': 21,
      'num_crossposts': 0,
      'subreddit': 'graphic_design',
      'time_elasped': 20178.515996932983,
      'title': 'I found this poster while walking and I thought it had a really nice minimal design.'},
     {'is_video': False,
      'num_comments': 9,
      'num_crossposts': 0,
      'subreddit': 'Minneapolis',
      'time_elasped': 34154.51599907875,
      'title': 'Posted in NE on 4th street'},
     {'is_video': False,
      'num_comments': 63,
      'num_crossposts': 0,
      'subreddit': 'WWII',
      'time_elasped': 41190.516000032425,
      'title': 'Practically 7 months in and the spectating glitch is STILL a problem. Unacceptable SHG.'},
     {'is_video': False,
      'num_comments': 10,
      'num_crossposts': 0,
      'subreddit': 'britishproblems',
      'time_elasped': 21509.516001939774,
      'title': 'Unplugged my charger on the phone and started a huge chain of people shutting laptops, putting away books and standing up despite us being 5 minutes from the station'},
     {'is_video': False,
      'num_comments': 9,
      'num_crossposts': 0,
      'subreddit': 'Jaguars',
      'time_elasped': 25560.51600408554,
      'title': 'Spicing things up'},
     {'is_video': False,
      'num_comments': 110,
      'num_crossposts': 0,
      'subreddit': 'bestof',
      'time_elasped': 51561.51600599289,
      'title': '/u/pigmentosa critiques the representation of the Vietnam war in video game and American popular culture in general from a Vietnamese perspective'},
     {'is_video': False,
      'num_comments': 5,
      'num_crossposts': 0,
      'subreddit': 'MineralPorn',
      'time_elasped': 30459.516007900238,
      'title': "Largest fluorite octahedron I've ever seen."},
     {'is_video': False,
      'num_comments': 10,
      'num_crossposts': 0,
      'subreddit': 'LGBTeens',
      'time_elasped': 17353.516008853912,
      'title': "[picture] when your parents are planning a graduation party but they don't know you're gay so you give your friends cautions"},
     {'is_video': False,
      'num_comments': 21,
      'num_crossposts': 0,
      'subreddit': 'realmadrid',
      'time_elasped': 15596.51601099968,
      'title': "They're finally symmetric, again."},
     {'is_video': False,
      'num_comments': 17,
      'num_crossposts': 0,
      'subreddit': 'oddlyterrifying',
      'time_elasped': 34391.51601314545,
      'title': 'Demons. Has to be demons.'},
     {'is_video': False,
      'num_comments': 35,
      'num_crossposts': 0,
      'subreddit': 'de',
      'time_elasped': 19836.51601409912,
      'title': 'Prösterchen'},
     {'is_video': False,
      'num_comments': 76,
      'num_crossposts': 0,
      'subreddit': 'MMA',
      'time_elasped': 23690.51601624489,
      'title': 'Rory MacDonald is down to fight Gegard Moussasi.'},
     {'is_video': False,
      'num_comments': 7,
      'num_crossposts': 0,
      'subreddit': 'engrish',
      'time_elasped': 35164.51601791382,
      'title': 'Found on r/memes'},
     {'is_video': False,
      'num_comments': 6,
      'num_crossposts': 0,
      'subreddit': 'ich_iel',
      'time_elasped': 36789.51601886749,
      'title': 'Ich🍆iel'},
     {'is_video': False,
      'num_comments': 15,
      'num_crossposts': 0,
      'subreddit': 'mylittlepony',
      'time_elasped': 16493.51602101326,
      'title': 'Is it just me or does anyone else think rainbow dash with a pony tail is really cute?'},
     {'is_video': False,
      'num_comments': 525,
      'num_crossposts': 0,
      'subreddit': 'MemeEconomy',
      'time_elasped': 32929.51602315903,
      'title': "Saw the template and thought I'd give it a go. Invest?"},
     {'is_video': False,
      'num_comments': 43,
      'num_crossposts': 0,
      'subreddit': 'delusionalcraigslist',
      'time_elasped': 36935.5160241127,
      'title': '“I got 300 floppy drives and 60 DVD-ROM drives. I know what they’re worth”'},
     {'is_video': False,
      'num_comments': 79,
      'num_crossposts': 0,
      'subreddit': 'canucks',
      'time_elasped': 21154.51602602005,
      'title': 'Canucks Sign Forward Petrus Palmu to a three-year, entry-level contract'},
     {'is_video': False,
      'num_comments': 17,
      'num_crossposts': 0,
      'subreddit': 'berserklejerk',
      'time_elasped': 31704.5160279274,
      'title': 'Accurate?'},
     {'is_video': False,
      'num_comments': 25,
      'num_crossposts': 0,
      'subreddit': 'NetflixBestOf',
      'time_elasped': 11434.516029834747,
      'title': '[US] this is just a recommendation. If you haven’t watched “Wind River” (2017) on Netflix yet plz do yourself the favor. Great film with a strong message. From a writer and director that’s up and coming'},
     {'is_video': False,
      'num_comments': 3,
      'num_crossposts': 0,
      'subreddit': 'WaltDisneyWorld',
      'time_elasped': 31611.51603102684,
      'title': 'This magnificent horse!'},
     {'is_video': False,
      'num_comments': 27,
      'num_crossposts': 0,
      'subreddit': 'batty',
      'time_elasped': 30877.51603293419,
      'title': 'Help! It’s Memorial Day and I’m trying to save this guy we found in our bathtub. Gave him some water, now he’s in a cat carrier with some towels. We have to work soon and aren’t sure what to do.'},
     {'is_video': False,
      'num_comments': 22,
      'num_crossposts': 0,
      'subreddit': 'shield',
      'time_elasped': 46943.516035079956,
      'title': 'In celebration of 5 amazing seasons I drew some fan art of the show, I hope you like it.'},
     {'is_video': False,
      'num_comments': 7,
      'num_crossposts': 0,
      'subreddit': 'sploot',
      'time_elasped': 39812.51603603363,
      'title': 'Half on, half off sploot'},
     {'is_video': False,
      'num_comments': 16,
      'num_crossposts': 0,
      'subreddit': 'CitiesSkylines',
      'time_elasped': 26412.51603794098,
      'title': 'My 200K pop city got bitchsmacked by a 9.3 tsunami. This is only part of it.'},
     {'is_video': False,
      'num_comments': 10,
      'num_crossposts': 0,
      'subreddit': 'GaySoundsShitposts',
      'time_elasped': 35846.51603984833,
      'title': 'Women’s clothing designers: lol'},
     {'is_video': False,
      'num_comments': 2,
      'num_crossposts': 1,
      'subreddit': 'ExposurePorn',
      'time_elasped': 45169.51604104042,
      'title': 'Snowy scene - 30 seconds during golden hour [OC] Norway [1800x1341]'},
     {'is_video': False,
      'num_comments': 24,
      'num_crossposts': 0,
      'subreddit': 'minnesota',
      'time_elasped': 26030.51604294777,
      'title': 'The Minnesotan "Wrong Number"'},
     {'is_video': False,
      'num_comments': 40,
      'num_crossposts': 0,
      'subreddit': 'analog',
      'time_elasped': 52494.51604413986,
      'title': 'Welcome to Israel // Nikon FM2 // 50mm 1.8 // Fuji C200 (I think)'},
     {'is_video': False,
      'num_comments': 15,
      'num_crossposts': 0,
      'subreddit': 'DarlingInTheFranxx',
      'time_elasped': 19702.51604604721,
      'title': 'Nonose'},
     {'is_video': False,
      'num_comments': 6,
      'num_crossposts': 0,
      'subreddit': 'kittens',
      'time_elasped': 37097.516047000885,
      'title': 'The new foster babies have arrived! Meet Rocky'},
     {'is_video': False,
      'num_comments': 95,
      'num_crossposts': 0,
      'subreddit': 'nintendo',
      'time_elasped': 34332.516048908234,
      'title': 'Wulverblade sells significantly (3x) more on Switch than PS4, Xbox One, &amp; Steam combined'},
     {'is_video': False,
      'num_comments': 12,
      'num_crossposts': 0,
      'subreddit': 'Bioshock',
      'time_elasped': 23300.516051054,
      'title': '[OC] Bioshock Inked-Bouncer'},
     {'is_video': False,
      'num_comments': 25,
      'num_crossposts': 0,
      'subreddit': 'cynicalbritofficial',
      'time_elasped': 37355.51605296135,
      'title': "Jim Sterling talking about tB on this week's Jimquisition"},
     {'is_video': False,
      'num_comments': 10,
      'num_crossposts': 0,
      'subreddit': 'hmmmgifs',
      'time_elasped': 32281.516055107117,
      'title': 'hmmm'},
     {'is_video': False,
      'num_comments': 38,
      'num_crossposts': 0,
      'subreddit': 'Python',
      'time_elasped': 33289.51605606079,
      'title': 'I wrote a script that builds Spotify playlists for me!'},
     {'is_video': False,
      'num_comments': 28,
      'num_crossposts': 0,
      'subreddit': 'functionalprint',
      'time_elasped': 31069.51605820656,
      'title': 'I made a lightweight backpacking timelapse rig that will track the stars out of a $15 clock movement.'},
     {'is_video': False,
      'num_comments': 120,
      'num_crossposts': 0,
      'subreddit': 'PS4',
      'time_elasped': 33022.516058921814,
      'title': 'Battlefield V on Twitter: "There are no more Battlepacks. Instead, players will be able to choose their rewards directly or through rank up events."'},
     {'is_video': False,
      'num_comments': 32,
      'num_crossposts': 0,
      'subreddit': 'AnimalsBeingBros',
      'time_elasped': 77232.51606082916,
      'title': 'HMK while I take the dog for a walk'},
     {'is_video': False,
      'num_comments': 12,
      'num_crossposts': 0,
      'subreddit': 'Bioshock',
      'time_elasped': 23300.516062021255,
      'title': '[OC] Bioshock Inked-Bouncer'},
     {'is_video': False,
      'num_comments': 24,
      'num_crossposts': 0,
      'subreddit': 'minnesota',
      'time_elasped': 26030.516063928604,
      'title': 'The Minnesotan "Wrong Number"'},
     {'is_video': False,
      'num_comments': 120,
      'num_crossposts': 0,
      'subreddit': 'PS4',
      'time_elasped': 33022.51606607437,
      'title': 'Battlefield V on Twitter: "There are no more Battlepacks. Instead, players will be able to choose their rewards directly or through rank up events."'},
     {'is_video': False,
      'num_comments': 17,
      'num_crossposts': 0,
      'subreddit': 'jellybeantoes',
      'time_elasped': 46946.51606798172,
      'title': 'Beans and sleeping smile.'},
     {'is_video': False,
      'num_comments': 28,
      'num_crossposts': 0,
      'subreddit': 'functionalprint',
      'time_elasped': 31069.516069173813,
      'title': 'I made a lightweight backpacking timelapse rig that will track the stars out of a $15 clock movement.'},
     {'is_video': False,
      'num_comments': 74,
      'num_crossposts': 0,
      'subreddit': 'Dachshund',
      'time_elasped': 39238.51607084274,
      'title': 'We’re about 3 weeks away from picking this lil boy up from the breeder and couldn’t be more excited. But him being our first dog we’re getting a bit antsy/nervous. We both work full time but work close enough to come home on lunch breaks and plan to get a dog walker too. Any advice is appreciated!'},
     {'is_video': False,
      'num_comments': 186,
      'num_crossposts': 1,
      'subreddit': 'fakealbumcovers',
      'time_elasped': 85802.51607298851,
      'title': 'This is America – Childish Gambino'},
     {'is_video': False,
      'num_comments': 26,
      'num_crossposts': 0,
      'subreddit': 'NetflixBestOf',
      'time_elasped': 11434.516074895859,
      'title': '[US] this is just a recommendation. If you haven’t watched “Wind River” (2017) on Netflix yet plz do yourself the favor. Great film with a strong message. From a writer and director that’s up and coming'},
     {'is_video': False,
      'num_comments': 6,
      'num_crossposts': 0,
      'subreddit': 'Heavymind',
      'time_elasped': 37615.51607608795,
      'title': 'COSMIC STRUCTURES by Oska'},
     {'is_video': False,
      'num_comments': 7,
      'num_crossposts': 0,
      'subreddit': 'medicalschool',
      'time_elasped': 47962.5160779953,
      'title': "I'll definitely get to it tomorrow... [Shitpost]"},
     {'is_video': False,
      'num_comments': 10,
      'num_crossposts': 0,
      'subreddit': 'funkopop',
      'time_elasped': 15747.516079187393,
      'title': "Today's project. Solo themed baseball bat case with various posters as the background."},
     {'is_video': False,
      'num_comments': 25,
      'num_crossposts': 0,
      'subreddit': 'imaginarymaps',
      'time_elasped': 27014.516080856323,
      'title': 'Map of Post-WWII Europe (Western Red Tide)'},
     {'is_video': False,
      'num_comments': 21,
      'num_crossposts': 0,
      'subreddit': 'japan',
      'time_elasped': 34866.51608300209,
      'title': 'Coca-Cola launches its first alcoholic drink in Japan'},
     {'is_video': False,
      'num_comments': 14,
      'num_crossposts': 0,
      'subreddit': 'fitbit',
      'time_elasped': 27130.51608490944,
      'title': 'Hit some milestones this weekend. Hiked El Capitan in Yosemite.'},
     {'is_video': False,
      'num_comments': 4,
      'num_crossposts': 0,
      'subreddit': 'wholesomebpt',
      'time_elasped': 25779.516086101532,
      'title': 'I too enjoy messaging the same person on multiple platforms'},
     {'is_video': False,
      'num_comments': 236,
      'num_crossposts': 1,
      'subreddit': 'funny',
      'time_elasped': 38099.51608800888,
      'title': 'Fit'},
     {'is_video': False,
      'num_comments': 95,
      'num_crossposts': 2,
      'subreddit': 'geopolitics',
      'time_elasped': 47345.51609015465,
      'title': 'India says it only follows U.N. sanctions, not unilateral US sanctions on Iran | Reuters'},
     {'is_video': False,
      'num_comments': 38,
      'num_crossposts': 0,
      'subreddit': 'whowouldwin',
      'time_elasped': 35870.5160908699,
      'title': "The entire British commonwealth's population's strength and speed is copied and absorbed by the queen, whose the strongest person she can beat"},
     {'is_video': False,
      'num_comments': 15,
      'num_crossposts': 0,
      'subreddit': 'DarlingInTheFranxx',
      'time_elasped': 19702.51609301567,
      'title': 'Nonose'},
     {'is_video': False,
      'num_comments': 4,
      'num_crossposts': 0,
      'subreddit': 'architecture',
      'time_elasped': 40096.51609492302,
      'title': "Château d'Azay-le-Rideau [building]"},
     {'is_video': False,
      'num_comments': 2,
      'num_crossposts': 0,
      'subreddit': 'DestroyedTanks',
      'time_elasped': 18688.516096115112,
      'title': 'Knocked out Iraqi T-54/55 in the Kuwait oilfields, 1991 Gulf War'},
     {'is_video': False,
      'num_comments': 2,
      'num_crossposts': 0,
      'subreddit': 'CabinPorn',
      'time_elasped': 47957.51609802246,
      'title': 'Cabin in Mount Lyford, New Zealand [OC]'},
     {'is_video': False,
      'num_comments': 22,
      'num_crossposts': 0,
      'subreddit': 'gaybros',
      'time_elasped': 19065.516100168228,
      'title': 'Happy Memorial Day'},
     {'is_video': False,
      'num_comments': 175,
      'num_crossposts': 0,
      'subreddit': 'kpop',
      'time_elasped': 54848.5161011219,
      'title': 'SHINee (샤이니) - 데리러 가 (Good Evening)'},
     {'is_video': False,
      'num_comments': 5,
      'num_crossposts': 0,
      'subreddit': 'bitchimabus',
      'time_elasped': 26764.51610302925,
      'title': "Bitch I'm a UFO Recovery Bus."},
     ...]




```python
df = pd.DataFrame(infos)
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
      <th>is_video</th>
      <th>num_comments</th>
      <th>num_crossposts</th>
      <th>subreddit</th>
      <th>time_elasped</th>
      <th>title</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>False</td>
      <td>191</td>
      <td>1</td>
      <td>aww</td>
      <td>14276.158592</td>
      <td>A bit overprotective</td>
    </tr>
    <tr>
      <th>1</th>
      <td>False</td>
      <td>275</td>
      <td>0</td>
      <td>pics</td>
      <td>12186.158596</td>
      <td>Let us never forget.</td>
    </tr>
    <tr>
      <th>2</th>
      <td>False</td>
      <td>1027</td>
      <td>1</td>
      <td>FortNiteBR</td>
      <td>12352.158598</td>
      <td>New updates/ rideable shopping carts</td>
    </tr>
    <tr>
      <th>3</th>
      <td>False</td>
      <td>650</td>
      <td>1</td>
      <td>funny</td>
      <td>20388.158608</td>
      <td>Dave Bautista has achieved full Drax.</td>
    </tr>
    <tr>
      <th>4</th>
      <td>False</td>
      <td>66</td>
      <td>1</td>
      <td>AnimalsBeingBros</td>
      <td>15225.158610</td>
      <td>Hey Human, Want a Treat?</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.num_comments.median()
```




    19.0




```python
## YOUR CODE HERE
```

### Save your results as a CSV
You may do this regularly while scraping data as well, so that if your scraper stops of your computer crashes, you don't lose all your data.


```python
# Export to csv
df.to_csv('./df.csv', index=False)
df.head()
```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    <ipython-input-1-f8fffc08cdf0> in <module>()
          1 # Export to csv
    ----> 2 df.to_csv('./df.csv', index=False)
          3 df.head()


    NameError: name 'df' is not defined


## Predicting comments using Random Forests + Another Classifier

#### Load in the the data of scraped results

## Reading and Cleaning
The code below shows the the data being read into from a csv to a pandas dataframe. I wanted to get rid of duplicate posts. To do this I needed to remove time_elasped from the data frame since the time the posts were pulled, even for duplicates, differ. Once I found all the duplicates I removed them from the dataframe and then added the time_elasped feature back to the dataframe lacking duplicates. The time_elasped feature is given in units of seconds from the UTC 'timezone'. To make it easier to interpret and read, I converted the seconds to hours by dividing by 3600.


```python
## YOUR CODE HERE
import pandas as pd
df = pd.read_csv('./df.csv')
feats = [ col for col in df.columns if col != 'time_elasped'] # time is different even for duplicates
df = df[df[feats].duplicated(keep=False)==False] # removing duplicates
df['time_elasped'] = df['time_elasped']/3600

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
      <th>is_video</th>
      <th>num_comments</th>
      <th>num_crossposts</th>
      <th>subreddit</th>
      <th>time_elasped</th>
      <th>title</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>False</td>
      <td>191</td>
      <td>1</td>
      <td>aww</td>
      <td>3.965600</td>
      <td>A bit overprotective</td>
    </tr>
    <tr>
      <th>1</th>
      <td>False</td>
      <td>275</td>
      <td>0</td>
      <td>pics</td>
      <td>3.385044</td>
      <td>Let us never forget.</td>
    </tr>
    <tr>
      <th>2</th>
      <td>False</td>
      <td>1027</td>
      <td>1</td>
      <td>FortNiteBR</td>
      <td>3.431155</td>
      <td>New updates/ rideable shopping carts</td>
    </tr>
    <tr>
      <th>3</th>
      <td>False</td>
      <td>650</td>
      <td>1</td>
      <td>funny</td>
      <td>5.663377</td>
      <td>Dave Bautista has achieved full Drax.</td>
    </tr>
    <tr>
      <th>4</th>
      <td>False</td>
      <td>66</td>
      <td>1</td>
      <td>AnimalsBeingBros</td>
      <td>4.229211</td>
      <td>Hey Human, Want a Treat?</td>
    </tr>
  </tbody>
</table>
</div>



#### We want to predict a binary variable - whether the number of comments was low or high. Compute the median number of comments and create a new binary variable that is true when the number of comments is high (above the median)

We could also perform Linear Regression (or any regression) to predict the number of comments here. Instead, we are going to convert this into a _binary_ classification problem, by predicting two classes, HIGH vs LOW number of comments.

While performing regression may be better, performing classification may help remove some of the noise of the extremely popular threads. We don't _have_ to choose the `median` as the splitting point - we could also split on the 75th percentile or any other reasonable breaking point.

In fact, the ideal scenario may be to predict many levels of comment numbers. 


```python
## YOUR CODE HERE
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, train_test_split, cross_val_score
from sklearn.preprocessing import Binarizer, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer, HashingVectorizer, TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import itertools
```

# The Binarizer
We want to make y, our target, the number of comments, into a binary variable, 'Hot or Not'. Recall that a 'Hot' post is one that's number of commets is above the median number of comments and a 'Not' post is below the median number of comments. To do this I used a sci-kit learn transformer called Binarizer to create a binary variable. Setting the Binarizer's threshold to the median number of commets transforms the number of comments feature, so that if the number of comments is above the median number of comments the Binarizer returns a 1 and if it is below returns a 0. I named this feature 'Hot or Not. 


```python
y = df[['num_comments']]
bi = Binarizer(threshold=np.median(y))
y = bi.fit_transform(y)
y = pd.DataFrame(y)
y = y.rename({0:'Hot_or_Not'}, axis =1)
y.head()
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
      <th>Hot_or_Not</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



#### Thought experiment: What is the baseline accuracy for this model?

The baseline accuracy for this model is the 50.59% of the majority class, the 'Nots'. The goal of the upcoming models will be to become more accurate than this baseline.


```python
## YOUR CODE HERE
print('Number of Hots in',(y == 1).sum(), 'Percent of Hots', (y==1).sum()/len(y))
print('Number of Nots in', (y==0).sum(), 'Percent of Nots', (y==0).sum()/len(y))
#percent of nots check it though.
```

    Number of Hots in Hot_or_Not    1253
    dtype: int64 Percent of Hots Hot_or_Not    0.494085
    dtype: float64
    Number of Nots in Hot_or_Not    1283
    dtype: int64 Percent of Nots Hot_or_Not    0.505915
    dtype: float64


#### Create a Random Forest model to predict High/Low number of comments using Sklearn. Start by ONLY using the subreddit as a feature. 

# Fist Model - Subreddit 
This model uses the subreddit as the only feature to predict if a comment is 'Hot' or 'Not'. To make sure the model tests well on unseen data a train test split was utitlized. Using Contvectorizer as the transformer, which creates  feature columns for each subreddit and Random Forest, which uses multiple decisions trees with different features in each tree, is the model that will be employed to predict 'Hot or Not'. Without any optimzation this model returns an accuracy of 63.25%, which is higher than the baseline. Using GridSearchCV to optimize parameters returns an accuracy of 60.10% which is worse, but still higher than the baseline. The other metric I investigated was the confusion matrix to see the distribution of true negatives, false positives, true positives and false positives. Than using feature importance I look at the top 10 subreddits that contribute to a 'Hot' post.


```python
y.head()
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
      <th>Hot_or_Not</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
## YOUR CODE HERE
X = df['subreddit'] # 1 feature
y = y['Hot_or_Not']
y


```




    0       1
    1       1
    2       1
    3       1
    4       1
    5       1
    6       1
    7       1
    8       1
    9       1
    10      1
    11      1
    12      1
    13      1
    14      1
    15      1
    16      1
    17      1
    18      1
    19      1
    20      1
    21      1
    22      1
    23      1
    24      1
    25      1
    26      1
    27      1
    28      1
    29      1
           ..
    2506    0
    2507    0
    2508    1
    2509    0
    2510    0
    2511    0
    2512    0
    2513    0
    2514    1
    2515    0
    2516    0
    2517    0
    2518    1
    2519    1
    2520    0
    2521    1
    2522    0
    2523    0
    2524    1
    2525    1
    2526    0
    2527    1
    2528    1
    2529    1
    2530    1
    2531    1
    2532    1
    2533    1
    2534    1
    2535    1
    Name: Hot_or_Not, Length: 2536, dtype: int64




```python
#Train-test-split
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=111)
X_train.shape
y_train.shape
y_test.shape
X_test.shape
```




    (634,)




```python
#1st model subreddit
pipe = Pipeline([
    ('cvec', CountVectorizer()),
    ('RF', RandomForestClassifier(random_state=47))
])
```


```python
pipe.fit(X_train, y_train)
pipe.score(X_test, y_test)
```




    0.6324921135646687




```python
params = {
    'cvec__binary':[False, True],
    'RF__n_estimators':np.arange(1,10,1),
    'RF__min_samples_split': np.arange(2,11,1),
    'RF__max_features':[None,"sqrt", "log2"]
}
```


```python
gs = GridSearchCV(pipe, param_grid=params, cv=3)
gs.fit(X_train, y_train)
```




    GridSearchCV(cv=3, error_score='raise',
           estimator=Pipeline(memory=None,
         steps=[('cvec', CountVectorizer(analyzer='word', binary=False, decode_error='strict',
            dtype=<class 'numpy.int64'>, encoding='utf-8', input='content',
            lowercase=True, max_df=1.0, max_features=None, min_df=1,
            ngram_range=(1, 1), preprocessor=None, stop_words=None,
            strip...stimators=10, n_jobs=1,
                oob_score=False, random_state=47, verbose=0, warm_start=False))]),
           fit_params=None, iid=True, n_jobs=1,
           param_grid={'cvec__binary': [False, True], 'RF__n_estimators': array([1, 2, 3, 4, 5, 6, 7, 8, 9]), 'RF__min_samples_split': array([ 2,  3,  4,  5,  6,  7,  8,  9, 10]), 'RF__max_features': [None, 'sqrt', 'log2']},
           pre_dispatch='2*n_jobs', refit=True, return_train_score='warn',
           scoring=None, verbose=0)




```python
#Top score and optimized parameters
print(gs.best_score_)
print(gs.best_params_)
```

    0.6009463722397477
    {'RF__max_features': 'log2', 'RF__min_samples_split': 9, 'RF__n_estimators': 9, 'cvec__binary': False}



```python
#confusion matrix
y_pred = gs.predict(X_test)
tn,fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
```


```python
print('tn', tn,
     'fp', fp,
     'fn', fn,
     'tp', tp)


```

    tn 284 fp 37 fn 189 tp 124



```python
# Source: http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html#sphx-glr-auto-examples-model-selection-plot-confusion-matrix-py
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Reds):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
```


```python
plt.figure()
plot_confusion_matrix(confusion_matrix(y_test,y_pred), classes =['Not', 'Hot'], title = 'Subreddit as Feature')
```


![png](/images/starter-code_files/starter-code_51_0.png)



```python
# Source: http://scikit-learn.org/stable/auto_examples/ensemble/plot_forest_importances.html
f= pipe.steps[1][1].feature_importances_
g = pipe.steps[0][1].get_feature_names()
s =sorted(list(zip(f, g)),reverse=True)[:10]
plt.figure(figsize=(20, 15))
plt.title('Subreddit Importance', fontsize=20)
sy = [y[0] for y in s]
sx = [x[1] for x in s]
plt.bar(sx, sy, color ='r')
plt.xticks(fontsize=12)
plt.tight_layout
plt.show()
```


![png](/images/starter-code_files/starter-code_52_0.png)


#### Create a few new variables in your dataframe to represent interesting features of a thread title.
- For example, create a feature that represents whether 'cat' is in the title or whether 'funny' is in the title. 
- Then build a new Random Forest with these features. Do they add any value?
- After creating these variables, use count-vectorizer to create features based on the words in the thread titles.
- Build a new random forest model with subreddit and these new features included.

# Second Model - Fun Features
This model uses a number of features that were engineered. The is_video feature, tells if the posts are a video or not. Utitlizing the '.astype(int) makes the trues and falses of is_video into a dummy column giving 1 for true and 0 for false. Investigating punctuation I looked into if posts' titles were questions, exclamations or if people appreciated good grammer and if the post ended with a period. The last two features are if cats were in the title and if the word when is in the title. Since all the variables have been dummied using '.astype(int)' this model just used randomforest. Without optimization the model returned an accuracy score of 55.99%, better than baseline, With GridsearchCV optimization the model returned an accuracy of 53.05%, a little worse but better than the baseline. A confusion matrix was used as a metric for this model as well as feature importance to see which of these fun features contributed to predicted 'Hot or Not'. 


```python
## YOUR CODE HERE
df['vidint'] = df['is_video'].astype(int)
df['question'] = df['title'].str.endswith('?').astype(int)

df['exclamation'] = df['title'].str.endswith('!').astype(int)
df['period'] = df['title'].str.endswith('.').astype(int)
df['cats'] = df['title'].str.contains(r'cat|cats|Cat|Cats').astype(int)
df['when'] = df['title'].str.contains(r'when|When').astype(int)
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
      <th>is_video</th>
      <th>num_comments</th>
      <th>num_crossposts</th>
      <th>subreddit</th>
      <th>time_elasped</th>
      <th>title</th>
      <th>vidint</th>
      <th>question</th>
      <th>exclamation</th>
      <th>period</th>
      <th>cats</th>
      <th>when</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>False</td>
      <td>191</td>
      <td>1</td>
      <td>aww</td>
      <td>3.965600</td>
      <td>A bit overprotective</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>False</td>
      <td>275</td>
      <td>0</td>
      <td>pics</td>
      <td>3.385044</td>
      <td>Let us never forget.</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>False</td>
      <td>1027</td>
      <td>1</td>
      <td>FortNiteBR</td>
      <td>3.431155</td>
      <td>New updates/ rideable shopping carts</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>False</td>
      <td>650</td>
      <td>1</td>
      <td>funny</td>
      <td>5.663377</td>
      <td>Dave Bautista has achieved full Drax.</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>False</td>
      <td>66</td>
      <td>1</td>
      <td>AnimalsBeingBros</td>
      <td>4.229211</td>
      <td>Hey Human, Want a Treat?</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
#feature selction and train test split
X = df[['vidint', 'question', 'exclamation', 'period', 'cats', 'when']]
X_train, X_test, y_train, y_test = train_test_split(X, y,stratify=y,random_state= 55)
X_train.shape
y_train.shape
X_test.shape
y_test.shape


```




    (634,)




```python
#2nd model fun features
pipe_2 = Pipeline([
    ('RF', RandomForestClassifier(random_state=30))
])
pipe_2.fit(X_train,y_train)
pipe_2.score(X_test, y_test)
```




    0.5599369085173501




```python
params = {
    'RF__n_estimators':np.arange(1,10,1),
    'RF__min_samples_split': np.arange(2,11,1),
    'RF__max_features':[None,"sqrt", "log2"]
}
```


```python
gs_2 = GridSearchCV(pipe_2, param_grid=params, cv=3)
gs_2.fit(X_train, y_train)
```




    GridSearchCV(cv=3, error_score='raise',
           estimator=Pipeline(memory=None,
         steps=[('RF', RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
                max_depth=None, max_features='auto', max_leaf_nodes=None,
                min_impurity_decrease=0.0, min_impurity_split=None,
                min_samples_leaf=1, min_samples_split=2,
                min_weight_fraction_leaf=0.0, n_estimators=10, n_jobs=1,
                oob_score=False, random_state=30, verbose=0, warm_start=False))]),
           fit_params=None, iid=True, n_jobs=1,
           param_grid={'RF__n_estimators': array([1, 2, 3, 4, 5, 6, 7, 8, 9]), 'RF__min_samples_split': array([ 2,  3,  4,  5,  6,  7,  8,  9, 10]), 'RF__max_features': [None, 'sqrt', 'log2']},
           pre_dispatch='2*n_jobs', refit=True, return_train_score='warn',
           scoring=None, verbose=0)




```python
print(gs_2.best_score_)
print(gs_2.best_params_)
```

    0.5304942166140905
    {'RF__max_features': 'sqrt', 'RF__min_samples_split': 2, 'RF__n_estimators': 4}



```python
y_pred = gs_2.predict(X_test)
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
```


```python
print('tn', tn,
     'fp', fp,
     'fn', fn,
     'tp', tp)
```

    tn 253 fp 68 fn 212 tp 101



```python
plt.figure()
plot_confusion_matrix(confusion_matrix(y_test,y_pred), classes =['Not', 'Hot'], title = 'Fun Features')
```


![png](/images/starter-code_files/starter-code_63_0.png)



```python
f = pipe_2.steps[0][1].feature_importances_
s =sorted(list(zip(f, X_train.columns)),reverse=True)[:6]
plt.figure(figsize=(20, 15))
plt.title('Fun Features Importance', fontsize=20)
sy = [y[0] for y in s]
sx = [x[1] for x in s]
plt.bar(sx, sy, color ='r')
plt.xticks(fontsize=16)
plt.tight_layout
plt.show()
```


![png](/images/starter-code_files/starter-code_64_0.png)


# Model 3 - Many features RandomForest
This model utilzes the fun features form model 2, the subbreddits from model 1, time_elasped, number of crossposts, and the words in the title of the reddit posts. To use the subreddit and the titles I needed to countvectorize each feature. Since titles contain multiple words I used a stop_word='eglish in the countvectorizer. This would filter out words that do not transmit much information in the english language, like 'the', 'a' , and 'an' for example. Countvectorizer returns matrices for your titles and subreddits so I unpacked them into arrays and made them into pandas dataframes. Once they are dataframes I concatenated them to the orginal features I wanted and this dataframe became my features for this model. Having numerical data like number of crossposts and time elasped I used the transformer standarscaler to scale all the features as a precaution. I ran a randomforest model which when not optimized produced an accuracy of 67.03% and when optimized using GridsearchCV produced an accuracy of 68.82%. A confusion matrix was used as a metric to test the model's performance and feature importance was used to see what features contributed to predicted a 'Hot' post.


```python
X =df[['title','subreddit','time_elasped','num_crossposts','vidint', 'question', 'exclamation', 'period', 'cats', 'when']]
```


```python
#transform titles
cvec = CountVectorizer(stop_words='english')
X_title = cvec.fit_transform(df['title'])
cvec.get_feature_names()
X_title.toarray()
words_in_title= pd.DataFrame(X_title.toarray(), columns=cvec.get_feature_names(), index = X.index)
```


```python
words_in_title.shape
```




    (2536, 6737)




```python
# transoform subreddit
cvec = CountVectorizer()
X_subreddit = cvec.fit_transform(df['subreddit'])
cvec.get_feature_names()
X_subreddit.toarray()
subred= pd.DataFrame(X_subreddit.toarray(), columns=cvec.get_feature_names(), index = X.index)
```


```python
subred.shape
```




    (2536, 1814)




```python
X =df[['time_elasped','num_crossposts','vidint', 'question', 'exclamation', 'period', 'cats', 'when']]
```


```python
#concat wanted features together
X = pd.concat([words_in_title, subred, X], axis=1)
```


```python
X.head()
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
      <th>00</th>
      <th>000</th>
      <th>008004</th>
      <th>02</th>
      <th>03</th>
      <th>04</th>
      <th>05</th>
      <th>050</th>
      <th>07</th>
      <th>0m</th>
      <th>...</th>
      <th>zerowaste</th>
      <th>zoomies</th>
      <th>time_elasped</th>
      <th>num_crossposts</th>
      <th>vidint</th>
      <th>question</th>
      <th>exclamation</th>
      <th>period</th>
      <th>cats</th>
      <th>when</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>3.965600</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>3.385044</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>3.431155</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>5.663377</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>4.229211</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 8559 columns</p>
</div>




```python
#train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state =121)
X_train.shape
y_train.shape
X_test.shape
y_test.shape
```




    (634,)




```python
# model 3 many feats
pipe_3 = Pipeline([
    ('ss', StandardScaler()),
    ('rf', RandomForestClassifier(random_state=8))
])
```


```python
pipe_3.fit(X_train, y_train)
pipe_3.score(X_test, y_test)
```




    0.6703470031545742




```python
params = {
    'rf__n_estimators':np.arange(1,10,1),
    'rf__min_samples_split': np.arange(2,11,1),
    'rf__max_features':[None,"sqrt", "log2"]
}
```


```python
gs_3 = GridSearchCV(pipe_3, param_grid=params, cv=3)
gs_3.fit(X_train, y_train)
```




    GridSearchCV(cv=3, error_score='raise',
           estimator=Pipeline(memory=None,
         steps=[('ss', StandardScaler(copy=True, with_mean=True, with_std=True)), ('rf', RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
                max_depth=None, max_features='auto', max_leaf_nodes=None,
                min_impurity_decrease=0.0...stimators=10, n_jobs=1,
                oob_score=False, random_state=8, verbose=0, warm_start=False))]),
           fit_params=None, iid=True, n_jobs=1,
           param_grid={'rf__n_estimators': array([1, 2, 3, 4, 5, 6, 7, 8, 9]), 'rf__min_samples_split': array([ 2,  3,  4,  5,  6,  7,  8,  9, 10]), 'rf__max_features': [None, 'sqrt', 'log2']},
           pre_dispatch='2*n_jobs', refit=True, return_train_score='warn',
           scoring=None, verbose=0)




```python
print(gs_3.best_score_)
print(gs_3.best_params_)

```

    0.6882229232386962
    {'rf__max_features': 'sqrt', 'rf__min_samples_split': 6, 'rf__n_estimators': 9}



```python
y_pred = gs_3.predict(X_test)
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
```


```python
print('tn', tn,
     'fp', fp,
     'fn', fn,
     'tp', tp)
```

    tn 229 fp 92 fn 117 tp 196



```python
plt.figure()
plot_confusion_matrix(confusion_matrix(y_test,y_pred), classes =['Not', 'Hot'], title = 'Many Features')
```


![png](/images/starter-code_files/starter-code_82_0.png)



```python
f = pipe_3.steps[1][1].feature_importances_
s = sorted(list(zip(f, X_train.columns)),reverse=True)[:10]
plt.figure(figsize=(20, 15))
plt.title('Many Features', fontsize=20)
sy = [y[0] for y in s]
sx = [x[1] for x in s]
plt.bar(sx, sy, color ='r')
plt.xticks(fontsize=12)
plt.tight_layout
plt.show()
```


![png](/images/starter-code_files/starter-code_83_0.png)


#### Use cross-validation in scikit-learn to evaluate the model above. 
- Evaluate the accuracy of the model, as well as any other metrics you feel are appropriate. 


```python
## YOUR CODE HERE
cross_val_score(pipe_3, X_test, y_test, scoring='accuracy')

```




    array([0.67924528, 0.65402844, 0.65402844])



#### Repeat the model-building process with a non-tree-based method.

# Model 4 - Many features Logistic Regression
To examine another model besides randomforest I choose logistic regrssion for this classification problem. Utiling the same features as in model 3 I transformed the features using a standardscalar.Running the the model without optimization the accuracy score was 62.15%. Optimizing with GridsearchCV I got an accuracy score of 69.24%. GridsearchCV optimized the logistic regression to have a penalty of l1 the LASSO penalty and a low C of 0.1. I used a confusion matrix to further investigate the modles performance. I used a logistic regression coefs_ to see which features best predicted a 'Hot' post.


```python
## YOUR CODE HERE
from sklearn.linear_model import LogisticRegression
#model 4 many feats with log reg
pipe_4 = Pipeline([
    ('ss', StandardScaler()),
    ('logreg', LogisticRegression(random_state=111))
])

```


```python
pipe_4.fit(X_train, y_train)
pipe_4.score(X_test, y_test)
```




    0.6214511041009464




```python
params = {
    'logreg__penalty':['l1', 'l2'],
    'logreg__C': np.arange(0.1,1, 0.1)
    
}
```


```python
gs_4 = GridSearchCV(pipe_4, param_grid=params, cv=3)
gs_4.fit(X_train, y_train)
```




    GridSearchCV(cv=3, error_score='raise',
           estimator=Pipeline(memory=None,
         steps=[('ss', StandardScaler(copy=True, with_mean=True, with_std=True)), ('logreg', LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
              intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1,
              penalty='l2', random_state=111, solver='liblinear', tol=0.0001,
              verbose=0, warm_start=False))]),
           fit_params=None, iid=True, n_jobs=1,
           param_grid={'logreg__penalty': ['l1', 'l2'], 'logreg__C': array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])},
           pre_dispatch='2*n_jobs', refit=True, return_train_score='warn',
           scoring=None, verbose=0)




```python
print(gs_4.best_score_)
print(gs_4.best_params_)

```

    0.692429022082019
    {'logreg__C': 0.1, 'logreg__penalty': 'l1'}



```python
y_pred = gs_4.predict(X_test)
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
```


```python
print('tn', tn,
     'fp', fp,
     'fn', fn,
     'tp', tp)
```

    tn 263 fp 58 fn 136 tp 177



```python
plt.figure()
plot_confusion_matrix(confusion_matrix(y_test,y_pred), classes =['Not', 'Hot'], title = 'Many Features(Log)')
```


![png](/images/starter-code_files/starter-code_95_0.png)



```python
c = pipe_4.steps[1][1].coef_
s = sorted(list(zip(c.T[:], X_train)),reverse=True)[:10]
plt.figure(figsize=(20, 15))
plt.title('Many Features(Log)', fontsize=20)
sy = [y[0][0] for y in s]
sx = [x[1] for x in s]
plt.bar(sx, sy, color ='r')
plt.xticks(fontsize=12)
plt.tight_layout
plt.show()

```


![png](/images/starter-code_files/starter-code_96_0.png)


#### Use Count Vectorizer from scikit-learn to create features from the thread titles. 
- Examine using count or binary features in the model
- Re-evaluate your models using these. Does this improve the model performance? 
- What text features are the most valuable? 

# Model 5 Title as only Feature
This model is alot like the first model with subreddit being the only feature, but now the words in titles are the only features. One big difference is the stop_words ='english in the CountVectorizer. This will filter out words that do not transmit vital information, such as words like,'for', 'or' and 'they' as examples. When running an model without optmization, the model returns an accuracy score of 49.84% the worst scoring model and the only model to score worse than the baseline. When running gridsearch to optimize the model the model returns a score of 55.84%. The Gridsearch CV optimized the title feature utilizing the countvectorizer's binary set to equal False, which means each word gets a count everytime that word shows up in a post's title. 


```python
## YOUR CODE HERE
X = df['title']
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=22)
X_train.shape
y_train.shape
X_test.shape
y_test.shape
```




    (634,)




```python
#model 5 just with title
pipe_5 = Pipeline([
    ('cvec', CountVectorizer(stop_words='english')),
    ('rf', RandomForestClassifier(random_state=121))
])
```


```python
pipe_5.fit(X_train, y_train)
pipe_5.score(X_test, y_test)
```




    0.5283911671924291




```python
params = {
    'cvec__binary':[False, True],
    'cvec__ngram_range':[(1,1), (1,2), (1,3)]
}
```


```python
gs_5 =GridSearchCV(pipe_5, param_grid=params, cv=3)
```


```python
gs_5.fit(X_train, y_train)
```




    GridSearchCV(cv=3, error_score='raise',
           estimator=Pipeline(memory=None,
         steps=[('cvec', CountVectorizer(analyzer='word', binary=False, decode_error='strict',
            dtype=<class 'numpy.int64'>, encoding='utf-8', input='content',
            lowercase=True, max_df=1.0, max_features=None, min_df=1,
            ngram_range=(1, 1), preprocessor=None, stop_words='english',
            ...timators=10, n_jobs=1,
                oob_score=False, random_state=121, verbose=0, warm_start=False))]),
           fit_params=None, iid=True, n_jobs=1,
           param_grid={'cvec__binary': [False, True], 'cvec__ngram_range': [(1, 1), (1, 2), (1, 3)]},
           pre_dispatch='2*n_jobs', refit=True, return_train_score='warn',
           scoring=None, verbose=0)




```python
print(gs_5.best_score_)
print(gs_5.best_params_)
```

    0.544689800210305
    {'cvec__binary': True, 'cvec__ngram_range': (1, 1)}



```python
y_pred = gs_5.predict(X_test)
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
```


```python
print('tn', tn,
     'fp', fp,
     'fn', fn,
     'tp', tp)
```

    tn 236 fp 85 fn 207 tp 106



```python
plt.figure()
plot_confusion_matrix(confusion_matrix(y_test,y_pred), classes =['Not', 'Hot'], title = 'Title')
```


![png](/images/starter-code_files/starter-code_108_0.png)



```python
f= pipe_5.steps[1][1].feature_importances_
g = pipe_5.steps[0][1].get_feature_names()
s =sorted(list(zip(f, g)),reverse=True)[:10]

```


```python
plt.figure(figsize=(20, 15))
plt.title('Title', fontsize=20)
sy = [y[0] for y in s]
sx = [x[1] for x in s]
plt.bar(sx, sy, color ='r')
plt.xticks(fontsize=16)
plt.tight_layout
plt.show()
```


![png](/images/starter-code_files/starter-code_110_0.png)



```python
df[df['num_comments'] > df['num_comments'].median()]['time_elasped'].mean()
# avg number of hours for a hot post
```




    9.956260710650474




```python
df[df['num_comments'] > df['num_comments'].median()]['num_crossposts'].mean()
#avg number of crossposts for hot post
```




    0.6280925778132482




```python
df.num_comments.median()
```




    21.0


