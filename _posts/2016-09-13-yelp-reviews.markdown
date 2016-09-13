---
title:  "Predicting Star Ratings from 1+ GBs of Yelp Reviews"
date:   2016-09-13
tags: [data science]

header:
  image: "last_fm_spotify_edited.png"
  caption: "Photo credit: [**Slash Gear**](http://www.slashgear.com/last-fm-and-spotify-team-up-to-give-better-music-recommendations-29315027/)"

excerpt: "Code Focused. Logistic Regression, Stochastic Gradient Descent, Natural Language Processing"
---

Intro




# Predicting Ratings from 1+ GB of Yelp Reviews

Let's take a look at the Yelp data. It's about 1.25 Gigabytes, so I really don't want to load all that into my Macbook Air's memory. I'll read in 10,000 rows and just take a quick look.


```python
import pandas as pd
import numpy as np
from __future__ import division
import seaborn as sns
import re
import matplotlib.pyplot as plt
%matplotlib inline

# display results to 3 decimal points, not in scientific notation
pd.set_option('display.float_format', lambda x: '%.3f' % x)
```


```python
reviews_data = pd.read_csv('/Users/nickbecker/Python_Projects/yelp_academic_challenge/yelp_academic_dataset_review.csv',
                             nrows = 10000)
reviews_data.head(3)
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>user_id</th>
      <th>review_id</th>
      <th>text</th>
      <th>votes.cool</th>
      <th>business_id</th>
      <th>votes.funny</th>
      <th>stars</th>
      <th>date</th>
      <th>type</th>
      <th>votes.useful</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Xqd0DzHaiyRqVH3WRG7hzg</td>
      <td>15SdjuK7DmYqUAj6rjGowg</td>
      <td>dr. goldberg offers everything i look for in a...</td>
      <td>1</td>
      <td>vcNAWiLM4dR7D2nwwJ7nCA</td>
      <td>0</td>
      <td>5</td>
      <td>2007-05-17</td>
      <td>review</td>
      <td>2</td>
    </tr>
    <tr>
      <th>1</th>
      <td>H1kH6QZV7Le4zqTRNxoZow</td>
      <td>RF6UnRTtG7tWMcrO2GEoAg</td>
      <td>Unfortunately, the frustration of being Dr. Go...</td>
      <td>0</td>
      <td>vcNAWiLM4dR7D2nwwJ7nCA</td>
      <td>0</td>
      <td>2</td>
      <td>2010-03-22</td>
      <td>review</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>zvJCcrpm2yOZrxKffwGQLA</td>
      <td>-TsVN230RCkLYKBeLsuz7A</td>
      <td>Dr. Goldberg has been my doctor for years and ...</td>
      <td>1</td>
      <td>vcNAWiLM4dR7D2nwwJ7nCA</td>
      <td>0</td>
      <td>4</td>
      <td>2012-02-14</td>
      <td>review</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



Looks pretty good. We have user, review, and business identifiers in addition to the review text, stars, and other information. Let's look a sample review.


```python
print reviews_data.iloc[1000, :].text
```

    I've been going here since I was a wee tot.  I have always enjoyed their food even though there's nothing too special about it.  I really like their breakfast menu.  You always seem to get lots of food for very little money, which is nice.  
    
    My only complaint is that it is WAY TOO SMALL and always very busy.  They should really look at expanding the restaurant.


In other situations, we might read the whole dataset into memory and build a tf-idf or bag of words matrix. In fact, I've done that for other text analysis tasks (see []() and []() for examples). But here, since the data is so large, our processing and tokenization time would be gigantic even if we could fit everything into memory.

## How do we train a model using big data?

To get around this, we need to change about the optimization process. When the dataset size is small, optimization may be fastest by solving directly for the solution. In OLS linear regression, this would be performing the linear algebra to solve for the optimal weights (w = (X'X)-1X'y). As the dataset becomes larger, gradient descent becomes a more efficient way to find the optimal weights. Gradient descent isn't guaranteed to find the optimal weights (whether it does or not depends on the shape of the loss or likelihood function space), but in practice we are often fine.

However, as the dataset becomes **extremely** large, gradient descent becomes less effective. The size of the data just massively increase the number of steps required for gradient descent to converge. To use our massive amount of data, we need a new method.

## Stochastic Gradient Descent (or Ascent)

Stochastic gradient descent is the solution. Essentially, we're going to do the same gradient descent iteratively on random mini-batches of the data and use these to iteratively update our gradient to (hopefully) reach the optimal solution. Again, we aren't guaranteed to converge to the global optimium (we can get stuck in local optima just like before), but this approach has proven to work extremely well.

As a note, since I'm going to do logistic regression, we're actually maximizing a likelihood function instead of minimizing a loss function. That makes it gradient ascent, instead of descent. 

First, I'll use `pandas.read_csv` to create an iterator object with chunks of size 1000. With this, I can loop through the iterator and each loop will return a chunk of size 1000 from the dataset until I reach the end.


```python
reviews_iterator = pd.read_csv('/Users/nickbecker/Python_Projects/yelp_academic_challenge/yelp_academic_dataset_review.csv',
                             chunksize = 1000)
```

Next, I'll define a function to clean the review text (the same function I've used before) and a tokenizer to stem the words in the review. The tokenizer will be passed to the HashingVectorizer to capture words that should be treated as the same but have different endings.


```python
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk import word_tokenize

def clean_review(review):
    """
    Function to clean review text to keep only letters and remove stopwords
    Returns a string of the cleaned bill text
    """
    letters_only = re.sub('[^a-zA-Z]', ' ', review)
    words = letters_only.lower().split()
    stopwords_eng = set(stopwords.words("english"))
    useful_words = [x for x in words if not x in stopwords_eng]
    
    # Combine words into a paragraph again
    useful_words_string = ' '.join(useful_words)
    return(useful_words_string)

stemmer = PorterStemmer()

def stem_words(words_list, stemmer):
    return [stemmer.stem(word) for word in words_list]

def tokenize(text):
    tokens = word_tokenize(text)
    stems = stem_words(tokens, stemmer)
    return stems
```

I'll apply our functions to the 10,000 rows I read into in memory and use it as a validation set. With this, I can plot the SGD accuracy curve.

I'll also get rid of the 3-star reviews, as they're neutral on a 5-star scale.


```python
reviews_data = reviews_data.query('stars != 3')
reviews_data['clean_review'] = reviews_data['text'].apply(clean_review)
reviews_data['star_sentiment'] = reviews_data['stars'].apply(lambda x: 1 if x > 3 else 0)
```

Okay, time to build the model.


```python
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import HashingVectorizer

vectorizer = HashingVectorizer(decode_error='ignore', n_features = 2**18,
                               tokenizer = tokenize, non_negative=True)
y_classes = np.array([0, 1])
max_iterations = 299

clf_logit_sgd = SGDClassifier(loss = 'log', n_jobs = 2, learning_rate = 'optimal',
                              random_state = 12, verbose = 0, shuffle = True)

validation_accuracy_list = []

x_validation = vectorizer.transform(reviews_data['clean_review'])
y_validation = reviews_data['star_sentiment']

for i, mini_batch in enumerate(reviews_iterator): 
    mini_reviews_data = mini_batch.copy().query('stars != 3')
    mini_reviews_data['clean_review'] = mini_reviews_data['text'].apply(clean_review)
    mini_reviews_data['star_sentiment'] = mini_reviews_data['stars'].apply(lambda x: 1 if x > 3 else 0)

    x_train = vectorizer.transform(mini_reviews_data['clean_review'])
    y_train = mini_reviews_data['star_sentiment']

    clf_logit_sgd.partial_fit(x_train, y_train, classes = y_classes)

    if i % 1 == 0:
        validation_accuracy = clf_logit_sgd.score(x_validation, y_validation)
        validation_accuracy_list.append((i, validation_accuracy))
        print 'Validation Accuracy at {0}: {1}'.format(i, validation_accuracy)
            
    if i >= max_iterations:
        break
```

    Validation Accuracy at 0: 0.825370281892
    Validation Accuracy at 1: 0.869804108935
    Validation Accuracy at 2: 0.874104156713
    Validation Accuracy at 3: 0.877209746775
    Validation Accuracy at 4: 0.876493072145
    Validation Accuracy at 5: 0.889154323937
    Validation Accuracy at 6: 0.891662685141
    Validation Accuracy at 7: 0.885570950788
    Validation Accuracy at 8: 0.875059722886
    Validation Accuracy at 9: 0.876493072145
    Validation Accuracy at 10: 0.887004300048
    Validation Accuracy at 11: 0.893693263258
    Validation Accuracy at 12: 0.892140468227
    Validation Accuracy at 13: 0.893215480172
    Validation Accuracy at 14: 0.88640707119
    Validation Accuracy at 15: 0.889990444338
    Validation Accuracy at 16: 0.886168179646
    Validation Accuracy at 17: 0.895365504061
    Validation Accuracy at 18: 0.900143334926
    Validation Accuracy at 19: 0.895365504061
    Validation Accuracy at 20: 0.891184902054
    Validation Accuracy at 21: 0.896679407549
    Validation Accuracy at 22: 0.875776397516
    Validation Accuracy at 23: 0.892498805542
    Validation Accuracy at 24: 0.897993311037
    Validation Accuracy at 25: 0.894409937888
    Validation Accuracy at 26: 0.897037744864
    Validation Accuracy at 27: 0.888079311992
    Validation Accuracy at 28: 0.901218346871
    Validation Accuracy at 29: 0.894648829431
    Validation Accuracy at 30: 0.893693263258
    Validation Accuracy at 31: 0.89823220258
    Validation Accuracy at 32: 0.900023889154
    Validation Accuracy at 33: 0.897754419494
    Validation Accuracy at 34: 0.897873865265
    Validation Accuracy at 35: 0.897873865265
    Validation Accuracy at 36: 0.897157190635
    Validation Accuracy at 37: 0.895365504061
    Validation Accuracy at 38: 0.89894887721
    Validation Accuracy at 39: 0.898471094123
    Validation Accuracy at 40: 0.89823220258
    Validation Accuracy at 41: 0.899187768753
    Validation Accuracy at 42: 0.901815575729
    Validation Accuracy at 43: 0.900023889154
    Validation Accuracy at 44: 0.901098901099
    Validation Accuracy at 45: 0.899307214525
    Validation Accuracy at 46: 0.894887720975
    Validation Accuracy at 47: 0.896559961777
    Validation Accuracy at 48: 0.899307214525
    Validation Accuracy at 49: 0.900023889154
    Validation Accuracy at 50: 0.903129479216
    Validation Accuracy at 51: 0.900621118012
    Validation Accuracy at 52: 0.881629240325
    Validation Accuracy at 53: 0.90265169613
    Validation Accuracy at 54: 0.89823220258
    Validation Accuracy at 55: 0.897873865265
    Validation Accuracy at 56: 0.89823220258
    Validation Accuracy at 57: 0.895962732919
    Validation Accuracy at 58: 0.889034878165
    Validation Accuracy at 59: 0.898471094123
    Validation Accuracy at 60: 0.901457238414
    Validation Accuracy at 61: 0.898709985667
    Validation Accuracy at 62: 0.902412804587
    Validation Accuracy at 63: 0.887243191591
    Validation Accuracy at 64: 0.89010989011
    Validation Accuracy at 65: 0.896559961777
    Validation Accuracy at 66: 0.898709985667
    Validation Accuracy at 67: 0.900382226469
    Validation Accuracy at 68: 0.897754419494
    Validation Accuracy at 69: 0.903487816531
    Validation Accuracy at 70: 0.900262780698
    Validation Accuracy at 71: 0.90563784042
    Validation Accuracy at 72: 0.900860009556
    Validation Accuracy at 73: 0.904562828476
    Validation Accuracy at 74: 0.90336837076
    Validation Accuracy at 75: 0.900979455327
    Validation Accuracy at 76: 0.903248924988
    Validation Accuracy at 77: 0.904443382704
    Validation Accuracy at 78: 0.900621118012
    Validation Accuracy at 79: 0.899307214525
    Validation Accuracy at 80: 0.904801720019
    Validation Accuracy at 81: 0.902054467272
    Validation Accuracy at 82: 0.902173913043
    Validation Accuracy at 83: 0.902890587673
    Validation Accuracy at 84: 0.898112756808
    Validation Accuracy at 85: 0.901815575729
    Validation Accuracy at 86: 0.901098901099
    Validation Accuracy at 87: 0.902532250358
    Validation Accuracy at 88: 0.903248924988
    Validation Accuracy at 89: 0.904682274247
    Validation Accuracy at 90: 0.904443382704
    Validation Accuracy at 91: 0.903129479216
    Validation Accuracy at 92: 0.904921165791
    Validation Accuracy at 93: 0.902771141902
    Validation Accuracy at 94: 0.903965599618
    Validation Accuracy at 95: 0.904323936933
    Validation Accuracy at 96: 0.902293358815
    Validation Accuracy at 97: 0.904682274247
    Validation Accuracy at 98: 0.904801720019
    Validation Accuracy at 99: 0.896201624462
    Validation Accuracy at 100: 0.897873865265
    Validation Accuracy at 101: 0.901457238414
    Validation Accuracy at 102: 0.901218346871
    Validation Accuracy at 103: 0.902412804587
    Validation Accuracy at 104: 0.900979455327
    Validation Accuracy at 105: 0.900740563784
    Validation Accuracy at 106: 0.899546106068
    Validation Accuracy at 107: 0.899187768753
    Validation Accuracy at 108: 0.898471094123
    Validation Accuracy at 109: 0.897873865265
    Validation Accuracy at 110: 0.900382226469
    Validation Accuracy at 111: 0.900979455327
    Validation Accuracy at 112: 0.899904443383
    Validation Accuracy at 113: 0.897873865265
    Validation Accuracy at 114: 0.89010989011
    Validation Accuracy at 115: 0.905279503106
    Validation Accuracy at 116: 0.894171046345
    Validation Accuracy at 117: 0.885093167702
    Validation Accuracy at 118: 0.885212613473
    Validation Accuracy at 119: 0.895126612518
    Validation Accuracy at 120: 0.903607262303
    Validation Accuracy at 121: 0.899187768753
    Validation Accuracy at 122: 0.896082178691
    Validation Accuracy at 123: 0.893573817487
    Validation Accuracy at 124: 0.900501672241
    Validation Accuracy at 125: 0.903726708075
    Validation Accuracy at 126: 0.905876731964
    Validation Accuracy at 127: 0.903248924988
    Validation Accuracy at 128: 0.901815575729
    Validation Accuracy at 129: 0.904204491161
    Validation Accuracy at 130: 0.904085045389
    Validation Accuracy at 131: 0.904801720019
    Validation Accuracy at 132: 0.905040611562
    Validation Accuracy at 133: 0.903965599618
    Validation Accuracy at 134: 0.905279503106
    Validation Accuracy at 135: 0.901218346871
    Validation Accuracy at 136: 0.905040611562
    Validation Accuracy at 137: 0.904562828476
    Validation Accuracy at 138: 0.903726708075
    Validation Accuracy at 139: 0.903010033445
    Validation Accuracy at 140: 0.900143334926
    Validation Accuracy at 141: 0.902771141902
    Validation Accuracy at 142: 0.897396082179
    Validation Accuracy at 143: 0.896559961777
    Validation Accuracy at 144: 0.904801720019
    Validation Accuracy at 145: 0.900501672241
    Validation Accuracy at 146: 0.894171046345
    Validation Accuracy at 147: 0.901576684185
    Validation Accuracy at 148: 0.897993311037
    Validation Accuracy at 149: 0.902293358815
    Validation Accuracy at 150: 0.905518394649
    Validation Accuracy at 151: 0.904204491161
    Validation Accuracy at 152: 0.900382226469
    Validation Accuracy at 153: 0.902532250358
    Validation Accuracy at 154: 0.902532250358
    Validation Accuracy at 155: 0.898351648352
    Validation Accuracy at 156: 0.900860009556
    Validation Accuracy at 157: 0.89751552795
    Validation Accuracy at 158: 0.900621118012
    Validation Accuracy at 159: 0.900979455327
    Validation Accuracy at 160: 0.904801720019
    Validation Accuracy at 161: 0.900501672241
    Validation Accuracy at 162: 0.899426660296
    Validation Accuracy at 163: 0.903607262303
    Validation Accuracy at 164: 0.898471094123
    Validation Accuracy at 165: 0.896082178691
    Validation Accuracy at 166: 0.896559961777
    Validation Accuracy at 167: 0.896798853321
    Validation Accuracy at 168: 0.900621118012
    Validation Accuracy at 169: 0.897157190635
    Validation Accuracy at 170: 0.898351648352
    Validation Accuracy at 171: 0.900382226469
    Validation Accuracy at 172: 0.904801720019
    Validation Accuracy at 173: 0.90707118968
    Validation Accuracy at 174: 0.907429526995
    Validation Accuracy at 175: 0.907668418538
    Validation Accuracy at 176: 0.905996177735
    Validation Accuracy at 177: 0.908026755853
    Validation Accuracy at 178: 0.900501672241
    Validation Accuracy at 179: 0.89823220258
    Validation Accuracy at 180: 0.896798853321
    Validation Accuracy at 181: 0.897873865265
    Validation Accuracy at 182: 0.895962732919
    Validation Accuracy at 183: 0.894290492117
    Validation Accuracy at 184: 0.894768275203
    Validation Accuracy at 185: 0.895484949833
    Validation Accuracy at 186: 0.896798853321
    Validation Accuracy at 187: 0.901696129957
    Validation Accuracy at 188: 0.905398948877
    Validation Accuracy at 189: 0.904801720019
    Validation Accuracy at 190: 0.902173913043
    Validation Accuracy at 191: 0.89894887721
    Validation Accuracy at 192: 0.89894887721
    Validation Accuracy at 193: 0.899665551839
    Validation Accuracy at 194: 0.89751552795
    Validation Accuracy at 195: 0.900740563784
    Validation Accuracy at 196: 0.903965599618
    Validation Accuracy at 197: 0.905160057334
    Validation Accuracy at 198: 0.908146201624
    Validation Accuracy at 199: 0.907907310081
    Validation Accuracy at 200: 0.90707118968
    Validation Accuracy at 201: 0.906593406593
    Validation Accuracy at 202: 0.906593406593
    Validation Accuracy at 203: 0.908862876254
    Validation Accuracy at 204: 0.903726708075
    Validation Accuracy at 205: 0.906473960822
    Validation Accuracy at 206: 0.905279503106
    Validation Accuracy at 207: 0.896559961777
    Validation Accuracy at 208: 0.901337792642
    Validation Accuracy at 209: 0.900501672241
    Validation Accuracy at 210: 0.900023889154
    Validation Accuracy at 211: 0.905160057334
    Validation Accuracy at 212: 0.902173913043
    Validation Accuracy at 213: 0.89894887721
    Validation Accuracy at 214: 0.899187768753
    Validation Accuracy at 215: 0.901696129957
    Validation Accuracy at 216: 0.900143334926
    Validation Accuracy at 217: 0.894171046345
    Validation Accuracy at 218: 0.899546106068
    Validation Accuracy at 219: 0.896559961777
    Validation Accuracy at 220: 0.893454371715
    Validation Accuracy at 221: 0.894768275203
    Validation Accuracy at 222: 0.892976588629
    Validation Accuracy at 223: 0.897037744864
    Validation Accuracy at 224: 0.89452938366
    Validation Accuracy at 225: 0.895007166746
    Validation Accuracy at 226: 0.893334925944
    Validation Accuracy at 227: 0.891901576684
    Validation Accuracy at 228: 0.893334925944
    Validation Accuracy at 229: 0.889632107023
    Validation Accuracy at 230: 0.895365504061
    Validation Accuracy at 231: 0.90265169613
    Validation Accuracy at 232: 0.896679407549
    Validation Accuracy at 233: 0.903129479216
    Validation Accuracy at 234: 0.904562828476
    Validation Accuracy at 235: 0.905040611562
    Validation Accuracy at 236: 0.90563784042
    Validation Accuracy at 237: 0.905398948877
    Validation Accuracy at 238: 0.905996177735
    Validation Accuracy at 239: 0.905398948877
    Validation Accuracy at 240: 0.905996177735
    Validation Accuracy at 241: 0.90635451505
    Validation Accuracy at 242: 0.907429526995
    Validation Accuracy at 243: 0.90635451505
    Validation Accuracy at 244: 0.905757286192
    Validation Accuracy at 245: 0.906593406593
    Validation Accuracy at 246: 0.905040611562
    Validation Accuracy at 247: 0.898351648352
    Validation Accuracy at 248: 0.897993311037
    Validation Accuracy at 249: 0.894768275203
    Validation Accuracy at 250: 0.898112756808
    Validation Accuracy at 251: 0.903487816531
    Validation Accuracy at 252: 0.898590539895
    Validation Accuracy at 253: 0.900382226469
    Validation Accuracy at 254: 0.897873865265
    Validation Accuracy at 255: 0.894768275203
    Validation Accuracy at 256: 0.893573817487
    Validation Accuracy at 257: 0.894409937888
    Validation Accuracy at 258: 0.897276636407
    Validation Accuracy at 259: 0.900143334926
    Validation Accuracy at 260: 0.900382226469
    Validation Accuracy at 261: 0.901696129957
    Validation Accuracy at 262: 0.903010033445
    Validation Accuracy at 263: 0.905398948877
    Validation Accuracy at 264: 0.904443382704
    Validation Accuracy at 265: 0.893932154802
    Validation Accuracy at 266: 0.888437649307
    Validation Accuracy at 267: 0.894171046345
    Validation Accuracy at 268: 0.896321070234
    Validation Accuracy at 269: 0.902771141902
    Validation Accuracy at 270: 0.90336837076
    Validation Accuracy at 271: 0.906712852365
    Validation Accuracy at 272: 0.906593406593
    Validation Accuracy at 273: 0.901696129957
    Validation Accuracy at 274: 0.893693263258
    Validation Accuracy at 275: 0.890707118968
    Validation Accuracy at 276: 0.891423793598
    Validation Accuracy at 277: 0.903129479216
    Validation Accuracy at 278: 0.901815575729
    Validation Accuracy at 279: 0.894887720975
    Validation Accuracy at 280: 0.904204491161
    Validation Accuracy at 281: 0.904323936933
    Validation Accuracy at 282: 0.903607262303
    Validation Accuracy at 283: 0.899904443383
    Validation Accuracy at 284: 0.9019350215
    Validation Accuracy at 285: 0.906951743908
    Validation Accuracy at 286: 0.906951743908
    Validation Accuracy at 287: 0.903607262303
    Validation Accuracy at 288: 0.903248924988
    Validation Accuracy at 289: 0.902173913043
    Validation Accuracy at 290: 0.900621118012
    Validation Accuracy at 291: 0.902054467272
    Validation Accuracy at 292: 0.90265169613
    Validation Accuracy at 293: 0.899546106068
    Validation Accuracy at 294: 0.902771141902
    Validation Accuracy at 295: 0.903487816531
    Validation Accuracy at 296: 0.897634973722
    Validation Accuracy at 297: 0.898590539895
    Validation Accuracy at 298: 0.900501672241
    Validation Accuracy at 299: 0.90336837076


Let's see how the accuracy evolved over time.


```python
y_min = 0.0
y_max = 1.0

sns.set(font_scale = 1.25)
sns.set_style("darkgrid")
f = plt.figure(figsize = (12, 8))
ax = plt.axes()
plt.title("SGD Logistic Regression Accuracy Evolution")
plt.xlabel("Iteration")
plt.ylabel("Accuracy")
plt.ylim(y_min, y_max)
plt.yticks(np.arange(y_min, y_max + .01, .10))
plt.plot([x[0] for x in validation_accuracy_list], [x[1] for x in validation_accuracy_list], 'green')
plt.show()
```


![png](output_20_0.png)


Very noisy (as expected). I'll take the average of every 10 iterations and plot that.


```python
iterations = [x[0] for x in validation_accuracy_list]
accuracies = [x[1] for x in validation_accuracy_list]

n = 10
accuracies_average = np.array(accuracies).reshape(-1, n).mean(axis = 1)
iterations_sampled = [x[0] for i, x in enumerate(validation_accuracy_list) if i % n == 0]
```


```python
y_min = 0.0
y_max = 1.0

sns.set(font_scale = 1.25)
sns.set_style("darkgrid")
f = plt.figure(figsize = (12, 8))
ax = plt.axes()
plt.title("SGD Logistic Regression Accuracy Evolution")
plt.xlabel("Iteration")
plt.ylabel("Accuracy")
plt.ylim(y_min, y_max)
plt.yticks(np.arange(y_min, y_max + .01, .10))
plt.plot(iterations_every5, accuracies_average, 'green')
plt.show()
```


![png](output_23_0.png)


90% accuracy! That's not bad for minimal pre-processing and a standard logistic regression. If we were smart, we might think that the willingness to give out stars varies across users. It makes sense to account for the fact that some people might systematically give mediocre reviews and 4 stars, while others might systematically write fantastic reviews and only give 4 stars. "Normalizing" the features to account for user history would almost definitely improve the model.

So why might predicting sentiment from reviews be useful? Maybe restaurants would pay for an automated service that tells them when a customer posts about a fantastic or terrible experience at their restaurant. I'd want to make sure to make as few mistakes as possible, so it would make sense to use only the reviews our model classifies above a certain probability of being positive or negative (maybe 95%). This is pretty straightforward with Logistic Regression.

Restaurants could get that kind of feedback by checking Yelp themselves, of course, but a weekly summary with the highlights and lowlights might save enough time and effort to be worth it.

# Concluding Thoughts and Online Learning

I only did 300 epochs of stochastic gradient descent. With a chunksize of 1000, that means the model only used 300,000 reviews. By that time, it was pretty clear the classifier was oscillating around 90% accuracy. We could have trained the model by reading in 300,000 reviews and doing a standard gradient descent. Why might the stochastic way be better (aside from being slightly faster)?

It's better because it lets us do online learning. Online learning is the way we can update our model in near real-time as we acquire more data. Data can come in all the time, and we don't want to train on the entire dataset every time it changes. Stochastic gradient descent let's us build our model in small batches and extremely accurately approximate the gradient descent solution. The flexibility of this approach more than makes up for the slightly more involved coding process when you have big and rapidly increasing data.
