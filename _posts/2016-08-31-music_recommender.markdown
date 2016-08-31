---
title:  "Music Recommendations with Collaborative Filtering and K-Nearest Neighbors"
date:   2016-08-31
tags: [data science]

header:
  image: "last_fm_spotify_edited.png"
  caption: "Photo credit: [**Slash Gear**](http://www.slashgear.com/last-fm-and-spotify-team-up-to-give-better-music-recommendations-29315027/)"

excerpt: "Code Focused. Music Recommender, Collaborative Filtering, K-Nearest Neighbors"
---




# Last-FM 360K data


```python
import pandas as pd
from scipy.spatial.distance import cosine
import numpy as np
from scipy.sparse import csr_matrix

# display results to 3 decimal points, not in scientific notation
pd.set_option('display.float_format', lambda x: '%.3f' % x)
```


```python
user_data = pd.read_table('/users/nickbecker/Downloads/lastfm-dataset-360K/usersha1-artmbid-artname-plays.tsv',
                          header = None, nrows = 2e7,
                          names = ['users', 'musicbrainz-artist-id', 'artist-name', 'plays'],
                          usecols = ['users', 'artist-name', 'plays'])
```


```python
print user_data.shape
user_data.head()
```

    (17535655, 3)





<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>users</th>
      <th>artist-name</th>
      <th>plays</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>00000c289a1829a808ac09c00daf10bc3c4e223b</td>
      <td>betty blowtorch</td>
      <td>2137</td>
    </tr>
    <tr>
      <th>1</th>
      <td>00000c289a1829a808ac09c00daf10bc3c4e223b</td>
      <td>die Ärzte</td>
      <td>1099</td>
    </tr>
    <tr>
      <th>2</th>
      <td>00000c289a1829a808ac09c00daf10bc3c4e223b</td>
      <td>melissa etheridge</td>
      <td>897</td>
    </tr>
    <tr>
      <th>3</th>
      <td>00000c289a1829a808ac09c00daf10bc3c4e223b</td>
      <td>elvenking</td>
      <td>717</td>
    </tr>
    <tr>
      <th>4</th>
      <td>00000c289a1829a808ac09c00daf10bc3c4e223b</td>
      <td>juliette &amp; the licks</td>
      <td>706</td>
    </tr>
  </tbody>
</table>
</div>



Are any artist names NaN? If so, remove rows where the artist name is missing


```python
print user_data['artist-name'].isnull().sum()
user_data = user_data.dropna(axis = 0, subset = ['artist-name'])
```

    2


User profile data


```python
user_profiles = pd.read_table('/users/nickbecker/Downloads/lastfm-dataset-360K/usersha1-profile.tsv',
                          header = None,
                          names = ['users', 'gender', 'age', 'country', 'signup'],
                          usecols = ['users', 'country'])
```


```python
print user_profiles.shape
user_profiles.head()
```

    (359347, 2)





<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>users</th>
      <th>country</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>00000c289a1829a808ac09c00daf10bc3c4e223b</td>
      <td>Germany</td>
    </tr>
    <tr>
      <th>1</th>
      <td>00001411dc427966b17297bf4d69e7e193135d89</td>
      <td>Canada</td>
    </tr>
    <tr>
      <th>2</th>
      <td>00004d2ac9316e22dc007ab2243d6fcb239e707d</td>
      <td>Germany</td>
    </tr>
    <tr>
      <th>3</th>
      <td>000063d3fe1cf2ba248b9e3c3f0334845a27a6bf</td>
      <td>Mexico</td>
    </tr>
    <tr>
      <th>4</th>
      <td>00007a47085b9aab8af55f52ec8846ac479ac4fe</td>
      <td>United States</td>
    </tr>
  </tbody>
</table>
</div>



## Popular artists only


```python
artist_plays = (user_data.
     groupby(by = ['artist-name'])['plays'].
     sum().
     reset_index().
     rename(columns = {'plays': 'total_artist_plays'})
     [['artist-name', 'total_artist_plays']]
    )

```


```python
print artist_plays.shape[0]
print artist_plays.head()
```

    292364
                                     artist-name  total_artist_plays
    0                                       04)]                   6
    1                                         2                 1606
    2                                  58725ab=>                  23
    3   80lİ yillarin tÜrkÇe sÖzlÜ aŞk Şarkilari                  70
    4                              amy winehouse                  23



```python
user_data_with_artist_plays = user_data.merge(artist_plays, left_on = 'artist-name', right_on = 'artist-name', how = 'left')
```


```python
print user_data_with_artist_plays.shape
user_data_with_artist_plays.head()
```

    (17535653, 4)





<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>users</th>
      <th>artist-name</th>
      <th>plays</th>
      <th>total_artist_plays</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>00000c289a1829a808ac09c00daf10bc3c4e223b</td>
      <td>betty blowtorch</td>
      <td>2137</td>
      <td>25651</td>
    </tr>
    <tr>
      <th>1</th>
      <td>00000c289a1829a808ac09c00daf10bc3c4e223b</td>
      <td>die Ärzte</td>
      <td>1099</td>
      <td>3704875</td>
    </tr>
    <tr>
      <th>2</th>
      <td>00000c289a1829a808ac09c00daf10bc3c4e223b</td>
      <td>melissa etheridge</td>
      <td>897</td>
      <td>180391</td>
    </tr>
    <tr>
      <th>3</th>
      <td>00000c289a1829a808ac09c00daf10bc3c4e223b</td>
      <td>elvenking</td>
      <td>717</td>
      <td>410725</td>
    </tr>
    <tr>
      <th>4</th>
      <td>00000c289a1829a808ac09c00daf10bc3c4e223b</td>
      <td>juliette &amp; the licks</td>
      <td>706</td>
      <td>90498</td>
    </tr>
  </tbody>
</table>
</div>



## Picking a threshold for popular artists

With nearly 300,000 different artists, it's almost a guarantee most artists have been played only a few times. Let's look at the descriptive statistics.


```python
print artist_plays['total_artist_plays'].describe()
```

    count     292364.000
    mean       12907.037
    std       185981.313
    min            1.000
    25%           53.000
    50%          208.000
    75%         1048.000
    max     30466827.000
    Name: total_artist_plays, dtype: float64


As expected, the median artist has only been played about 200 times. Let's take a look at the top of the distribution.


```python
print artist_plays['total_artist_plays'].quantile(np.arange(.75, 1, .05))
```

    0.750    1048.000
    0.800    1651.000
    0.850    2909.000
    0.900    6138.000
    0.950   19964.250
    Name: total_artist_plays, dtype: float64



```python
print artist_plays['total_artist_plays'].quantile(np.arange(.9, 1, .01)), 
```

    0.900     6138.000
    0.910     7410.000
    0.920     9102.960
    0.930    11475.590
    0.940    14898.440
    0.950    19964.250
    0.960    28419.880
    0.970    43541.330
    0.980    79403.440
    0.990   198482.590
    Name: total_artist_plays, dtype: float64



```python
print artist_plays['total_artist_plays'].quantile(np.arange(.9, 1, .01))
```

    0.900     6138.000
    0.910     7410.000
    0.920     9102.960
    0.930    11475.590
    0.940    14898.440
    0.950    19964.250
    0.960    28419.880
    0.970    43541.330
    0.980    79403.440
    0.990   198482.590
    Name: total_artist_plays, dtype: float64


So about 1% of artists have roughly 200,000 or more listens, 2% have 80,000 or more, and 3% have 40,000 or more. Since we have so many artists, we'll limit it to the top 3%. We won't be able to recommend lesser known artists, but the computational burden will decrease significantly. Besides, roughly 9000 artists is a pretty good number anyway.


```python
popularity_threshold = 40000

user_data_popular_artists = user_data_with_artist_plays.query('total_artist_plays >= @popularity_threshold')
user_data_popular_artists.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>users</th>
      <th>artist-name</th>
      <th>plays</th>
      <th>total_artist_plays</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>00000c289a1829a808ac09c00daf10bc3c4e223b</td>
      <td>die Ärzte</td>
      <td>1099</td>
      <td>3704875</td>
    </tr>
    <tr>
      <th>2</th>
      <td>00000c289a1829a808ac09c00daf10bc3c4e223b</td>
      <td>melissa etheridge</td>
      <td>897</td>
      <td>180391</td>
    </tr>
    <tr>
      <th>3</th>
      <td>00000c289a1829a808ac09c00daf10bc3c4e223b</td>
      <td>elvenking</td>
      <td>717</td>
      <td>410725</td>
    </tr>
    <tr>
      <th>4</th>
      <td>00000c289a1829a808ac09c00daf10bc3c4e223b</td>
      <td>juliette &amp; the licks</td>
      <td>706</td>
      <td>90498</td>
    </tr>
    <tr>
      <th>5</th>
      <td>00000c289a1829a808ac09c00daf10bc3c4e223b</td>
      <td>red hot chili peppers</td>
      <td>691</td>
      <td>13547741</td>
    </tr>
  </tbody>
</table>
</div>




```python
# print the unique popular artists
unique_popular_artist = sorted(set(user_data_popular_artists['artist-name'].tolist()))
print len(unique_popular_artist)
```

    9300


So there are about 9000 artists with more than 40,000 listens. This is by design, since we just took roughly 3% of about 300,000 artists

## Filtering to US Users Only

Since I'm in Washington, D.C., I'll limit the user data to just those from the United States. First, I'll merge in the user profile data that has the user's country. Then I'll filter the data to only users in the United States.

```python
combined = user_data_popular_artists.merge(user_profiles, left_on = 'users', right_on = 'users', how = 'left')
```


```python
combined.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>users</th>
      <th>artist-name</th>
      <th>plays</th>
      <th>total_artist_plays</th>
      <th>country</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>00000c289a1829a808ac09c00daf10bc3c4e223b</td>
      <td>die Ärzte</td>
      <td>1099</td>
      <td>3704875</td>
      <td>Germany</td>
    </tr>
    <tr>
      <th>1</th>
      <td>00000c289a1829a808ac09c00daf10bc3c4e223b</td>
      <td>melissa etheridge</td>
      <td>897</td>
      <td>180391</td>
      <td>Germany</td>
    </tr>
    <tr>
      <th>2</th>
      <td>00000c289a1829a808ac09c00daf10bc3c4e223b</td>
      <td>elvenking</td>
      <td>717</td>
      <td>410725</td>
      <td>Germany</td>
    </tr>
    <tr>
      <th>3</th>
      <td>00000c289a1829a808ac09c00daf10bc3c4e223b</td>
      <td>juliette &amp; the licks</td>
      <td>706</td>
      <td>90498</td>
      <td>Germany</td>
    </tr>
    <tr>
      <th>4</th>
      <td>00000c289a1829a808ac09c00daf10bc3c4e223b</td>
      <td>red hot chili peppers</td>
      <td>691</td>
      <td>13547741</td>
      <td>Germany</td>
    </tr>
  </tbody>
</table>
</div>




```python
usa_data = combined.query('country == \'United States\'')
usa_data.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>users</th>
      <th>artist-name</th>
      <th>plays</th>
      <th>total_artist_plays</th>
      <th>country</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>156</th>
      <td>00007a47085b9aab8af55f52ec8846ac479ac4fe</td>
      <td>devendra banhart</td>
      <td>456</td>
      <td>2366807</td>
      <td>United States</td>
    </tr>
    <tr>
      <th>157</th>
      <td>00007a47085b9aab8af55f52ec8846ac479ac4fe</td>
      <td>boards of canada</td>
      <td>407</td>
      <td>6115545</td>
      <td>United States</td>
    </tr>
    <tr>
      <th>158</th>
      <td>00007a47085b9aab8af55f52ec8846ac479ac4fe</td>
      <td>cocorosie</td>
      <td>386</td>
      <td>2194862</td>
      <td>United States</td>
    </tr>
    <tr>
      <th>159</th>
      <td>00007a47085b9aab8af55f52ec8846ac479ac4fe</td>
      <td>aphex twin</td>
      <td>213</td>
      <td>4248296</td>
      <td>United States</td>
    </tr>
    <tr>
      <th>160</th>
      <td>00007a47085b9aab8af55f52ec8846ac479ac4fe</td>
      <td>animal collective</td>
      <td>203</td>
      <td>3495537</td>
      <td>United States</td>
    </tr>
  </tbody>
</table>
</div>



Before doing any analysis, we should make sure dataset is internally consistent. Every user should only have a play count variable once for each artist. So we'll check for instances where rows have the same `users` and `artist-name` values.


```python
# Make sure there are no duplicate user/artist rows with different playcounts

if not usa_data[usa_data.duplicated(['users', 'artist-name'])].empty:
    initial_rows = usa_data.shape[0]
    
    print 'Initial dataframe shape {0}'.format(usa_data.shape)
    usa_data = usa_data.drop_duplicates(['users', 'artist-name'])
    current_rows = usa_data.shape[0]
    print 'New dataframe shape {0}'.format(usa_data.shape)
    print 'Removed {0} rows'.format(initial_rows - current_rows)

```

    Initial dataframe shape (2788019, 5)
    New dataframe shape (2788013, 5)
    Removed 6


## Reshaping the data for K-Nearest Neighbors

For K-Nearest Neighbors, we want the data to be in an `(artist, user)` array, where each row is an artist and each column is a user. To reshape the dataframe, we'll `pivot` the dataframe to a wide format. Then we'll fill the missing observations with `0`s since we're going to be performing linear algebra operations (calculating distances between vectors). Finally, we transform the values of the dataframe into a scipy sparse matrix for more efficient calculations.


```python
wide_artist_data = usa_data.pivot(index = 'artist-name', columns = 'users', values = 'plays').fillna(0)
wide_artist_data_sparse = csr_matrix(wide_artist_data.values)
```


```python
print type(wide_artist_data_sparse)
print wide_artist_data_sparse.shape
```

    <class 'scipy.sparse.csr.csr_matrix'>
    (9127, 66913)


## Implemeting the Nearest Neighbor Model

Time to implement the model. We'll initialize the `NearestNeighbors` class as `model_knn` and `fit` our sparse matrix to the instance. By specifying the `metric = cosine`, the model will measure similarity bectween artist vectors by using cosine similarity.


```python
from sklearn.neighbors import NearestNeighbors

model_knn = NearestNeighbors(metric = 'cosine', algorithm = 'brute')
model_knn.fit(wide_artist_data_sparse)
```




    NearestNeighbors(algorithm='brute', leaf_size=30, metric='cosine',
             metric_params=None, n_jobs=1, n_neighbors=5, p=2, radius=1.0)



And we're ready to make some recommendations!


```python
query_index = np.random.choice(wide_data.shape[0])
print query_index

distances, indices = model_knn.kneighbors(wide_data.iloc[query_index, :].reshape(1, -1), n_neighbors = 6)

for i in range(0, len(distances.flatten())):
    if i == 0:
        print 'Recommendations for {0}:\n'.format(wide_data.index[query_index])
    else:
        print '{0}: {1}, with distance of {2}:'.format(i, wide_data.index[indices.flatten()[i]], distances.flatten()[i])
```

    8374
    Recommendations for tony bennett:
    
    1: frank sinatra, with distance of 0.226917809755:
    2: keiko matsui, with distance of 0.39397227453:
    3: andy williams, with distance of 0.477163884119:
    4: chic, with distance of 0.488077997533:
    5: cherry poppin daddies, with distance of 0.4908909547:


Pretty good! Frank Sinatra and Andy Williams are obviously good recommendations. I'd never heard of [Keiko Matsui](https://www.youtube.com/watch?v=XzqsWxau_gI) or [Cherry Poppin Daddies](https://www.youtube.com/watch?v=1IqH3uliwJY), but they both seem like good recommendations after listening to their music. [Chic](https://www.youtube.com/watch?v=eKl6EZShaaw), though, doesn't seem as similar to me as the other artists (they sound a little more disco).

Why would our model recommend Chic? Since we're doing item-based collaborative filtering with K-Nearest Neighbors on the actual play count data, outliers can have a big influence. If a few users listened to Tony Bennett and Chic a _**ton**_, our distance metric between vectors will be heavily influenced by those individual observations.

So is this good? Maybe. Depending on our goal, we might want **super-users** to have disproportionate weight in the distance calculation. But could we represent the data differently to avoid this feature?

## Binary Play Count Data

Previously, we used the actual play counts as values in our artist vectors. Another approach would be convert each vector into a binary (1 or 0): either a user played the song or they did not. We can do this by applying the `sign` function in `numpy` to each column in the dataframe.


```python
wide_data_zero_one = wide_data.apply(np.sign)
wide_data_zero_one_sparse = csr_matrix(wide_data_zero_one.values)
```


```python
model_nn_binary = NearestNeighbors(metric='cosine', algorithm='brute')
model_nn_binary.fit(wide_data_zero_one_sparse)
```




    NearestNeighbors(algorithm='brute', leaf_size=30, metric='cosine',
             metric_params=None, n_jobs=1, n_neighbors=5, p=2, radius=1.0)



Let's make a quick comparison. Which recommendations for Tony Bennett look better?


```python
query_index = np.random.choice(wide_data_zero_one.shape[0])
query_index = 8374 # tony bennett

distances, indices = model_nn_binary.kneighbors(wide_data_zero_one.iloc[query_index, :].reshape(1, -1), n_neighbors = 6)

#print query_index

for i in range(0, len(distances.flatten())):
    if i == 0:
        print 'Recommendations with binary play data for {0}:\n'.format(wide_data_zero_one.index[query_index])
    else:
        print '{0}: {1}, with distance of {2}:'.format(i, wide_data_zero_one.index[indices.flatten()[i]], distances.flatten()[i])
```

    Recommendations with binary play data for tony bennett:
    
    1: nat king cole, with distance of 0.771590841231:
    2: dean martin, with distance of 0.791135426625:
    3: frank sinatra, with distance of 0.815388695965:
    4: bobby darin, with distance of 0.818033228367:
    5: doris day, with distance of 0.81859043384:


These are great, too. At least for Tony Bennett, the binary data representation recommendations looks just as good. Someone who likes Tony Bennett might also like Nat King Cole or Frank Sinatra. The distances are higher, but that's due to squashing the data by using the sign function. Again, it's not obvious which method is better. Since ultimately it's the listeners's future actions that indicate which recommender system is better, it's a great candidate for A/B Testing. For now, I'll stick with the binary data representation model.

### Recommending Artists with Fuzzy Matching

Previously we picked query artists at random. But really, we want to make recommendations for a specific artist on command. Since some artists's names are ambiguous or commonly mispelled, we'll include fuzzy matching part in the process so we don't need exact name matches.

So we can do this anytime we want, we'll define a function `print_artist_recommendations` to do it.


```python
import string
from fuzzywuzzy import fuzz
```


```python
def print_artist_recommendations(query_artist, artist_plays_matrix, knn_model, k):
    """
    Inputs:
    query_artist: query artist name
    artist_plays_matrix: artist play count dataframe (not the sparse one, our previously fitted model)
    knn_model: a  previously fitted sklearn knn model
    k: the number of nearest neighbors.
    
    Prints: Artist recommendations for the query artist
    Returns: None
    """
    
    from operator import itemgetter

    query_index = None
    ratio_tuples = []
    
    for i in artist_plays_matrix.index:
        ratio = fuzz.ratio(i.lower(), query_artist.lower())
        if ratio >= 75:
            current_query_index = artist_plays_matrix.index.tolist().index(i)
            ratio_tuples.append((i, ratio, current_query_index))
    
    print 'Possible matches: {0}\n'.format([(x[0], x[1]) for x in ratio_tuples])
    
    try:
        query_index = max(ratio_tuples, key = lambda x: x[1])[2] # get the index of the best artist match in the data
    except:
        print 'Your artist didn\'t match any artists in the data. Try again'
        return None
    
    distances, indices = knn_model.kneighbors(artist_plays_matrix.iloc[query_index, :].reshape(1, -1), n_neighbors = k + 1)

    for i in range(0, len(distances.flatten())):
        if i == 0:
            print 'Recommendations for {0}:\n'.format(artist_plays_matrix.index[query_index])
        else:
            print '{0}: {1}, with distance of {2}:'.format(i, artist_plays_matrix.index[indices.flatten()[i]], distances.flatten()[i])

    return None
```

Time to try a few sample bands


```python
print_artist_recommendations('red hot chili peppers', wide_data_zero_one, model_nn_binary, k = 10)
```

    Possible matches: [('red hot chili peppers', 100)]
    
    Recommendations for red hot chili peppers:
    
    1: incubus, with distance of 0.686632912166:
    2: the beatles, with distance of 0.693856742888:
    3: sublime, with distance of 0.70540037526:
    4: foo fighters, with distance of 0.71155686859:
    5: coldplay, with distance of 0.716691422348:
    6: led zeppelin, with distance of 0.722488787624:
    7: nirvana, with distance of 0.724943983169:
    8: green day, with distance of 0.734603813118:
    9: radiohead, with distance of 0.737372302802:
    10: rage against the machine, with distance of 0.740136491957:



```python
print_artist_recommendations('arctic monkeys', wide_data_zero_one, model_nn_binary, k = 10)
```

    Possible matches: [('arctic monkeys', 100)]
    
    Recommendations for arctic monkeys:
    
    1: the strokes, with distance of 0.746696592481:
    2: the kooks, with distance of 0.767492571954:
    3: bloc party, with distance of 0.772120302741:
    4: franz ferdinand, with distance of 0.774566073856:
    5: the killers, with distance of 0.807176759929:
    6: radiohead, with distance of 0.812762633074:
    7: the fratellis, with distance of 0.814611330462:
    8: kings of leon, with distance of 0.815408152181:
    9: the beatles, with distance of 0.815680085574:
    10: the white stripes, with distance of 0.81607343278:



```python
print_artist_recommendations('u2', wide_data_zero_one, model_nn_binary, k = 10)
```

    Possible matches: [('u2', 100)]
    
    Recommendations for u2:
    
    1: r.e.m., with distance of 0.690057376797:
    2: coldplay, with distance of 0.697303504846:
    3: the beatles, with distance of 0.726085401263:
    4: the police, with distance of 0.756131589948:
    5: radiohead, with distance of 0.77692434746:
    6: pearl jam, with distance of 0.778566201394:
    7: the rolling stones, with distance of 0.782771546531:
    8: led zeppelin, with distance of 0.788269370325:
    9: dave matthews band, with distance of 0.788520439902:
    10: bruce springsteen, with distance of 0.789619233997:



```python
print_artist_recommendations('dispatch', wide_data_zero_one, model_nn_binary, k = 10)
```

    Possible matches: [('dispatch', 100)]
    
    Recommendations for dispatch:
    
    1: state radio, with distance of 0.665125909027:
    2: o.a.r., with distance of 0.749067403207:
    3: jack johnson, with distance of 0.778821492779:
    4: dave matthews band, with distance of 0.792342999987:
    5: guster, with distance of 0.821963512261:
    6: ben harper, with distance of 0.824748164332:
    7: slightly stoopid, with distance of 0.837663503717:
    8: the john butler trio, with distance of 0.841162765581:
    9: donavon frankenreiter, with distance of 0.841397926422:
    10: sublime, with distance of 0.846795058358:



```python
print_artist_recommendations('pearl jam', wide_data_zero_one, model_nn_binary, k = 10)
```

    Possible matches: [('pearl jam', 100)]
    
    Recommendations for pearl jam:
    
    1: stone temple pilots, with distance of 0.706407198885:
    2: alice in chains, with distance of 0.714291500864:
    3: nirvana, with distance of 0.724837631157:
    4: soundgarden, with distance of 0.728456327455:
    5: foo fighters, with distance of 0.741056843206:
    6: the smashing pumpkins, with distance of 0.751413106905:
    7: red hot chili peppers, with distance of 0.759106030065:
    8: led zeppelin, with distance of 0.76954203648:
    9: u2, with distance of 0.778566201394:
    10: radiohead, with distance of 0.793734608023:


To be brief, these are fantastic. 

## Future Ideas

#### Scaling up to Massive Datasets

Since we're using K-Nearest Neighbors, we have to calculate the distance of each artist vector in our `wide_artist_data_sparse` array to the query artist vector every time we make a query. If our data is fairly small (like in this post), this isn't an issue.

If we had the entirety of Last.fm's user data, we'd be bottlenecked like crazy at query time. Fortunately, there's been great work done on Approximate Nearest Neighbor Search techniques such as [locality sensitive hashing](http://www.mit.edu/~andoni/LSH/). These techniques sacrifice the **guarantee** of finding the nearest neighbors for  increases in computational efficiency, and work extremely well with high dimensional data. The [Machine Learning: Clustering & Retrieval](https://www.coursera.org/learn/ml-clustering-and-retrieval) course on Coursera has a great walk-through of LSH for those curious.

#### Recommending less popular artists

While our recommendation engine is doing a great job, it's only recommending popular artists (by design). A really cool alternative recommender might recommend us unknown artists given a query artist so we can discover new music. 

Recommending lesser known artists is a huge challenge that does fit well with collaborative filtering, so we might want to incorporate feature based recommendations into such a system.
