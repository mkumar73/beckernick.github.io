---
title:  "Clustering Poetry using TF-IDF"
date:   2016-08-17
categories: [data science, poetry, clustering, natural language processing]
tags: [data science, poetry, natural langauge processing]
header:
  image: "i_love_poetry.jpg"
---


# Clustering Poetry by Textual Similarity


```python
import bs4
import urllib2
import re
import time
import pickle
import pandas as pd
import numpy as np


### Clustering great poems
poem_list_url = 'http://poemhunter.com/p/m/l.asp?a=0&l=top500&order=title&p='

# store the relevant tags in a list to avoid scraping the site too much
relevant_links = []

for i in xrange(1, 21):
    print 'Page {0}'.format(i)
    
    response = urllib2.urlopen(poem_list_url + str(i))
    html = response.read()
    soup = bs4.BeautifulSoup(html)
    tags = soup.find('table', {'class' : 'poems-listing'})
    tr_list = tags.find_all('tr')
    #print tr_list
    # 25 poems per page
    time.sleep(3)
    for j in xrange(1, 26):
        #print j
        temp = tr_list[j].find('td', {'class' : 'title'})
        #print temp.find_all('a'), '\n'
        
        hrefs = temp.find_all('a')
        hrefs = [str(x) for x in hrefs]
        
        relevant_links.append(hrefs)
        



poetry_tuples_list = []

for i in range(len(relevant_links)):
    #print i
    #print relevant_links[i], '\n'
    
    title_start = relevant_links[i][0].find('title="') + len('title="')
    title_end = relevant_links[i][0].find('poem">')
    title = relevant_links[i][0][title_start:title_end - 1]
    
    author_start = relevant_links[i][1].find('title="') + len('title="')
    author_end = relevant_links[i][1].find('poet">')
    author = relevant_links[i][1][author_start:author_end - 1]    
    
    href_start = relevant_links[i][0].find('href="') + len('href="')
    href_end = relevant_links[i][0].find('/"')
    
    poem_href = relevant_links[i][0][href_start:href_end + 1]
    
    poetry_tuple = (title, author, poem_href)
    
    print poetry_tuple, '\n'
    poetry_tuples_list.append(poetry_tuple)
    


# Get the text of the poems
poem_base_url = 'http://poemhunter.com'
poetry_dictionary = {}

for i in range(len(poetry_tuples_list)):
    print poetry_tuples_list[i][0]
    
    response = urllib2.urlopen(poem_base_url + poetry_tuples_list[i][2])
    html = response.read()
    soup = bs4.BeautifulSoup(html)
    tags = soup.find('div', {'class' : 'KonaBody'})
    text = tags.find('p')
    
    for br in text.find_all('br'):
        br.replace_with('\n')

    text = text.decode().encode('utf-8')
    
    text = re.sub('</p>', '', text)
    text = re.sub('<p>', '', text)
    text = text.strip()
        
    poetry_dictionary[poetry_tuples_list[i][0]] = {'author' : poetry_tuples_list[i][1],
                                            'text' : text}
    
    # Save every 10 iterations
    if (i + 1) % 10 == 0:
        with open('/users/nickbecker/Python_Projects/great_poetry/poetry_dict_{0}.pickle'.format(i+1), 'wb') as handle:
            pickle.dump(poetry_dictionary, handle)
            
    time.sleep(4)


with open('/users/nickbecker/Python_Projects/great_poetry/poetry_dict_full.pickle', 'wb') as handle:
    pickle.dump(poetry_dictionary, handle)
    



# Compute tf-idf vecors using sklearn and nltk
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.stem.porter import PorterStemmer
import string


with open('/users/nickbecker/Python_Projects/great_poetry/poetry_dict_full.pickle', 'r') as handle:
    poetry_dictionary = pickle.load(handle)


# Replace '\n' with spaces in the dictionary, remove punctuation, make lowercase
for key in poetry_dictionary.keys():
    print key
    poetry_dictionary[key]['text'] = re.sub('\n', ' ', poetry_dictionary[key]['text'])
    poetry_dictionary[key]['text'] = poetry_dictionary[key]['text'].lower()
    poetry_dictionary[key]['text'] = poetry_dictionary[key]['text'].translate(None, string.punctuation)

    if poetry_dictionary[key]['text'] == 'font colorredb the text of this poem could not be published because of copyright laws bfont':
        del poetry_dictionary[key]

token_dict = {}
stemmer = PorterStemmer()

def stem_tokens(tokens, stemmer):
    stemmed = []
    for item in tokens:
        stemmed.append(stemmer.stem(item))
    return stemmed

def tokenize(text):
    tokens = nltk.word_tokenize(text)
    stems = stem_tokens(tokens, stemmer)
    return stems

for key in poetry_dictionary.keys():
    text_clean = re.sub('\n', ' ', poetry_dictionary[key]['text'])
    text_clean = text_clean.lower()
    text_clean = text_clean.translate(None, string.punctuation)
    token_dict[key] = text_clean
    

tokenize(token_dict['Her Voice'])


tfidf = TfidfVectorizer(tokenizer=tokenize, stop_words='english')
tfs = tfidf.fit_transform(token_dict.values())
tf_idfs = dict(zip(tfidf.get_feature_names(), tfidf.idf_))

from sklearn.metrics.pairwise import cosine_similarity

z = cosine_similarity(tfs[0, :], tfs).flatten()

pairwise_similarity = (tfs * tfs.T).A
y = pd.DataFrame(pairwise_similarity, index = token_dict.keys(), columns = token_dict.keys())

# How "tight" is this set of poems"?
distances_from_mean = [np.linalg.norm(pairwise_similarity.mean(axis = 0) - pairwise_similarity[:, col]) for col in range(len(pairwise_similarity))]
document_set_tightness = np.mean(distances_from_mean)


# Find closest poem to a query poem
def find_closest_k_poems_indices_scores(poem_tfidf, overall_tfidf, k):
    cosine_sim = cosine_similarity(poem_tfidf, overall_tfidf).flatten()
    indices = np.argsort(cosine_sim)[-(k+1):-1]
    return indices, cosine_sim[indices]

def get_names_of_close_poems(indices):
    lst =  [token_dict.keys()[x] for x in indices]
    lst.reverse() # closest match decending order
    return lst



## Closest poems dictionary

closest_poems_dict = {}
k = 3
for i in xrange(len(token_dict.keys())):
    print token_dict.keys()[i], i
    
    closest_indices, cosine_sims = find_closest_k_poems_indices_scores(tfs[i, :], tfs, k)
    closest_poems = get_names_of_close_poems(closest_indices)
    closest_poems_dict[token_dict.keys()[i]] = zip(closest_poems, cosine_sims)


z = y.loc['Expect Nothing', :]


close_match_dict = {}
threshold = 0.25
for key, values in closest_poems_dict.items():
    #print key, values
    closest = [x for x in values if x[1] >= threshold]
    if closest:
        close_match_dict[key] = closest

with open('/users/nickbecker/Python_Projects/great_poetry/closest_poems_dict.pickle', 'wb') as handle:
    pickle.dump(closest_poems_dict, handle)
    




```


```python

```


```python

```
