---
title:  "Identifying Similar US Laws using TF-IDF and K-Nearest Neighbors"
date:   2016-08-18
tags: [data science, natural langauge processing]
header:
  image: "congress_edge_detection.jpg"
---

It's 100 degrees in the nation's capitol (and it feels like 114), so what else is there to do but drink beer, listen to music, and write code? [Golden Monkey in hand](https://www.beeradvocate.com/beer/profile/345/1003/), I sat down on my couch and promptly watched the Olympics for three hours with my roommates. Eeventually, I got to work.

Since I'm doing some natural language processing at work, I figured I might as well write a post about NLP in Python. Inspired by the view of The Capitol from my apartment's window, I decided to use US laws for this post. Unfortuantely, I couldn't find a repository with the text of every US law, so I decided to create it myself by scraping laws from [www.congress.gov](https://www.congress.gov/legislation).



### Scraping www.congress.gov with BeautifulSoup
First, I needed to scrape https://www.congress.gov/legislation)to get the links for each bill. Each page has 25 bills on it, among other things. The URL ending of ```?q=%7B"bill-status"%3A"law"%7D``` filters the results to only be enacted bills (bills that became law). By looking at a few of the pages, I noticed that the hyperlinks I need are essentially in the same place on every page (inside ```<h2>``` within an ```<ol>``` tag of class ```results_list```).

So I can scrape all the hyperlinks with a nested loop. The outer loop grabs the data in the table on each page, and the inner loop extracts the hyperlinks for bills. Out of respect for Congress's servers, I store the links in a list and write the list to a text file so I don't have to scrape them again.


```python
from __future__ import division
import re
import nltk
from nltk.corpus import stopwords
import bs4
import urllib2
import time
import pickle
```


```python
# Get the hyperlinks for all of the bills
base_law_url = 'https://www.congress.gov/legislation?q=%7B"bill-status"%3A"law"%7D&page='
relevant_links = []
data_path = '~/Python_Projects/clustering_laws/'

for i in xrange(1, 150):
    print 'Page {0}'.format(i)
    
    response = urllib2.urlopen(base_law_url + str(i))
    html = response.read()
    soup = bs4.BeautifulSoup(html)
    tags = soup.find('ol', {'class' : 'results_list'})
    h2_list = tags.find_all('h2')
    
    for j in xrange(0, len(h2_list)):
        #print j
        href = h2_list[j].find('a').decode().encode('utf-8')
        
        m = re.search('href="(.+)">', href)
        if m:
            relevant_links.append(m.group(1))

with open(data_path + 'law_links.txt', 'w') as handle:
    for link in relevant_links:
        handle.write("%s\n" % link)
```

    Page 1
    Page 2
    Page 3
    [...]


With the hyperlinks to few thousand enacted bills in relevant_links, I next need to download the actual bills. By looking at a few of the pages, I noticed that access to the text version of the bills (instead of the PDF) is controlled by the URL ending ```/text?format=txt```. On each page, the text stored in a ```<pre>``` tag with id ```billTextContainer```.

By looping through ```relevant_links```, I grab the title and text of each bill and store them in ```titles_list``` and ```bills_text_list```. I'll zip them into a the dictionary ```bills_dictionary``` with the titles as keys and text as values.


```python
# Get the text and titles of the bills
titles_list = []
bills_text_list = []
text_ending = '/text?format=txt'

html_tags_re = re.compile(r'<[^>]+>')
def remove_html_tags(text):
    return html_tags_re.sub('', text)


for i in xrange(len(relevant_links)):
    print i
    response = urllib2.urlopen(relevant_links[i] + text_ending)
    html = response.read()
    soup = bs4.BeautifulSoup(html)
    
    # text
    text_tags = soup.find('pre', {'id' : 'billTextContainer'})
    
    if text_tags:
        text = text_tags.decode().encode('utf-8')
        text = remove_html_tags(text)
        bills_text_list.append(text)
        
        # titles
        title_soup = soup.find('h1', {'class' : 'legDetail'})
        title_soup = title_soup.decode().encode('utf-8')
        st = title_soup.find('">') + len('">')
        end = title_soup.find('<span>')
        
        titles_list.append(title_soup[st:end])

bills_dictionary = dict(zip(titles_list, bills_text_list))

with open(data_path + 'bills_dictionary.pickle', 'wb') as handle:
    pickle.dump(bills_clean_dictionary, handle)
```

### Cleaning the bills
Let's take a look at the text of one of the bills:


```python
print bills_dictionary['H.R.1000 - William Howard Taft National Historic Site Boundary Adjustment Act of 2001']
```

    
    [107th Congress Public Law 60]
    [From the U.S. Government Printing Office]
    
    
    &amp;lt;DOC&amp;gt;
    [DOCID: f:publ060.107]
    
    
    [[Page 115 STAT. 408]]
    
    Public Law 107-60
    107th Congress
    
                                     An Act
    
    
     
    To adjust the boundary of the William Howard Taft National Historic Site 
      in the State of Ohio, to authorize an exchange of land in connection 
        with the historic site, and for other purposes. &amp;lt;&amp;lt;NOTE: Nov. 5, 
                             2001 -  [H.R. 1000]&amp;gt;&amp;gt; 
    
        Be it enacted by the Senate and House of Representatives of the 
    United States of America in Congress &amp;lt;&amp;lt;NOTE: William Howard Taft 
    National Historic Site Boundary Adjustment Act of 2001. 16 USC 461 
    note.&amp;gt;&amp;gt; assembled,
    
    SECTION 1. SHORT TITLE.
    
        This Act may be cited as the ``William Howard Taft National Historic 
    Site Boundary Adjustment Act of 2001''.
    
    SEC. 2. EXCHANGE OF LANDS AND BOUNDARY ADJUSTMENT, WILLIAM HOWARD TAFT 
                NATIONAL HISTORIC SITE, OHIO.
    
        (a) Definitions.--In this section:
                (1) Historic site.--The term ``historic site'' means the 
            William Howard Taft National Historic Site in Cincinnati, Ohio, 
            established pursuant to Public Law 91-132 (83 Stat. 273; 16 
            U.S.C. 461 note).
                (2) Map.--The term ``map'' means the map entitled ``Proposed 
            Boundary Map, William Howard Taft National Historic Site, 
            Hamilton County, Cincinnati, Ohio,'' numbered 448/80,025, and 
            dated November 2000.
                (3) Secretary.--The term ``Secretary'' means the Secretary 
            of the Interior, acting through the Director of the National 
            Park Service.
    
        (b) Authorization of Land Exchange.--
                (1) Exchange.--The Secretary may acquire a parcel of real 
            property consisting of less than one acre, which is depicted on 
            the map as the ``Proposed Exchange Parcel (Outside Boundary)'', 
            in exchange for a parcel of real property, also consisting of 
            less than one acre, which is depicted on the map as the 
            ``Current USA Ownership (Inside Boundary)''.
                (2) Equalization of values.--If the values of the parcels to 
            be exchanged under paragraph (1) are not equal, the difference 
            may be equalized by donation, payment using donated or 
            appropriated funds, or the conveyance of additional land.
                (3) Adjustment of boundary.--The Secretary shall revise the 
            boundary of the historic site to reflect the exchange upon its 
            completion.
    
        (c) Additional Boundary Revision and Acquisition Authority.--
                (1) &amp;lt;&amp;lt;NOTE: Effective date.&amp;gt;&amp;gt; Inclusion of parcel in 
            boundary.--Effective on the date of the enactment of this Act, 
            the boundary of the historic site is revised to include an 
            additional parcel of real property, which is depicted on the map 
            as the ``Proposed Acquisition''.
    
    [[Page 115 STAT. 409]]
    
                (2) Acquisition authority.--The Secretary may acquire the 
            parcel referred to in paragraph (1) by donation, purchase from 
            willing sellers with donated or appropriated funds, or exchange.
    
        (d) Availability of Map.--The map shall be on file and available for 
    public inspection in the appropriate offices of the National Park 
    Service.
        (e) Administration of Acquired Lands.--Any lands acquired under this 
    section shall be administered by the Secretary as part of the historic 
    site in accordance with applicable laws and regulations.
    
        Approved November 5, 2001.
    
    LEGISLATIVE HISTORY--H.R. 1000:
    ---------------------------------------------------------------------------
    
    HOUSE REPORTS: No. 107-88 (Comm. on Resources).
    SENATE REPORTS: No. 107-76 (Comm. on Energy and Natural Resources).
    CONGRESSIONAL RECORD, Vol. 147 (2001):
                June 6, considered and passed House.
                Oct. 17, considered and passed Senate.
    
                                      &amp;lt;all&amp;gt;
    
    


Nice! This looks pretty good for a raw file. Since we're going to use tf-idf, we need to clean the text to keep only letters. We'll also remove stopwords (using the standard list included in NLTK) to remove words that provide very little information.

After cleaning, I zip the titles and clean text into a new dictionary ```bills_clean_dictionary``` and save it for quick recall.


```python
def clean_bill(raw_bill):
    """
    Function to clean bill text to keep only letters and remove stopwords
    Returns a string of the cleaned bill text
    """
    letters_only = re.sub('[^a-zA-Z]', ' ', raw_bill)
    words = letters_only.lower().split()
    stopwords_eng = set(stopwords.words("english"))
    useful_words = [x for x in words if not x in stopwords_eng]
    
    # Combine words into a paragraph again
    useful_words_string = ' '.join(useful_words)
    return(useful_words_string)

bills_text_list_clean = map(clean_bill, bills_text_list)
bills_clean_dictionary = dict(zip(titles_list, bills_text_list_clean))

with open(data_path + 'bills_dictionary_clean.pickle', 'wb') as handle:
    pickle.dump(bills_clean_dictionary, handle)
```

Let's see how the cleaned law looks:


```python
print clean_bills_dictionary['H.R.1000 - William Howard Taft National Historic Site Boundary Adjustment Act of 2001']
```

    th congress public law u government printing office amp lt doc amp gt docid f publ page stat public law th congress act adjust boundary william howard taft national historic site state ohio authorize exchange land connection historic site purposes amp lt amp lt note nov h r amp gt amp gt enacted senate house representatives united states america congress amp lt amp lt note william howard taft national historic site boundary adjustment act usc note amp gt amp gt assembled section short title act may cited william howard taft national historic site boundary adjustment act sec exchange lands boundary adjustment william howard taft national historic site ohio definitions section historic site term historic site means william howard taft national historic site cincinnati ohio established pursuant public law stat u c note map term map means map entitled proposed boundary map william howard taft national historic site hamilton county cincinnati ohio numbered dated november secretary term secretary means secretary interior acting director national park service b authorization land exchange exchange secretary may acquire parcel real property consisting less one acre depicted map proposed exchange parcel outside boundary exchange parcel real property also consisting less one acre depicted map current usa ownership inside boundary equalization values values parcels exchanged paragraph equal difference may equalized donation payment using donated appropriated funds conveyance additional land adjustment boundary secretary shall revise boundary historic site reflect exchange upon completion c additional boundary revision acquisition authority amp lt amp lt note effective date amp gt amp gt inclusion parcel boundary effective date enactment act boundary historic site revised include additional parcel real property depicted map proposed acquisition page stat acquisition authority secretary may acquire parcel referred paragraph donation purchase willing sellers donated appropriated funds exchange d availability map map shall file available public inspection appropriate offices national park service e administration acquired lands lands acquired section shall administered secretary part historic site accordance applicable laws regulations approved november legislative history h r house reports comm resources senate reports comm energy natural resources congressional record vol june considered passed house oct considered passed senate amp lt amp gt


Perfect! Way harder to read, but way more useful for finding textual similarities

### Calculating TF-IDF Vectors

So we've got a dictionary of laws and their text. Now it's time to calculate the tf-idf vectors. We'll initialize a stemmer from NLTK to treat words like ```incredible``` and ```incredibly``` as the same token. Then we'll initialize a TfidfVectorizer from ```sklearn``` and fit our corpus to the vectorizer. Since our corpus is all the values of the dictionary ```clean_bills_dictionary```, we'll pass ```clean_bills_dictionary.values()``` to the vectorizer.


```python
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.porter import PorterStemmer

with open('/users/nickbecker/Python_Projects/bills_dictionary_clean.pickle', 'r') as handle:
    clean_bills_dictionary = pickle.load(handle)

stemmer = PorterStemmer()

def stem_words(words_list, stemmer):
    return [stemmer.stem(word) for word in words_list]

def tokenize(text):
    tokens = nltk.word_tokenize(text)
    stems = stem_words(tokens, stemmer)
    return stems

tfidf = TfidfVectorizer(tokenizer=tokenize, stop_words='english')
tfs = tfidf.fit_transform(clean_bills_dictionary.values())
```

So what do we actually have now? ```tfs``` should be a matrix, where each row represents a law and each column represents a token (word) in the corpus. Let's see if we're right.


```python
tfs
```




    <3725x30894 sparse matrix of type '<type 'numpy.float64'>'
        with 1125934 stored elements in Compressed Sparse Row format>



Perfect. It's a sparse numpy array because it's essentially a matrix of zeros, with a handful of nonzero elements per row. The sparse matrix format is more efficient storage wise. Let's save the sparse matrix so we won't have to recalculate it. These functions (from the Scipy user group) let us easily save and load the array.


```python
def save_sparse_csr(filename,array):
    np.savez(filename,data = array.data ,indices=array.indices,
             indptr =array.indptr, shape=array.shape )

def load_sparse_csr(filename):
    loader = np.load(filename)
    return csr_matrix((loader['data'], loader['indices'], loader['indptr']),
                         shape = loader['shape'])

save_sparse_csr(data_path + 'laws_tf_idf.npz', tfs)
```

### Finding a Law's Nearest Neighbors with Cosine Distance

Finally, we can find similar laws! We'll initialize a NearestNeighbors class and fit our tf-idf matrix to it. Since we want to use cosine distance as our distance metric, I'll initialize it with ```metric='cosine'```.


```python
from sklearn.neighbors import NearestNeighbors

model_tf_idf = NearestNeighbors(metric='cosine', algorithm='brute')
model_tf_idf.fit(tfs)
```




    NearestNeighbors(algorithm='brute', leaf_size=30, metric='cosine',
             metric_params=None, n_jobs=1, n_neighbors=5, p=2, radius=1.0)



Let's define a function to print the k nearest neighbors for any query law. Since we have our corpus stored as a dictionary, we'll define the function to take the dictionary as an input.


```python
def print_nearest_neighbors(query_tf_idf, full_bill_dictionary, knn_model, k):
    """
    Inputs: a query tf_idf vector, the dictionary of bills, the knn model, and the number of neighbors
    Prints the k nearest neighbors
    """
    distances, indices = knn_model.kneighbors(query_tf_idf, n_neighbors = k+1)
    nearest_neighbors = [full_bill_dictionary.keys()[x] for x in indices.flatten()]
    
    for bill in xrange(len(nearest_neighbors)):
        if bill == 0:
            print 'Query Law: {0}\n'.format(nearest_neighbors[bill])
        else:
            print '{0}: {1}\n'.format(bill, nearest_neighbors[bill])
```

Time to test it! I'll pick a random law and find it's nearest neighbors.


```python
bill_id = np.random.choice(tfs.shape[0])
print_nearest_neighbors(tfs[bill_id], clean_bills_dictionary, model_tf_idf, k=5)
```

    Query Law: H.R.2517 - Protecting Our Children Comes First Act of 2007
    
    1: H.R.3209 - Recovering Missing Children Act
    
    2: S.1738 - PROTECT Our Children Act of 2008
    
    3: S.249 - Missing, Exploited, and Runaway Children Protection Act
    
    4: H.R.3092 - E. Clay Shaw, Jr. Missing Children's Assistance Reauthorization Act of 2013
    
    5: S.151 - PROTECT Act
    



```python
bill_id = np.random.choice(tfs.shape[0])
print_nearest_neighbors(tfs[bill_id], clean_bills_dictionary, model_tf_idf, k=5)
```

    Query Law: H.R.539 - Caribbean National Forest Act of 2005
    
    1: S.4001 - New England Wilderness Act of 2006
    
    2: H.R.4750 - Big Sur Wilderness and Conservation Act of 2002 
    
    3: H.R.233 - Northern California Coastal Wild Heritage Wilderness Act
    
    4: S.503 - Spanish Peaks Wilderness Act of 2000
    
    5: H.R.15 - Otay Mountain Wilderness Act of 1999
    


Pretty good! Remember, we didn't use the title of the laws in the corpus. So these rankings are just based on the textual similarity.

## What's Next?
