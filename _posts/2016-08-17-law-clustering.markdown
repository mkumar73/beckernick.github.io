---
title:  "Clustering US Laws using TF-IDF, and K Nearest Neighbors"
date:   2016-08-18
categories: [data science, law, clustering, natural language processing]
tags: [data science, natural langauge processing]
header:
  image: "congress_edge_detection.jpg"
---



# Download Enacted Laws from www.congress.gov

### Web Scraping with BeautifulSoup
First, I needed to scrape https://www.congress.gov/legislation to get the links for each bill. Each page has 25 bills on it, among other things. The URL ending of ```?q=%7B"bill-status"%3A"law"%7D``` filters the results to only be enacted bills (bills that became law). By looking at a few of the pages, I noticed that the hyperlinks I need are essentially in the same place on every page (inside 'h2' tags within an 'ol' tag of class 'results_list').

So I can scrape all the hyperlinks with a nested loop. The outer loop grabs the data in the table on each page, and the inner loop extracts the hyperlinks for bills. Out of respect for www.congress.gov's servers, I store the links in a list and write the list to a text file so I don't have to scrape them again.


```python
from __future__ import division
import re
from nltk.corpus import stopwords
import bs4
import urllib2
import time
import pickle


# Get the hyperlinks for all of the bills
base_law_url = 'https://www.congress.gov/legislation?q=%7B"bill-status"%3A"law"%7D&page='
relevant_links = []

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

#with open('/users/nickbecker/Python_Projects/law_links.txt', 'w') as handle:
#    for link in relevant_links:
#        handle.write("%s\n" % link)





```

    Page 1


    /Users/nickbecker/anaconda/lib/python2.7/site-packages/bs4/__init__.py:166: UserWarning: No parser was explicitly specified, so I'm using the best available HTML parser for this system ("lxml"). This usually isn't a problem, but if you run this code on another system, or in a different virtual environment, it may use a different parser and behave differently.
    
    To get rid of this warning, change this:
    
     BeautifulSoup([your markup])
    
    to this:
    
     BeautifulSoup([your markup], "lxml")
    
      markup_type=markup_type))


    Page 2
    Page 3
    Page 4
    Page 5
    Page 6
    Page 7
    Page 8
    Page 9
    Page 10
    Page 11
    Page 12
    Page 13
    Page 14
    Page 15
    Page 16
    Page 17
    Page 18
    Page 19
    Page 20
    Page 21
    Page 22
    Page 23
    Page 24
    Page 25
    Page 26
    Page 27
    Page 28
    Page 29
    Page 30
    Page 31
    Page 32
    Page 33
    Page 34
    Page 35
    Page 36
    Page 37
    Page 38
    Page 39
    Page 40
    Page 41
    Page 42
    Page 43
    Page 44
    Page 45
    Page 46
    Page 47
    Page 48
    Page 49
    Page 50
    Page 51
    Page 52
    Page 53
    Page 54
    Page 55
    Page 56
    Page 57
    Page 58
    Page 59
    Page 60
    Page 61
    Page 62
    Page 63
    Page 64
    Page 65
    Page 66
    Page 67
    Page 68
    Page 69
    Page 70
    Page 71
    Page 72
    Page 73
    Page 74
    Page 75
    Page 76
    Page 77
    Page 78
    Page 79
    Page 80
    Page 81
    Page 82
    Page 83
    Page 84
    Page 85
    Page 86
    Page 87
    Page 88
    Page 89
    Page 90
    Page 91
    Page 92
    Page 93
    Page 94
    Page 95
    Page 96
    Page 97
    Page 98
    Page 99
    Page 100
    Page 101
    Page 102
    Page 103
    Page 104
    Page 105
    Page 106
    Page 107
    Page 108
    Page 109
    Page 110
    Page 111
    Page 112
    Page 113
    Page 114
    Page 115
    Page 116
    Page 117
    Page 118
    Page 119
    Page 120
    Page 121
    Page 122
    Page 123
    Page 124
    Page 125
    Page 126
    Page 127
    Page 128
    Page 129
    Page 130
    Page 131
    Page 132



    ---------------------------------------------------------------------------

    KeyboardInterrupt                         Traceback (most recent call last)

    <ipython-input-1-749fa48a17c6> in <module>()
         16 
         17     response = urllib2.urlopen(base_law_url + str(i))
    ---> 18     html = response.read()
         19     soup = bs4.BeautifulSoup(html)
         20     tags = soup.find('ol', {'class' : 'results_list'})


    /Users/nickbecker/anaconda/lib/python2.7/socket.pyc in read(self, size)
        353             while True:
        354                 try:
    --> 355                     data = self._sock.recv(rbufsize)
        356                 except error, e:
        357                     if e.args[0] == EINTR:


    /Users/nickbecker/anaconda/lib/python2.7/httplib.pyc in read(self, amt)
        610         # connection, and the user is reading more bytes than will be provided
        611         # (for example, reading in 1k chunks)
    --> 612         s = self.fp.read(amt)
        613         if not s and amt:
        614             # Ideally, we would raise IncompleteRead if the content-length


    /Users/nickbecker/anaconda/lib/python2.7/socket.pyc in read(self, size)
        382                 # fragmentation issues on many platforms.
        383                 try:
    --> 384                     data = self._sock.recv(left)
        385                 except error, e:
        386                     if e.args[0] == EINTR:


    /Users/nickbecker/anaconda/lib/python2.7/ssl.pyc in recv(self, buflen, flags)
        754                     "non-zero flags not allowed in calls to recv() on %s" %
        755                     self.__class__)
    --> 756             return self.read(buflen)
        757         else:
        758             return self._sock.recv(buflen, flags)


    /Users/nickbecker/anaconda/lib/python2.7/ssl.pyc in read(self, len, buffer)
        641                 v = self._sslobj.read(len, buffer)
        642             else:
    --> 643                 v = self._sslobj.read(len)
        644             return v
        645         except SSLError as x:


    KeyboardInterrupt: 


With the hyperlinks to few thousands enacted bills in relevant_links, I next need to download the actual bills. By looking at a few of the pages, I noticed that access to the text version of the bills (instead of the PDF) is controlled by the URL ending ```/text?format=txt```. On each page, the text stored in a ```<pre>``` tag with id ```billTextContainer```.

By looping through ```relevant_links```, I grab the title and text of each bill and store them in ```titles_list``` and ```bills_text_list```.


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

```

### Cleaning the bills
Let's take a look at the text of one of the bills:


```python
with open('/users/nickbecker/Python_Projects/bills_dictionary.pickle', 'r') as handle:
    bills_dictionary = pickle.load(handle)
```


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

After cleaning, I zip the titles and clean text into a dictionary ```bills_clean_dictionary``` with the titles as keys and text as values to save them efficiently.


```python
def clean_bill(raw_bill):
    """
    Function to clean bill text to keep only letters and remove stopwords
    Returns a string of the cleaned bill text
    """
    letters_only = re.sub("[^a-zA-Z]",
                          " ",
                          raw_bill)
    words = letters_only.lower().split()
    stopwords_eng = set(stopwords.words("english"))
    useful_words = [x for x in words if not x in stopwords_eng]
    
    # Combine words into a paragraph again
    useful_words_string = " ".join(useful_words)
    return(useful_words_string)

bills_text_list_clean = map(clean_bill, bills_text_list)

bills_clean_dictionary = dict(zip(titles_list, bills_text_list_clean))

#with open('/users/nickbecker/Python_Projects/bills_dictionary_clean.pickle', 'wb') as handle:
#    pickle.dump(bills_clean_dictionary, handle)


```

Let's see how the cleaned law looks:


```python
#with open('/users/nickbecker/Python_Projects/bills_dictionary_clean.pickle', 'r') as handle:
#    clean_bills_dictionary = pickle.load(handle)
```


```python
print clean_bills_dictionary['H.R.1000 - William Howard Taft National Historic Site Boundary Adjustment Act of 2001']
```

    th congress public law u government printing office amp lt doc amp gt docid f publ page stat public law th congress act adjust boundary william howard taft national historic site state ohio authorize exchange land connection historic site purposes amp lt amp lt note nov h r amp gt amp gt enacted senate house representatives united states america congress amp lt amp lt note william howard taft national historic site boundary adjustment act usc note amp gt amp gt assembled section short title act may cited william howard taft national historic site boundary adjustment act sec exchange lands boundary adjustment william howard taft national historic site ohio definitions section historic site term historic site means william howard taft national historic site cincinnati ohio established pursuant public law stat u c note map term map means map entitled proposed boundary map william howard taft national historic site hamilton county cincinnati ohio numbered dated november secretary term secretary means secretary interior acting director national park service b authorization land exchange exchange secretary may acquire parcel real property consisting less one acre depicted map proposed exchange parcel outside boundary exchange parcel real property also consisting less one acre depicted map current usa ownership inside boundary equalization values values parcels exchanged paragraph equal difference may equalized donation payment using donated appropriated funds conveyance additional land adjustment boundary secretary shall revise boundary historic site reflect exchange upon completion c additional boundary revision acquisition authority amp lt amp lt note effective date amp gt amp gt inclusion parcel boundary effective date enactment act boundary historic site revised include additional parcel real property depicted map proposed acquisition page stat acquisition authority secretary may acquire parcel referred paragraph donation purchase willing sellers donated appropriated funds exchange d availability map map shall file available public inspection appropriate offices national park service e administration acquired lands lands acquired section shall administered secretary part historic site accordance applicable laws regulations approved november legislative history h r house reports comm resources senate reports comm energy natural resources congressional record vol june considered passed house oct considered passed senate amp lt amp gt


Perfect! Way harder to read, but way more useful for finding textual similarities


```python

```

### Clustering laws using TF-IDF and Cosine Distance

So we've got a dictionary of laws and their text. Now it's time to calculate the tf-idf vectors. We'll initialize a stemmer from NLTK to treat words like ```incredible``` and ```incredibly``` as the same token. Then we'll initialize a TfidfVectorizer from ```sklearn``` and fit our corpus to the vectorizer. Since our corpus is all the values of the dictionary ```clean_bills_dictionary```, we'll pass ```clean_bills_dictionary.values()``` to the vectorizer.


```python
from __future__ import division
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.stem.porter import PorterStemmer
import pickle


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



Perfect. It's a sparse numpy array because it's essentially a matrix of zeros, with a handful of nonzero elements per row. The sparse matrix format is more efficient storage wise.

### Finding Laws' Nearest Neighbors

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

Time to test! I'll pick a random law and find it's nearest neighbors.


```python
#np.random.seed(12)
bill_id = np.random.choice(tfs.shape[0])
print_nearest_neighbors(tfs[bill_id], clean_bills_dictionary, model_tf_idf, k=5)
```

    Query Law: H.R.3734 - To designate the Federal building located at Fifth and Richardson Avenues in Roswell, New Mexico, as the "Joe Skeen Federal Building".
    
    1: H.R.3147 - To designate the Federal building located at 324 Twenty-Fifth Street in Ogden, Utah, as the "James V. Hansen Federal Building".
    
    2: H.R.4957 - To designate the Federal building located at 99 New York Avenue, N.E., in the District of Columbia as the "Ariel Rios Federal Building".
    
    3: H.R.3639 - To designate the Federal building located at 2201 C Street, Northwest, in the District of Columbia, currently headquarters for the Department of State, as the "Harry S. Truman Federal Building".
    
    4: H.R.5773 - To designate the Federal building located at 6401 Security Boulevard in Baltimore, Maryland, commonly known as the Social Security Administration  Operations Building, as the "Robert M. Ball Federal Building".
    
    5: H.R.821 - To designate the facility of the United States Postal Service located at 1030 South Church Street in Asheboro, North Carolina, as the "W. Joe Trogdon Post Office Building".
    



```python
bill_id = np.random.choice(tfs.shape[0])
print_nearest_neighbors(tfs[bill_id], clean_bills_dictionary, model_tf_idf, k=5)
```

    Query Law: H.R.1953 - San Francisco Old Mint Commemorative Coin Act
    
    1: H.R.3373 - To require the Secretary of the Treasury to mint coins in conjunction with the minting of coins by the Republic of Iceland in commemoration of the millennium of the discovery of the New World by Lief Ericson.
    
    2: H.R.3229 - National Infantry Museum and Soldier Center Commemorative Coin Act
    
    3: H.R.2768 - John Marshall Commemorative Coin Act
    
    4: H.R.634 - American Veterans Disabled for Life Commemorative Coin Act
    
    5: H.R.5714 - United States Army Commemorative Coin Act of 2008
    



```python

```

## Final Thoughts
Pretty good! Remember, we didn't use the title of the laws in the corpus.


```python

```

