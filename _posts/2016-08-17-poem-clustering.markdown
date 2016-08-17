---
title:  "Clustering Poetry using TF-IDF"
date:   2016-08-17
categories: [data science, poetry, clustering, natural language processing]
tags: [data science, poetry, natural langauge processing]
header:
  image: "i_love_poetry.jpg"
---



# Download the Top 500 Poems from Poemhunter.com

#### Web Scraping with BeautifulSoup
First, I needed to scrape the 20 pages compring the top 500 poems. Each page has 25 poems on it, among other things. By looking at a few of the pages, I noticed that the hyperlinks I need are essentially in the same place on every page (inside a table whose class is 'poems-listing'). So I can scrape all 500 hyperlinks with a nested loop. The outer loop grabs the data in the table on each page, and the inner loop extracts the hyperlinks for poems. Out of respect for poemhunter.com's server, I pause for a few seconds between each HTTP request.


```python
import bs4
import urllib2
import re
import time
import pickle

poem_list_url = 'http://poemhunter.com/p/m/l.asp?a=0&l=top500&order=title&p='
relevant_links = []

number_pages = 20
poems_per_page = 25

for i in xrange(1, number_pages + 1):
    print 'Page {0}'.format(i)
    
    response = urllib2.urlopen(poem_list_url + str(i))
    html = response.read()
    soup = bs4.BeautifulSoup(html)
    tags = soup.find('table', {'class' : 'poems-listing'})
    tr_list = tags.find_all('tr')

    # 25 poems per page
    time.sleep(3)
    for j in xrange(1, poems_per_page + 1):
        temp = tr_list[j].find('td', {'class' : 'title'})        
        hrefs = temp.find_all('a')
        hrefs = [str(x) for x in hrefs]
        relevant_links.append(hrefs)
```

    Page 1
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



```python
If this works, relevant_links should contain hyperlinks for 500 poems.
```


```python
print len(relevant_links)
print relevant_links[0]
print relevant_links[1]
```

    500
    ['<a href="/poem/phenomenal-woman/" title="Phenomenal Woman poem">Phenomenal Woman</a>', '<a href="/maya-angelou/" title="Maya Angelou poet">Maya Angelou</a>']
    ['<a href="/poem/still-i-rise/" title="Still I Rise poem">Still I Rise</a>', '<a href="/maya-angelou/" title="Maya Angelou poet">Maya Angelou</a>']


500

['<a href="/poem/phenomenal-woman/" title="Phenomenal Woman poem">Phenomenal Woman</a>', '<a href="/maya-angelou/" title="Maya Angelou poet">Maya Angelou</a>']

['<a href="/poem/still-i-rise/" title="Still I Rise poem">Still I Rise</a>', '<a href="/maya-angelou/" title="Maya Angelou poet">Maya Angelou</a>']


```python

        



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



```


```python

```


```python

```
