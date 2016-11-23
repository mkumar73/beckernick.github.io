---
title:  "The Election's Effect on Candidates's Wikipedia Page Views"
date:   2016-11-23
tags: [Politics, Consumer Behavior]

header:
  image: "/wikipedia_views/manhattan_skyline.jpg"

excerpt: "Presidential Candidates, Wikipedia, Visualization"
---

When I want to find out the background and policy positions of politicians, I usually start by checking their wikipedia pages. I'd bet that most people do, too.

I recently learned that the Wikimedia Foundation provides a ton of [data](https://meta.wikimedia.org/wiki/Research:Data) for research purposes, and it's totally free! One of the first things I stumbled upon was hourly page view data for every article on Wikipedia (over 800 million as of September 2013). The datasets are large, though, so using them was a bit cumbersome on my Macbook Air (mostly since I had to download them).

Fortunately, there's an [API](https://www.mediawiki.org/wiki/API:Main_page) and a [Python package](https://github.com/mediawiki-utilities/python-mwviews), eliminating the need to download and comb through the raw hourly datasets for simple queries. The API only provides access to data going back through mid-2015, but it's still awesome for a free service.

Since, we just had a presidential election, I was curious how the election might have affected the candidates's popularity on Wikipedia. The results are pretty dramatic.


![](/images/wikipedia_views/presidential_wiki_views.png?raw=true)

All the candidates's pages were visited more on the day after the election, but Trump had a staggering 5.5 million more views the day after the election.


The raw hourly page views datasets are updated every hour, which is pretty cool. I bet there are signals in the Wikipedia usage data that would be pretty useful for short-term forecasting a ton of different phenomena.

***

For those interested, the Jupyter Notebook with code to get the data and the R code to make the visualization can be found in the [Github repository](https://github.com/beckernick/visualizations/tree/master/State_of_the_Union_Wordcloud) for this post. 

Additionally, all analyses and conclusions presented on this website reflect my views and do not indicate concurrence by the Board of Governors or the Federal Reserve System.



