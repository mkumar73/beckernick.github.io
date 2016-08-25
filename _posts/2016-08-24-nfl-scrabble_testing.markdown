---
title:  "Which NFL Team Would Win in Scrabble? testing"
date:   2016-08-25
tags: [data science]

header:
  image: "fox_sports_images/scrabble_tile_holder.jpg"
  caption: "Photo credit: [**Etsy**](https://img0.etsystatic.com/007/0/7176622/il_570xN.376562392_f99r.jpg)"

excerpt: "pandas, web scraping, scrabble, and the nfl"
---

Has anyone ever wondered which NFL team would win in Scrabble? Well, my roommate [Dan Nolan](https://www.facebook.com/thedanpnolan/) did, and we decided to find out as best we could. I calculated the Scrabble point value for every player's name in the NFL to find out which teams have the highest combined scores. 

***

First, let's look at the top 25 NFL players by Scrabble point value of their names.

![](/images/fox_sports_images/top_player_scores.png?raw=true)


These are some high scores. _**Strong**_ performances by Jacquizz Rodgers, Kyle Juszcyzk, and Al-Hajj Shabazz due to the `j` and two `z`'s in their names. Now for the NFL team rankings.



![](/images/fox_sports_images/team_scores.png?raw=true)

Not a bad showing by the Washington Redskins (8th), except for the fact that the Dallas Cowboys and New York Giants came in 3rd and 4th, respectively. Still, I think most Redskins fans would be happy if they made the Divisional Round of the NFL Playoffs (quarterfinals).


***

For those interested, the code to scrape the data from Fox Sports, score the names, and generate the graphs can be found in the [Github repository](https://github.com/beckernick/scrabble_score_nfl) for this post. 




