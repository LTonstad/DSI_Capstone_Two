# Proposals:

## 1) Pro Sports Predictor based on distribution of funds within the team
  * Can get game/season stats using [Sportsipy](https://sportsipy.readthedocs.io/en/latest/)
  * Could get salary info here as [NBA Sportstrac](https://www.spotrac.com/nba/positional/breakdown/)

* **Goal(s)**: Cluster NBA players into different positional categories based off season stats, then train Neural Network to predict something like win percentage based off salary distribution within a team (or perhaps have it categorize into something like [Great Team, Good Team, Okay Team, Bad Team, Awful Team] if that is more feasible)
 * *Bonus*: Compare results to the current ongoing season

## 2) Music & Playlist categorizing using Librosa
  * Using my own Amazon Music data that includes these csv's I was able to download: Library (11767 songs), Playlists (22 different playlists made by myself and/or others on our family plan), Listening data (Has 155,237 rows of data)

*Side note: This one may be ambitious & the goals may need some tuning as I'm not sure the best way to go about this one, but I would just really enjoy first of all using my own dataset & exploring audio file processing with librosa* 

* **Goal(s)**: Categorize music using using data gathered by librosa, then using the information gathered to see if there is songs in a current playlist that do not belong in comparison to the rest of the playlist
 * *Bonus*: Not sure how possible it is but to have a generated playlist from categorizations and popularity amongst play counts in the family account

## 3) Analyzing IMDB movie data
 * [Data from Kaggle here](https://www.kaggle.com/sankha1998/tmdb-top-10000-popular-movies-dataset)

* **Goal**: Use NLP to identify what words or phrases in a movies description correlate with a higher rating on IMDB

------------------------

* Originally wanted to use this [PCPartPicker API](https://pypi.org/project/pcpartpicker/) to gather pc component data and categorize according to budget for a PC build but in practice I found it very hard to gather the data and move it into pandas properly

*Other datasets I had considered*
[Food Images](https://www.kaggle.com/kmader/food41)
[Brewery Dataset](https://www.kaggle.com/ankurnapa/brewery-dataset?select=beers.csv)
[Reddit Recipes](https://www.kaggle.com/michau96/recipes-from-reddit?select=Recipes_2.csv)
