import numpy as np
import pandas as pd
from sportsreference.nba.boxscore import Boxscore

df_boxscores = pd.read_csv('data/boxscore_twenty_years.csv')
box_list = df_boxscores['boxscore'].values

for game in box_list:
    print(game)
    if idx == 0:
        df = Boxscore(game).dataframe
        continue
    df = df.append(Boxscore(game).dataframe, ignore_index=True)

df.to_csv('all_boxscores.csv')