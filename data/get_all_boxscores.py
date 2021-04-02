import numpy as np
import pandas as pd
from datetime import datetime
import requests
from dateutil.relativedelta import relativedelta
import sys

from sportsreference.nba.boxscore import Boxscore
from sportsreference.nba.roster import Roster, Player
from sportsreference.nba.schedule import Schedule
from sportsreference.nba.teams import Teams

df_boxscores = pd.read_csv('boxscore_twenty_years.csv')
box_list = df_boxscores['boxscore'].values

for idx, game in enumerate(box_list):
    print(game)
    if idx == 0:
        df = Boxscore(game).dataframe
        continue
    df = df.append(Boxscore(game).dataframe, ignore_index=True)

df.to_csv('all_boxscores.csv')