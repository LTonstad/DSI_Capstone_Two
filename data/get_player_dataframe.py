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

# helper function to get player age during each season.
def get_age(year, bd):
    if year[0] == "Career":
        return None
    else:
        year_dt = datetime(int(year[0][0:4]) + 1, 1, 1)
        age_years = relativedelta(year_dt, bd).years + relativedelta(year_dt, bd).months/12
        return age_years
    
# helper function to get year for each row and denote
# rows that contain career totals.
def get_year(ix):
    if ix[0] == "Career":
        return "Career"
    elif ix[0] == "1999-00":
        return "2000"
    else:
        return ix[0][0:2] + ix[0][-2:]

# Function to get player info from Player class object.def get_player_df(player):

def get_player_df(player):
    # get player df and add some extra info
    player_df = player.dataframe
    player_df['birth_date'] = player.birth_date
    player_df['player_id'] = player.player_id
    player_df['name'] = player.name
    player_df['year'] = [get_year(ix) for ix in player_df.index]
    player_df['id'] = [player_id + ' ' + year for player_id, year in zip(player_df['player_id'], player_df['year'])]
    player_df['age'] = [get_age(year, bd) for year, bd in zip(player_df.index, player_df['birth_date'])]
    
    # Stuff I've added
    years_played = list(filter(lambda x: x != 'Career', player_df.salary.reset_index().iloc[:,0].to_list()))
    player_df['avg_salary'] = player_df.salary / len(years_played)
    player_df['years_played'] = len(years_played)
    player_df['year_list'] = str(years_played)
    player_df['current_player'] = player_df['year_list'].str.contains('2020-21')

    if (player_df['current_player']).any():
        salary = player.contract
        if salary != None:
            sal_nums = [str(x).replace(',','').replace('$','') for x in list(salary.values())]
            contract_total = np.sum([int(x) for x in sal_nums])
            player_df['contract_total'] = contract_total
            player_df['contract_length'] = len(sal_nums)
            player_df['current_salary'] = sal_nums[0]
            player_df['current_avg_salary'] = contract_total / len(sal_nums)
            player_df['current_team'] = player._team_abbreviation[-2]

    player_df.set_index('id', drop = True, inplace = True)
    
    return player_df


# initialize a list of players that we have pulled data for
players_collected = []
season_df_init = 0
career_df_init = 0
season_df = 0
career_df = 0# iterate through years.
for year in range(2020, 1999, -1):
    print('\n' + str(year))
        
    # iterate through all teams in that year.
    for team in Teams(year = str(year)).dataframes.index:
        print('\n' + team + '\n')
        
        # iterate through every player on a team roster.
        for player_id in Roster(team, year = year,
                         slim = True).players.keys():
            
            # only pull player info if that player hasn't
            # been pulled already.
            if player_id not in players_collected:
                
                player = Player(player_id)
                player_info = get_player_df(player)
                player_seasons = player_info[
                                 player_info['year'] != "Career"]
                player_career = player_info[
                                player_info['year'] == "Career"]
                
                # create season_df if not initialized
                if not season_df_init:
                    season_df = player_seasons
                    season_df_init = 1
                
                # else concatenate to season_df
                else:
                    season_df = pd.concat([season_df,
                                   player_seasons], axis = 0)
                    
                if not career_df_init:
                    career_df = player_career
                    career_df_init = 1
                
                # else concatenate to career_df
                else:
                    career_df = pd.concat([career_df,
                                   player_career], axis = 0)
                
                # add player to players_collected
                players_collected.append(player_id)
                print(player.name)

season_df.to_csv('2nba_player_stats_by_season.csv')
career_df.to_csv('2nba_player_stats_by_career.csv')