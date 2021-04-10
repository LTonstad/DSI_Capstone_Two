import numpy as np
import pandas as pd
from tensorflow import keras

from sportsreference.nba.boxscore import Boxscore
from sportsreference.nba.roster import Roster, Player
from sportsreference.nba.schedule import Schedule
from sportsreference.nba.teams import Teams

woj_net = keras.models.load_model('models/logcosh.h5')

def ask_woj(player_name):
    current_player_features = ['contract_total', 'contract_length', 'current_salary', 'current_avg_salary', 'current_team']
    non_stats = ['salary', 'year','nationality', 'position', 'birth_date', 'year_list', 'team_abbreviation', 'player_id', 'name', 'salary']

    df_current_players = pd.read_csv('data/current_players.csv')

    df_player = df_current_players[df_current_players['name'] == str(player_name).title()]
    player_id = df_player['player_id'].iloc[0]
    year_info = df_player[df_player['age'] == df_player['age'].values[-2]]
    df_player = df_player[df_player['age'] == df_player['age'].values[-2]]
    df_player = df_player.drop(non_stats, axis=1)
    df_player = df_player.drop(current_player_features, axis=1)
    df_player = df_player.drop('Unnamed: 0', axis=1)
    df_player = df_player.astype('float64')

    sal = df_player['avg_salary']
    stats = df_player.drop('avg_salary', axis=1)

    supamax = ((95671+168750607)/2)*.35
    
    player = Player(player_id)
    contract = player.contract
    current_salary = contract['2020-21']
    
    sal_num = str(current_salary).replace(',','').replace('$','')
    sal_num = float(sal_num)
    
    if sal_num > supamax: # Roughly what a supermax salary would be, though is dependent on teams salary cap
        multiplier = 18
    else:
        multiplier = 7
    print(f'multiplier: {multiplier}', '\n')
    
    prediction = woj_net.predict(stats)
    mod_prediction = prediction[0][0]*multiplier
    mod_min = mod_prediction - (201332 * multiplier) # mae from Woj_net
    mod_max = mod_prediction + (201332 * multiplier)
    modified_prediction = "${:,.2f}".format(mod_prediction)
    print(f'Precise prediction from Woj: {modified_prediction}', '\n')
    
    modified_min = "${:,.2f}".format(mod_min)
    modified_max = "${:,.2f}".format(mod_max)
    print(f'Woj_net says this year that {player.name} should be paid between: {modified_min} - {modified_max}', '\n')
    print(f'{player.name} actually paid {current_salary}', '\n')
    
    if sal_num > mod_max:
        mod_net = "${:,.2f}".format(sal_num - mod_prediction)
        print(f'Woj_net believes {player.name} is getting overpaid this year by about {mod_net}')
        
    elif sal_num < mod_min:
        mod_net = "${:,.2f}".format(mod_prediction - sal_num)
        print(f'Woj_net believes {player.name} is getting underpaid this year by about {mod_net}')
        
    else:
        print(f'Woj_net believes {player.name} is getting paid properly this year :)')
    
    return year_info