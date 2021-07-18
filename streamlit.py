import streamlit as st
import pandas as pd
from src.woj_functions import *

'''

# Woj_Net will tell you whether your player is getting paid accurately or not

-------------
'''

df_season = pd.read_csv('data/nba_player_stats_by_season.csv')

