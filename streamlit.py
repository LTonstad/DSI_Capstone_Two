import streamlit as st
import pandas as pd

'''

# NBA Salary Predictions

* **Goal**: Use different modeling techniques to try and accurately predict what the yearly salary of a player would be based on as seasonal player stats from the last 20 years
  * **Hidden Goal**  - To be *that guy*  that uses advanced Machine Learning models to predict salary worth when a friend complains that the Bucks overpaid for a player

-------------
'''

df_season = pd.read_csv('data/2nba_player_stats_by_season.csv')

st.write('Below is the season based DataFrame')
st.dataframe(df_season)

st.line_chart(df_season)