import numpy as np
import pandas as pd
from datetime import datetime
import itertools
import scipy.stats as stats
from scipy.spatial.distance import pdist, squareform, cdist
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.manifold import TSNE
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
import matplotlib.cm as cm
import seaborn as sn

from sportsreference.nba.boxscore import Boxscore, Boxscores
from sportsreference.nba.roster import Roster, Player
from sportsreference.nba.schedule import Schedule
from sportsreference.nba.teams import Teams

import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')

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

# season_df.to_csv('../data/nba_player_stats_by_season.csv')
# career_df.to_csv('../data/nba_player_stats_by_career.csv')

# Get Season Games until now
def get_current_season():
    current_season = datetime.strptime("12-22-2020", '%m-%d-%Y')
    today = datetime.date.today()
    
    boxscore = Boxscores(current_season, today)

    d = boxscore.__dict__

    d_cleaned = d['_boxscores']
    d_cleaned = {k: v for k, v in d.items() if len(v) is not 0}

    d_vals = list(d_cleaned.values())

    from collections import defaultdict

    d = defaultdict(list)

    for outer_lst in d_vals:
        for i in outer_lst:
            for k, v in i.items():
                d[k].append(v)
    
    df_boxscore = pd.DataFrame.from_dict(d)

    return df_boxscore



# Checks for columns that don't have unique values and returns list of columns
def check_columns(df):
    for col in df.columns:
        unique = df[col].unique()
        val = len(unique)
        remove_lst = []
        
        if val < 20:
            print(f'{col}: has {val} values')
            print(f'Values include:')
            print(f'     {unique}')
        elif val == 1:
            remove_lst.append(col)
    
    return remove_lst

# Modifies height to be shown in inches
def to_inches(height):
    feet, inches = str(height).split('-')
    return (int(feet)*12) + int(inches)

# Normalizing numeric data
def normalize(df):
    x = df.values
    min_max_scalar = MinMaxScaler()
    x_scaled = min_max_scalar.fit_transform(x)
    df_normalized = pd.DataFrame(x_scaled)

    return df_normalized

# Plot PCA graph (x = df_normalized from normalize function) & returns the amount of variance to be explained by this amount of components
def plot_pca(x):
    pca = PCA(x.shape[1])
    pca.fit(x)
    var = pca.explained_variance_ratio_
    var1 = np.cumsum(np.round(pca.explained_variance_ratio_, decimals=4)*100)
    var_to_beat = np.argwhere(var1 >= 98)
    var_explained = np.round(var1[var_to_beat[0]], decimals=2)[0]
    comps = var_to_beat.flatten()[1]
    
    fig, ax = plt.subplots(figsize=(16,10))

    ax.plot(var1)
    ax.set_title('Choosing PCA Components')
    ax.set_xlabel('Number of PCA Components')
    ax.set_ylabel('Cumulative Explained Variance (%)')
    ax.axvline(var_to_beat[0], c='r', linestyle='--')
    ax.annotate(f'{var_explained}% can be explained at {comps} components', xy=(comps, var_explained-20))

    return var_explained


# Return dictionary of amount of times a certain number of clusters had the highest silhouette score
def find_best_cluster_amount(x, loops):
    d = {}
    for i in range(loops):
        after_pca = PCA(n_components=27, whiten=True).fit_transform(x)
        k_range = range(2,16)
        k_means_var = [KMeans(n_clusters=k).fit(after_pca) for k in k_range]
        labels = [i.labels_ for i in k_means_var]
        sil_score = [silhouette_score(after_pca, i, metric='cosine') for i in labels]
        centroids = [i.cluster_centers_ for i in k_means_var]
        k_euclid = [cdist(after_pca,cent,'cosine') for cent in centroids]
        dist = [np.min(ke, axis=1) for ke in k_euclid]
        wcss = [sum(d**2) for d in dist]
        tss = sum(pdist(after_pca)**2/after_pca.shape[0])
        bss = tss - wcss
        
        sil_arr = np.transpose(sil_score)*100
        max_score = np.max(sil_arr)
        max_score_index = np.argmax(sil_arr == max_score)
        
        if max_score_index in d:
            d[max_score_index] += 1
        else:
            d[max_score_index] = 1
    
    return d

# Plot bar graph of results from find_best_cluster_amount()
def plot_d_clusters(d_clusters):
    d_clusters = dict(sorted(d_clusters.items(), key=lambda item: item[1], reverse=True))

    fig, ax = plt.subplots(figsize=(16,10))

    ax.barh(list(d_clusters.keys()), list(d_clusters.values()))
    ax.set_title('Choosing Amount of Clusters')
    ax.set_xlabel('Times Cluster had Max Silhouette Score')
    ax.set_ylabel('Number of Clusters')
    ax.set_yticks(np.arange(2,16))

    fig = ax.figure
    fig.set_size_inches(16,10)
    fig.tight_layout(pad=1)