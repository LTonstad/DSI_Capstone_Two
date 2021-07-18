import numpy as np
import pandas as pd
pd.set_option('display.max_columns', None)

from datetime import datetime, timedelta
from collections import defaultdict
import itertools
import scipy.stats as stats
from scipy.spatial.distance import pdist, squareform, cdist
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.manifold import TSNE
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold, train_test_split, cross_validate, cross_val_score, permutation_test_score
from sklearn.decomposition import PCA
import matplotlib.cm as cm
import seaborn as sn

from sportsreference.nba.boxscore import Boxscore, Boxscores
from sportsreference.nba.roster import Roster, Player
from sportsreference.nba.schedule import Schedule
from sportsreference.nba.teams import Teams

import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')

# -------------------------
# Getting Data

# Get Season Games until now, should look into getting last games date from data for the strptime function

def get_current_season():
    current_season = datetime.strptime("12-22-2020", '%m-%d-%Y')
    yesterday = datetime.now() - timedelta(1)
    datetime.strftime(yesterday, '%m-%d-%Y')
    
    boxscore = Boxscores(current_season, yesterday)

    d = boxscore.__dict__
    
    d_cleaned = d['_boxscores']
    
    d_cleaned = {k: v for k, v in d_cleaned.items() if len(v) != 0}
    d_vals = list(d_cleaned.values())

    d1 = defaultdict(list)

    for outer_lst in d_vals:
        for i in outer_lst:
            for k, v in i.items():
                d1[k].append(v)
    
    df_boxscore = pd.DataFrame.from_dict(d1)

    df_boxscore['total_score'] = df_boxscore['away_score'] + df_boxscore['home_score']
    df_boxscore.set_index('boxscore', inplace=True)
    return df_boxscore

# Function to get all player data for particular season
def update_players_dataframes(df):
    


# -------------------------
# Cleaning Data

# Changes number to show in Millions
def mil_format(x):
    return "${:.1f}M".format(x/1000000)

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

# Function that finds columns in which values have less than 20 unique values are taken in and removed
def remove_career_cols(df):
    drop_lst = []
    for col in df.columns:
        unique = df[col].unique()
        val = len(unique)
        if val < 20:
            drop_lst.append(str(col))
            print(f'{col}: has {val} values')
            print(f'Values include:')
            print(f'     {unique}')

    print(f'Drop List: {drop_lst}')
    return df.drop(drop_lst, axis=1)

# Removing non-Salaried Players
def clean_salaries(df):
    # Dropping players that don't have any recorded salary
    df = df.dropna(subset=['salary'])
    # And players whos salary appeared as $0.0 for some reason
    df = df[df['salary'] >= 0]
    return df


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

# Creating current/past players DF's & then dropping the columns that only pertain to current players from career_df
def current_and_past(df):
    df_past_players = df[df['current_player'] == False]
    df_current_players = df[df['current_player'] == True]
    current_player_features = ['contract_total', 'contract_length', 'current_salary', 'current_avg_salary', 'current_team']
    df_past_players = df_past_players.drop(current_player_features, axis=1)
    return df_past_players, df_current_players

# Creating DF that only contains stats that are already averaged
def just_avgs(df):
    df_avgs = df[['assist_percentage', 'avg_salary', 'block_percentage', 'box_plus_minus',
    'defensive_box_plus_minus', 'defensive_rebound_percentage', 'effective_field_goal_percentage',
    'field_goal_percentage', 'field_goal_perc_sixteen_foot_plus_two_pointers',
    'field_goal_perc_ten_to_sixteen_feet', 'field_goal_perc_three_to_ten_feet',
    'field_goal_perc_zero_to_three_feet', 'field_goal_percentage',
    'free_throw_attempt_rate', 'free_throw_percentage', 'offensive_box_plus_minus',
    'offensive_rebound_percentage', 'player_efficiency_rating',
    'three_point_percentage', 'total_rebound_percentage', 'true_shooting_percentage',
    'turnover_percentage', 'two_point_percentage', 'usage_percentage',
    'win_shares', 'win_shares_per_48_minutes', 'years_played']]
    return df_avgs

#  -------------------------
# Plotting Data:

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

#  -------------------------
# Clustering Data:

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







#  -------------------------

# Pulled from TF Example

def plot_loss(history):
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.ylim([0, 10])
    plt.xlabel('Epoch')
    plt.ylabel('Error [avg_salary]')
    plt.legend()
    plt.grid(True)