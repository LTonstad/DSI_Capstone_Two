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

# Get Season Games until now

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

# Pulled from Hands on Machine Learning github

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def relu(z):
    return np.maximum(0, z)

def derivative(f, z, eps=0.000001):
    return (f(z + eps) - f(z - eps))/(2 * eps)


# Pulled from CrossVal

def my_rmse(y_true, y_pred):
    mse = ((y_true - y_pred)**2).mean()
    return np.sqrt(mse)

def my_cross_val_scores(X_data, y_data, num_folds=3):
    ''' Returns error for k-fold cross validation. '''
    kf = KFold(n_splits=num_folds)
    train_error = np.empty(num_folds)
    test_error = np.empty(num_folds)
    index = 0
    reg = KNeighborsRegressor()
    for train, test in kf.split(X_data):
        reg.fit(X_data[train], y_data[train])
        pred_train = reg.predict(X_data[train])
        pred_test = reg.predict(X_data[test])
        train_error[index] = my_rmse(pred_train, y_data[train])
        test_error[index] = my_rmse(pred_test, y_data[test])
        index += 1
    return np.mean(test_error), np.mean(train_error)

# Pulled below from Gradient Boosting Regression Solutions

def load_and_split_data():
    ''' Loads sklearn's boston dataset and splits it into train:test datasets
        in a ratio of 80:20. Also sets the random_state for reproducible 
        results each time model is run.
    
        Parameters: None

        Returns:  (X_train, X_test, y_train, y_test):  tuple of numpy arrays
                  column_names: numpy array containing the feature names
    '''
    boston = load_boston() #load sklearn's dataset 
    X, y = boston.data, boston.target
    column_names = boston.feature_names
    X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                       test_size = 0.2, 
                                       random_state = 42)
    return (X_train, X_test, y_train, y_test), column_names


def cross_val(estimator, X_train, y_train, nfolds):
    ''' Takes an instantiated model (estimator) and returns the average
        mean square error (mse) and coefficient of determination (r2) from
        kfold cross-validation.

        Parameters: estimator: model object
                    X_train: 2d numpy array
                    y_train: 1d numpy array
                    nfolds: the number of folds in the kfold cross-validation

        Returns:  mse: average mean_square_error of model over number of folds
                  r2: average coefficient of determination over number of folds
    
        There are many possible values for scoring parameter in cross_val_score.
        http://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter

        kfold is easily parallelizable, so set n_jobs = -1 in cross_val_score
    '''
    mse = cross_val_score(estimator, X_train, y_train, 
                          scoring='neg_mean_squared_error',
                          cv=nfolds, n_jobs=-1) * -1
    # mse multiplied by -1 to make positive
    r2 = cross_val_score(estimator, X_train, y_train, 
                         scoring='r2', cv=nfolds, n_jobs=-1)
    mean_mse = mse.mean()
    mean_r2 = r2.mean()
    name = estimator.__class__.__name__
    print("{0:<25s} Train CV | MSE: {1:0.3f} | R2: {2:0.3f}".format(name,
                                                        mean_mse, mean_r2))
    return mean_mse, mean_r2
    
def stage_score_plot(estimator, X_train, y_train, X_test, y_test):
    '''
        Parameters: estimator: GradientBoostingRegressor or AdaBoostRegressor
                    X_train: 2d numpy array
                    y_train: 1d numpy array
                    X_test: 2d numpy array
                    y_test: 1d numpy array

        Returns: A plot of the number of iterations vs the MSE for the model for
        both the training set and test set.
    '''
    estimator.fit(X_train, y_train)
    name = estimator.__class__.__name__.replace('Regressor', '')
    learn_rate = estimator.learning_rate
    # initialize 
    train_scores = np.zeros((estimator.n_estimators,), dtype=np.float64)
    test_scores = np.zeros((estimator.n_estimators,), dtype=np.float64)
    # Get train score from each boost
    for i, y_train_pred in enumerate(estimator.staged_predict(X_train)):
        train_scores[i] = mean_squared_error(y_train, y_train_pred)
    # Get test score from each boost
    for i, y_test_pred in enumerate(estimator.staged_predict(X_test)):
        test_scores[i] = mean_squared_error(y_test, y_test_pred)
    plt.plot(train_scores, alpha=.5, label="{0} Train - learning rate {1}".format(
                                                                name, learn_rate))
    plt.plot(test_scores, alpha=.5, label="{0} Test  - learning rate {1}".format(
                                                      name, learn_rate), ls='--')
    plt.title(name, fontsize=16, fontweight='bold')
    plt.ylabel('MSE', fontsize=14)
    plt.xlabel('Iterations', fontsize=14)

def rf_score_plot(randforest, X_train, y_train, X_test, y_test):
    '''
        Parameters: randforest: RandomForestRegressor
                    X_train: 2d numpy array
                    y_train: 1d numpy array
                    X_test: 2d numpy array
                    y_test: 1d numpy array

        Returns: The prediction of a random forest regressor on the test set
    '''
    randforest.fit(X_train, y_train)
    y_test_pred = randforest.predict(X_test)
    test_score = mean_squared_error(y_test, y_test_pred)
    plt.axhline(test_score, alpha = 0.7, c = 'y', lw=3, ls='-.', label = 
                                                        'Random Forest Test')

def gridsearch_with_output(estimator, parameter_grid, X_train, y_train):
    '''
        Parameters: estimator: the type of model (e.g. RandomForestRegressor())
                    paramter_grid: dictionary defining the gridsearch parameters
                    X_train: 2d numpy array
                    y_train: 1d numpy array

        Returns:  best parameters and model fit with those parameters
    '''
    model_gridsearch = GridSearchCV(estimator,
                                    parameter_grid,
                                    n_jobs=-1,
                                    verbose=True,
                                    scoring='neg_mean_squared_error')
    model_gridsearch.fit(X_train, y_train)
    best_params = model_gridsearch.best_params_ 
    model_best = model_gridsearch.best_estimator_
    print("\nResult of gridsearch:")
    print("{0:<20s} | {1:<8s} | {2}".format("Parameter", "Optimal", "Gridsearch values"))
    print("-" * 55)
    for param, vals in parameter_grid.items():
        print("{0:<20s} | {1:<8s} | {2}".format(str(param), 
                                                str(best_params[param]),
                                                str(vals)))
    return best_params, model_best



def display_default_and_gsearch_model_results(model_default, model_gridsearch, 
                                              X_test, y_test):
    '''
        Parameters: model_default: fit model using initial parameters
                    model_gridsearch: fit model using parameters from gridsearch
                    X_test: 2d numpy array
                    y_test: 1d numpy array
        Return: None, but prints out mse and r2 for the default and model with
                gridsearched parameters
    '''
    name = model_default.__class__.__name__.replace('Regressor', '') # for printing
    y_test_pred = model_gridsearch.predict(X_test)
    mse = mean_squared_error(y_test, y_test_pred)
    r2 = r2_score(y_test, y_test_pred)
    print("Results for {0}".format(name))
    print("Gridsearched model mse: {0:0.3f} | r2: {1:0.3f}".format(mse, r2))
    y_test_pred = model_default.predict(X_test)
    mse = mean_squared_error(y_test, y_test_pred)
    r2 = r2_score(y_test, y_test_pred)
    print("     Default model mse: {0:0.3f} | r2: {1:0.3f}".format(mse, r2))