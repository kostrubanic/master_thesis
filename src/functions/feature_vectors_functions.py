# In this file, function for adding the feature vectors to the features is implemented

import pandas as pd
import numpy as np

# Adds feature vectors from the given matrices to the given dataset
#
# dataset- dataset with features to add the feature vectors to
# A - dataset with away feature vectors for each team
# H - dataset with home feature vectors for each team
# include_all - whether also add home feature vector to the away team
# and away feature vector to the home team
#
# Returns dataset with the feature vectors added withou the names of the teams
def add_feature_vector(dataset, A, H, include_all=False):
  dataset = dataset.reset_index(drop=True)
  away_vectors = []
  home_vectors = []
  away_home_vectors = []
  home_away_vectors = []
  for i in range(dataset.shape[0]):
    home_team = dataset['HomeTeam'][i]
    away_team = dataset['AwayTeam'][i]
    if away_team not in A:
      away_vector = np.zeros((A.shape[0],))
    else:
      away_vector = A[away_team]
    away_vectors.append(away_vector)
    if home_team not in H:
      home_vector = np.zeros((H.shape[0],))
    else:
      home_vector = H[home_team]
    home_vectors.append(home_vector)
    if include_all:
      if away_team not in H:
        away_home_vector = np.zeros((H.shape[0],))
      else:
        away_home_vector = H[away_team]
      away_home_vectors.append(away_home_vector)
      if home_team not in A:
        home_away_vector = np.zeros((A.shape[0],))
      else:
        home_away_vector = A[home_team]
      home_away_vectors.append(home_away_vector)
  away = pd.DataFrame(np.array(away_vectors))
  home = pd.DataFrame(np.array(home_vectors))
  dataset = pd.concat([dataset, away, home], axis=1)
  # Team names were only useful for getting the vectors
  dataset.drop(['HomeTeam', 'AwayTeam'], axis=1, inplace=True)
  dataset.columns = dataset.columns.astype(str)
  return dataset