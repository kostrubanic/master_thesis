# In this file functions for the baseline using Matrix Factorization are implemented.
# There is a function for creating a matrix from a file with results, class which implements
# Matrix Factorization (I used implementation from https://albertauyeung.github.io/2017/04/23/python-matrix-factorization.html/#source-code.)
# and function for predicting and testing a season.

import testing_functions
import pandas as pd
import numpy as np

# Creates a matrix from given results file. 1 means home team win
# 3 means away team win and 2 means draw. Number on position [i, j]
# represents result of match between teams i and j when team i was 
# home team.
#
# results_file - resuls file to create matrix from
# till_index - index of first match to not include into the matrix
#              if None, all matches will be included
#
# Returns created matrix.
def get_matrix(results_file, till_index=None):
  results = pd.read_csv(results_file)
  number_of_teams = len(results['HomeTeam'].unique())
  matrix = np.zeros((number_of_teams, number_of_teams))
  teams = list(results['HomeTeam'].unique())
  if till_index is not None:
    end_index = till_index
  else:
    end_index = results.shape[0] # Including all matches

  for index in range(int(end_index)):
    row = teams.index(results.loc[index, 'HomeTeam'])
    col = teams.index(results.loc[index, 'AwayTeam'])
    result = results.loc[index, 'FTR']
    if result == 'H':
      result_to_write = 1
    elif result == 'A':
      result_to_write = 3
    else:
      result_to_write = 2
    matrix[row][col] = result_to_write
  return matrix

class MF():
    # Perform matrix factorization to predict empty entries in a matrix. Empty
    # entries are represented as zeros. Creates matrices P and Q, their product
    # aproximates matrix R.
    #
    # R - matrix to predict
    # K - number of latent dimensions
    # alpha - learning rate
    # beta - regularization parameter
    def __init__(self, R, K, alpha, beta, iterations):
        self.R = R
        self.num_users, self.num_items = R.shape
        self.K = K
        self.alpha = alpha
        self.beta = beta
        self.iterations = iterations

    def train(self):
        # Initializing user and item latent feature matrice
        self.P = np.random.normal(scale=1./self.K, size=(self.num_users, self.K))
        self.Q = np.random.normal(scale=1./self.K, size=(self.num_items, self.K))

        # Initializing the biases
        self.b_u = np.zeros(self.num_users)
        self.b_i = np.zeros(self.num_items)
        self.b = np.mean(self.R[np.where(self.R != 0)])

        # Creating a list of training samples
        self.samples = [
            (i, j, self.R[i, j])
            for i in range(self.num_users)
            for j in range(self.num_items)
            if self.R[i, j] > 0
        ]

        # Performing stochastic gradient descent for number of iterations
        training_process = []
        for i in range(self.iterations):
            np.random.shuffle(self.samples)
            self.sgd()
            mse = self.mse()
            training_process.append((i, mse))

        return training_process

    # Computes the total mean square error
    def mse(self):
        xs, ys = self.R.nonzero()
        predicted = self.full_matrix()
        error = 0
        for x, y in zip(xs, ys):
            error += pow(self.R[x, y] - predicted[x, y], 2)
        return np.sqrt(error)

    # Performs stochastic graident descent
    def sgd(self):
        for i, j, r in self.samples:
            # Computing prediction and error
            prediction = self.get_rating(i, j)
            e = (r - prediction)

            # Updating biases
            self.b_u[i] += self.alpha * (e - self.beta * self.b_u[i])
            self.b_i[j] += self.alpha * (e - self.beta * self.b_i[j])

            # Update user and item latent feature matrices
            self.P[i, :] += self.alpha * (e * self.Q[j, :] - 
                                          self.beta * self.P[i,:])
            self.Q[j, :] += self.alpha * (e * self.P[i, :] - 
                                          self.beta * self.Q[j,:])

    # Gets the predicted rating of user i and item j
    def get_rating(self, i, j):
        prediction = self.b + self.b_u[i] + self.b_i[j] + \
                     self.P[i, :].dot(self.Q[j, :].T)
        return prediction

    # Computes the full matrix using the resultant biases, P and Q
    def full_matrix(self):
        return self.b + self.b_u[:,np.newaxis] + self.b_i[np.newaxis:,] + \
                        self.P.dot(self.Q.T)

# Predicts the second half of the season of given results file. Creates matrix
# for the whole season and for the first half of the season. Trains matrix
# factorization on the first half matrix and compares the prediction for the
# second half with full season matrix.
#
# results_file - file of season to create matrices from
# threshold - how much around 2 predict draw
# K - number of latent dimensions to use
# 
# Returns achieved accuracy on the second half of the season.
def predict_season(results_file, threshold, K):
  results = pd.read_csv(results_file)
  # whole season matrix
  ground_truth = get_matrix(results_file)
  # half season matrix
  to_complete = get_matrix(results_file, results.shape[0] / 2)
  mf = MF(to_complete, K=K, alpha=0.1, beta=0.01, iterations=100)
  mf.train()
  pred = mf.full_matrix()
  # Creating predictions from float numbers
  pred[pred < 2 - threshold] = 1
  pred[pred > 2 + threshold] = 3
  pred[(pred > 1) & (pred < 3)] = 2
  # On the diagonal are matches of the same teams, we dont want to predict that.
  np.fill_diagonal(to_complete, -1)
  correct = 0.
  all = 0.
  for i in range(to_complete.shape[0]):
    for j in range(to_complete.shape[1]):
      if to_complete[i][j] == 0: # Zeros mean values to predict.
        all += 1
        if pred[i][j] == ground_truth[i][j]:
          correct += 1
  return correct / all

# Evaluates model on given season as on testing dataset. The model is trained
# on the beginning of the training dataset, it is evaluated on one round
# of the testing dataset and this round is added to the training dataset.
#
# results_file - file of dataset to do evaluation on
# threshold - how much around 2 predict draw
# K - number of latent dimensions to use
#
# Returns list of creates slices.
def test_season(results_file, threshold, K):
  results = pd.read_csv(results_file)
  results = results[['HomeTeam', 'AwayTeam', 'FTR', 'Date']]
  matches_per_round = int(results['HomeTeam'].unique().size / 2)
  start_index = int(results.shape[0] / 2)
  # Rounds of the testing dataset
  slices = testing_functions.get_slices(results, matches_per_round, start_index)
  results = pd.read_csv(results_file)
  teams = list(results['HomeTeam'].unique())
  weighted_sum = 0
  sum = 0
  for slc in slices:
    slc = slc.reset_index()
    first_index = slc.loc[0, 'index']
    # Whole season matrix
    ground_truth = get_matrix(results_file)
    # Half season matrix
    to_complete = get_matrix(results_file, first_index)
    mf = MF(to_complete, K=K, alpha=0.1, beta=0.01, iterations=100)
    mf.train()
    pred = mf.full_matrix()
    # Creating predictions from float numbers
    pred[pred < 2 - threshold] = 1
    pred[pred > 2 + threshold] = 3
    pred[(pred > 1) & (pred < 3)] = 2
    # Assigning zeros only to matches of current slice
    to_complete[to_complete == 0] = -1
    for index in range(slc.shape[0]):
      row = teams.index(slc.loc[index, 'HomeTeam'])
      col = teams.index(slc.loc[index, 'AwayTeam'])
      to_complete[row][col] = 0
    correct = 0.
    all = 0.
    for i in range(to_complete.shape[0]):
      for j in range(to_complete.shape[1]):
        if to_complete[i][j] == 0:
          all += 1
          if pred[i][j] == ground_truth[i][j]: # Zeros mean values to predict.
            correct += 1
    weighted_sum += (correct / all * slc.shape[0])
    sum += slc.shape[0]
  return weighted_sum / sum