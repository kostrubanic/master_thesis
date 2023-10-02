# In this file functions for the tensor completion model are implemented.
# There are functions for creating matrix and tensor from files with resluts,
# for saving a tensor into a file, for predicting and testing a season, for
# getting scores for different number of maximum iterations, for plotting the
# confusion matrix, for getting SVD from a matrix and for saving feature
# matrices into files.

import testing_functions
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.utils.extmath import randomized_svd

# Creates a matrix from given results file. 1 means home team win
# -1 means away team win and 0 means draw. Number on position [i, j]
# represents result of match between teams i and j when team i was 
# home team.
#
# results_file - resuls file to create matrix from
# till_index - index of first match to not include into the matrix
#              if None, all matches will be included
#
# Returns created matrix.
def get_matrix(results_file, teams, till_index=None):
  results = pd.read_csv(results_file, encoding='windows-1252')
  number_of_teams = len(teams)
  matrix = np.empty((number_of_teams, number_of_teams))
  matrix[:] = np.NaN
  num_matches = results.shape[0]
  if isinstance(till_index, int):
    end_index = till_index
  elif isinstance(till_index, float):
    end_index = num_matches * till_index
  else:
    end_index = num_matches # Including all matches
  
  for index in range(int(end_index)):
    row = teams.index(results.loc[index, 'HomeTeam'])
    col = teams.index(results.loc[index, 'AwayTeam'])
    result = results.loc[index, 'FTR']
    if result == 'H':
      result_to_write = 1
    elif result == 'A':
      result_to_write = -1
    else:
      result_to_write = 0
    matrix[row][col] = result_to_write
  return matrix

# Creates a tensor from given results files. 1 means home team win
# -1 means away team win and 0 means draw. Number on position [i, j]
# of the k-th slice represents result of match between teams i and j
# in k-th season when team i was home team.
#
# results_files - resuls file to create tensor from
# last_till_index - index of first match to not include into the last matrix
#              if None, all matches will be included
#
# Returns created tensor and a list of all teams.
def get_tensor(results_files, last_till_index=None):
  datasets = []
  for results_file in results_files:
    dataset = pd.read_csv(results_file, encoding='windows-1252')
    datasets.append(dataset)
  all_matches = pd.concat(datasets)
  teams = list(all_matches['HomeTeam'].unique())

  matrices = []
  i = 1
  for results_file in results_files:
    # Only part of the last matrix could be wanted
    if i == len(results_files):
      matrix = get_matrix(results_file, teams, till_index=last_till_index)
    else:
      matrix = get_matrix(results_file, teams)
    matrices.append(matrix)
    i += 1
  tensor = np.array(matrices)
  return tensor, teams

# Saves given tensor into a file in a format, where each row represents one
# element with it's coordinates and value. Pyten package takes tensor in this
# format.
#
# tensor - tensor to be saved
# file_name - name of o file to save the tensor into
#
def tensor_to_file(tensor, file_name):
  f = open(file_name, "a")
  f.write("x1;x2;x3;r\n")
  for i in range(tensor.shape[0]):
    for j in range(tensor.shape[1]):
      for k in range(tensor.shape[2]):
	# Writes coordinates
        f.write(';'.join(str(x) for x in (i + 1, j + 1, k + 1)))
        if ~np.isnan(tensor[i][j][k]):
	  # Writes value
          f.write(';' + str(tensor[i][j][k]) + '\n')
        else:
          f.write(';\n')
  f.close()

# Predicts the wanted part of the last season of given results files. Creates
# a full tensor from given results files and a tensor with some missing data
# in the last season according to last_till_index. Fills the tensor with given
# function and compares it to the full tensor.
#
# results_files - files of seasons to train on, the last season will be predicted
# last_till_index - index of first match to not include into the last matrix, defines,
#		    which matches will be predicted, if None, all matches will be
# 		    included in the last matrix
# function - tensor completion algorithm to be used
# tol - if the prediction is off by a smaller number than tol, stop the algorithm
# maxiter - maximum number of iterations to let the algorithm run
# pyten - Pyten package, it takes long to import in Google Colab, that's why it is
#	  handed as an argument
# threshold - how much around 0 predict draw
# r - parameter r for the tensor completion algorithm
# C - parameter C for the tensor completion algorithm
# 
# Returns achieved accuracy on the predicted part of the last season.
def predict_season(results_files, last_till_index, function, tol, maxiter, pyten, 
                   threshold, r=None, C=None):
  # Full tensor
  ground_truth, teams = get_tensor(results_files)
  # Tensor with missing values to be predicted
  to_complete, teams = get_tensor(results_files, last_till_index)
  tensor_to_file(to_complete, "train_validation.csv")
  # Running tensor completion algorithm
  [Ori, full, Final, Rec] = pyten.UI.basic(file_name='train_validation.csv', 
                                           function_name=function, tol=tol, 
                                           maxiter=maxiter, recover='1', r=r,
                                           C=C)
  os.remove("train_validation.csv")
  # Creating predictions from float numbers
  pred = Rec.data
  pred[pred < -threshold] = -1
  pred[pred > threshold] = 1
  pred[(pred > -1) & (pred < 1)] = 0
  last_season = pd.read_csv(results_files[-1])
  num_matches = last_season.shape[0]
  if isinstance(last_till_index, int):
    start_index = last_till_index
  elif isinstance(last_till_index, float):
    start_index = num_matches * last_till_index
  predicted_matches = last_season[int(start_index):]
  predicted_matches = predicted_matches.reset_index()
  for index in range(predicted_matches.shape[0]):
      row = teams.index(predicted_matches.loc[index, 'HomeTeam'])
      col = teams.index(predicted_matches.loc[index, 'AwayTeam'])
      to_complete[-1][row][col] = 42 # Values to predict
  correct = 0.
  all = 0.
  for i in range(to_complete.shape[1]):
    for j in range(to_complete.shape[2]):
      if to_complete[-1][i][j] == 42: # Values to predict.
        all += 1
        if pred[-1][i][j] == ground_truth[-1][i][j]:
          correct += 1
  return correct / all

# Runs the tensor completion algorithm 20 times for each of given iteration_nums
# as the maximum number of iterations. For each prints the accuracies and their
# variance.
#
# results_files - files of seasons to train on, the last season will be predicted
# last_till_index - index of first match to not include into the last matrix, defines,
#		    which matches will be predicted, if None, all matches will be
# 		    included in the last matrix
# function - tensor completion algorithm to be used
# tol - if the prediction is off by a smaller number than tol, stop the algorithm
# threshold - how much around 0 predict draw
# iteration_nums - numbers to be tested as maximum number of iterations
# pyten - Pyten package, it takes long to import in Google Colab, that's why it is
#	  handed as an argument
# r - parameter r for the tensor completion algorithm
# C - parameter C for the tensor completion algorithm
#
# Returns the accuracies.
def get_scores_iterations(results_files, last_till_index, function, tol, threshold,
                  iteration_nums, pyten, r=None, C=None):
  scores = []
  for iterations in iteration_nums:
    cur_scores = []
    for i in range(20):
      score = predict_season(results_files, last_till_index, function, tol,
                             iterations, pyten, threshold, r, C)
      cur_scores.append(score * 100)
    var = np.var(cur_scores)
    print("iterations: ", iterations, " scores: ", cur_scores, " variance: ", var)
    scores.append(cur_scores)
  return np.array(scores)

# Evaluates the wanted part of the last season of given results files. The model is
# trained on the beginning of the training dataset, it is evaluated on one round
# of the testing dataset and this round is added to the training dataset.
#
# results_files - files of seasons to train on, the last season will be evaluated
# last_till_index - index of first match to not include into the last matrix, defines,
#		    which matches will be evaluated, if None, all matches will be
# 		    included in the last matrix
# function - tensor completion algorithm to be used
# tol - if the prediction is off by a smaller number than tol, stop the algorithm
# maxiter - maximum number of iterations to let the algorithm run
# threshold - how much around 0 predict draw
# pyten - Pyten package, it takes long to import in Google Colab, that's why it is
#	  handed as an argument
# r - parameter r for the tensor completion algorithm
# C - parameter C for the tensor completion algorithm
# return_cm - whether to renurn the confusion matrix
# 
# Returns achieved accuracy on the evaluated part of the last season, possibly
# also the confusion matrix
def test_season(results_files, last_till_index, function, tol, maxiter,
                threshold, pyten, r=None, C=None, return_cm=False):
  test_file = results_files[-1]
  results = pd.read_csv(test_file)
  results = results[['HomeTeam', 'AwayTeam', 'FTR', 'Date']]
  matches_per_round = int(results['HomeTeam'].unique().size / 2)
  start_index = int(results.shape[0] / 2)
  # Rounds of the testing dataset
  slices = testing_functions.get_slices(results, matches_per_round, start_index)
  weighted_sum = 0
  sum = 0
  cm = np.zeros((3, 3))
  for slc in slices:
    slc = slc.reset_index()
    first_index = int(slc.loc[0, 'index'])
    # Full tensor
    ground_truth, teams = get_tensor(results_files)
    # Tensor with missing values to be predicted
    to_complete, teams = get_tensor(results_files, first_index)
    tensor_to_file(to_complete, "testing.csv")
    [Ori, full, Final, Rec] = pyten.UI.basic(file_name='testing.csv', 
                                           function_name=function, tol=tol, 
                                           maxiter=maxiter, recover='1', r=r,
                                           C=C)
    os.remove("testing.csv")
    # Creating predictions from float numbers
    pred = Rec.data
    pred[pred < -threshold] = -1
    pred[pred > threshold] = 1
    pred[(pred > -1) & (pred < 1)] = 0
    to_complete[to_complete == 0] = -1
    for index in range(slc.shape[0]):
      row = teams.index(slc.loc[index, 'HomeTeam'])
      col = teams.index(slc.loc[index, 'AwayTeam'])
      to_complete[-1][row][col] = 42 # Values to be evaluated
    correct = 0.
    all = 0.
    for i in range(to_complete.shape[1]):
      for j in range(to_complete.shape[2]):
        if to_complete[-1][i][j] == 42: # Values to be evaluated
          all += 1
          if pred[-1][i][j] == ground_truth[-1][i][j]:
            correct += 1
	  # Creating the confusion matrix
          if ground_truth[-1][i][j] == 1 and pred[-1][i][j] == 1:
            cm[0][0] += 1
          elif ground_truth[-1][i][j] == 1 and pred[-1][i][j] == 0:
            cm[0][1] += 1
          elif ground_truth[-1][i][j] == 1 and pred[-1][i][j] == -1:
            cm[0][2] += 1
          elif ground_truth[-1][i][j] == 0 and pred[-1][i][j] == 1:
            cm[1][0] += 1
          elif ground_truth[-1][i][j] == 0 and pred[-1][i][j] == 0:
            cm[1][1] += 1
          elif ground_truth[-1][i][j] == 0 and pred[-1][i][j] == -1:
            cm[1][2] += 1
          elif ground_truth[-1][i][j] == -1 and pred[-1][i][j] == 1:
            cm[2][0] += 1
          elif ground_truth[-1][i][j] == -1 and pred[-1][i][j] == 0:
            cm[2][1] += 1
          elif ground_truth[-1][i][j] == -1 and pred[-1][i][j] == -1:
            cm[2][2] += 1
    weighted_sum += (correct / all * slc.shape[0])
    print(correct / all)
    sum += slc.shape[0]
  if return_cm:
    return weighted_sum / sum, cm
  else:
    return weighted_sum / sum

# Plots given confusion matrix with percentages using snss
#
# cm - the confusion matrix to be ploted
#
def plot_cm(cm):
  cm = np.divide(cm.T, cm.sum(axis=1)).T # Calculating the percentages
  plt.figure(figsize=(5,5))
  sns.heatmap(cm, annot=True, xticklabels=['H', 'D', 'A'], yticklabels=['H', 'D', 'A'], fmt=".2f", vmin=0, vmax=1)
  plt.title('Confusion matrix')
  plt.ylabel('True value')
  plt.xlabel('Prediction')

# Creates feature matrices of last season from filled tensor created from given files.
#
# results_files - files of seasons to train on, the last season will be evaluated
# last_till_index - index of first match to not include into the last matrix, defines,
#		    which matches will be filled, if None, all matches will be
# 		    included in the last matrix
# function - tensor completion algorithm to be used
# tol - if the prediction is off by a smaller number than tol, stop the algorithm
# maxiter - maximum number of iterations to let the algorithm run
# threshold - how much around 0 predict draw
# pyten - Pyten package, it takes long to import in Google Colab, that's why it is
#	  handed as an argument
# r - parameter r for the tensor completion algorithm
# C - parameter C for the tensor completion algorithm
# before_threshod - whether to extract the feature vectors before or after applying
# 		    the threshold
#
# Returns the feature matrices and all the teams
def get_svd(results_files, last_till_index, function, tol, maxiter,
                   threshold, pyten, r=None, C=None, before_threshold=False):
  # Tensor to be filled
  to_complete, teams = get_tensor(results_files, last_till_index)
  tensor_to_file(to_complete, "train_validation.csv")
  [Ori, full, Final, Rec] = pyten.UI.basic(file_name='train_validation.csv', 
                                           function_name=function, tol=tol, 
                                           maxiter=maxiter, recover='1', r=r,
                                           C=C)
  os.remove("train_validation.csv")
  # Creating predictions from float numbers
  pred = Rec.data
  if before_threshold == False:
    pred[pred < -threshold] = -1
    pred[pred > threshold] = 1
    pred[(pred > -1) & (pred < 1)] = 0
  last_season = pred[-1]
  # SVD of the last season
  U, Sigma, VT = randomized_svd(last_season, n_components=len(teams))
  # Featue matrices
  H = U @ np.sqrt(np.diag(Sigma))
  A = np.sqrt(np.diag(Sigma)) @ VT
  return H, A, teams

# Saves given feature matrices into csv files. Creates pandas dataframes with teams
# as columns and saves them as csv.
#
# H - home feature matrix to be saved
# A - away feature matrix to be saved
# teams - teams to be used as columns
# H_name - name of a file to save H into
# A_name - name of a file to save A into
def save_matrices(H, A, teams, H_name, A_name):
  dfH = pd.DataFrame(H.T)
  dfH.columns = teams
  dfH.to_csv(H_name)
  dfA = pd.DataFrame(A)
  dfA.columns = teams
  dfA.to_csv(A_name)