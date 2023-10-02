# In this file, functions for data preparation for evaluating the models on testing
# dataset are implemented. There is a function for dividing testing dataset into slices
# and function to prepare the first half of the testing dataset to be added to the training
# data.

import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np

# Divides given results dataset into slices. Each slice consists of matches
# of approximately 1 round. Number of matches per round matches are added
# to the slice. Then all matches, which were played on the same day as the end
# of the round, are also added.
#
# results - dataset with results and features to make slices of
# matches_per_round - number of matches per round
# start_index - index of the first match to start creating slices from
#
# Returns list of creates slices.
def get_slices(results, matches_per_round, start_index):
  results = results[start_index:]
  results = results.reset_index(drop=True)
  slices = []
  cur_index = 0
  while cur_index + matches_per_round < results.shape[0]:
    end_index = cur_index + matches_per_round
    # Date of the end of the round
    end_date = results.loc[end_index, 'Date']
    while end_index < results.shape[0]:
      date = results.loc[end_index, 'Date']
      if date != end_date: # Match is on another day, dont add it to this slice.
        break
      end_index += 1
    slc = results[cur_index:end_index]
    slices.append(slc)
    cur_index = end_index
  if end_index < results.shape[0]:
    slices.append(results[end_index:])
  return slices

# Extract the first half of the testing season to be added to the training data.
#
# results - dataset with results and features of the testing season
# last_index - index of the last match to be appended to the training data
# drop_draws - whether to ignore the matches, which ended with draw
#
# Returns list the part of testing season to be added as features X_test_to_append
# and explained variable y_test_to_append.
def prepare_test_to_append(results_test, last_index, drop_draws=False):
  test_to_append = results_test[:last_index]
  if drop_draws: 
    test_to_append = test_to_append[test_to_append['FTR'] != 2]
  X_test_to_append = test_to_append.drop(['FTR', 'Date'], axis=1)
  y_test_to_append = test_to_append['FTR']
  return X_test_to_append, y_test_to_append