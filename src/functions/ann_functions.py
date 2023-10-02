# In this file, functions for the baseline using ANN are implemented.
# Here belong functions for gathering the original features according to the paper and
# a function for creating the ANN model.
#
# The baseline is implemented according to the paper An Improved Prediction System
# for Football a Match Result bu C. P. Igiri and E. O. Nwachukwu from 2014. # The authors
# used several features achieved from the previous results and from
# the games statistics. I kept the features as similar as possible, but a few ones I had to
# replace. I downloaded the data from https://www.football-data.co.uk/.

import pandas as pd
from datetime import datetime
import requests
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import *

# Gathers average goals of teams from given dataset.
#
# cur_season_dataset - dataset to gather average goals from
# start_index - index of the first match to gather average goals for
#
# Returns list of home teams average scored goals and list of away teams average
# scored goals
def get_avg_goals(cur_season_dataset, start_index):
  home_scored_goalss = []
  away_scored_goalss = []
  for index in range(start_index, cur_season_dataset.shape[0]):
    home_team = cur_season_dataset.loc[index, 'HomeTeam']
    away_team = cur_season_dataset.loc[index, 'AwayTeam']
    # Taking only already finished matches.
    cur_season_dataset_finished = cur_season_dataset[:index]
    home_team_home_matches = cur_season_dataset_finished[
        cur_season_dataset_finished['HomeTeam'] == home_team]
    home_team_away_matches = cur_season_dataset_finished[
        cur_season_dataset_finished['AwayTeam'] == home_team]
    away_team_home_matches = cur_season_dataset_finished[
        cur_season_dataset_finished['HomeTeam'] == away_team]
    away_team_away_matches = cur_season_dataset_finished[
        cur_season_dataset_finished['AwayTeam'] == away_team]

    home_scored_goals_sum = home_team_home_matches['FTHG'].sum()
    home_scored_goals_sum += home_team_away_matches['FTAG'].sum()
    away_scored_goals_sum = away_team_home_matches['FTHG'].sum()
    away_scored_goals_sum += away_team_away_matches['FTAG'].sum()
    home_matches_num = home_team_home_matches.shape[0] + \
                       home_team_away_matches.shape[0]
    away_matches_num = away_team_home_matches.shape[0] + \
                       away_team_away_matches.shape[0]
    # Calculating and appending the averages.
    home_scored_goalss.append(home_scored_goals_sum / home_matches_num)
    away_scored_goalss.append(away_scored_goals_sum / away_matches_num)

  return home_scored_goalss, away_scored_goalss

# Gathers average shots of teams from given dataset.
#
# cur_season_dataset - dataset to gather average shots from
# start_index - index of the first match to gather average shots for
#
# Returns list of home teams average fired shots and list of away teams average
# fired shots
def get_avg_shots(cur_season_dataset, start_index):
  home_fired_shots = []
  away_fired_shots = []
  for index in range(start_index, cur_season_dataset.shape[0]):
    home_team = cur_season_dataset.loc[index, 'HomeTeam']
    away_team = cur_season_dataset.loc[index, 'AwayTeam']
    # Taking only already finished matches.
    cur_season_dataset_finished = cur_season_dataset[:index]
    home_team_home_matches = cur_season_dataset_finished[
        cur_season_dataset_finished['HomeTeam'] == home_team]
    home_team_away_matches = cur_season_dataset_finished[
        cur_season_dataset_finished['AwayTeam'] == home_team]
    away_team_home_matches = cur_season_dataset_finished[
        cur_season_dataset_finished['HomeTeam'] == away_team]
    away_team_away_matches = cur_season_dataset_finished[
        cur_season_dataset_finished['AwayTeam'] == away_team]

    home_fired_shots_sum = home_team_home_matches['HS'].sum()
    home_fired_shots_sum += home_team_away_matches['AS'].sum()
    away_fired_shots_sum = away_team_home_matches['HS'].sum()
    away_fired_shots_sum += away_team_away_matches['AS'].sum()
    home_matches_num = home_team_home_matches.shape[0] + \
                       home_team_away_matches.shape[0]
    away_matches_num = away_team_home_matches.shape[0] + \
                       away_team_away_matches.shape[0]
    # Calculating and appends the averages.
    home_fired_shots.append(home_fired_shots_sum / home_matches_num)
    away_fired_shots.append(away_fired_shots_sum / away_matches_num)

  return home_fired_shots, away_fired_shots

# Gathers average corners of teams from given dataset.
#
# cur_season_dataset - dataset to gather average corners from
# start_index - index of the first match to gather average corners for
#
# Returns list of home teams average corners and list of away teams average
# corners
def get_avg_corners(cur_season_dataset, start_index):
  home_corners = []
  away_corners = []
  for index in range(start_index, cur_season_dataset.shape[0]):
    home_team = cur_season_dataset.loc[index, 'HomeTeam']
    away_team = cur_season_dataset.loc[index, 'AwayTeam']
    # Taking only already finished matches.
    cur_season_dataset_finished = cur_season_dataset[:index]
    home_team_home_matches = cur_season_dataset_finished[
        cur_season_dataset_finished['HomeTeam'] == home_team]
    home_team_away_matches = cur_season_dataset_finished[
        cur_season_dataset_finished['AwayTeam'] == home_team]
    away_team_home_matches = cur_season_dataset_finished[
        cur_season_dataset_finished['HomeTeam'] == away_team]
    away_team_away_matches = cur_season_dataset_finished[
        cur_season_dataset_finished['AwayTeam'] == away_team]

    home_corners_sum = home_team_home_matches['HC'].sum()
    home_corners_sum += home_team_away_matches['AC'].sum()
    away_corners_sum = away_team_home_matches['HC'].sum()
    away_corners_sum += away_team_away_matches['AC'].sum()
    home_matches_num = home_team_home_matches.shape[0] + \
                       home_team_away_matches.shape[0]
    away_matches_num = away_team_home_matches.shape[0] + \
                       away_team_away_matches.shape[0]
    # Calculating and appends the averages.
    home_corners.append(home_corners_sum / home_matches_num)
    away_corners.append(away_corners_sum / away_matches_num)

  return home_corners, away_corners

# Gathers bookmakers odds for matches from given dataset.
#
# cur_season_dataset - dataset to gather odds from
# start_index - index of the first match to gather odd for
#
# Returns list of home team odds and list of away team odds
def get_odds(cur_season_dataset, start_index):
  home_odds = []
  away_odds = []
  for index in range(start_index, cur_season_dataset.shape[0]):
    home_odd = cur_season_dataset.loc[index, 'B365H']
    away_odd = cur_season_dataset.loc[index, 'B365A']
    home_odds.append(home_odd)
    away_odds.append(away_odd)

  return home_odds, away_odds

# Downloads the attack stats from FIFA game of given league and season of 
# teams from given dataset.
#
# cur_season_dataset - dataset to gather stats for teams from
# fifa_league_id - id of the wanted league on the fifaindex website
# season - last 2 gigits of the year in which the wanted season ends
# team_names_map - mapping from dataset to FIFA team names
# secondary_team_names_map - mapping to use when team_names_map mapping 
#                            does not work
#
# Returns list of home team attack strenght and list of away team
# attack strenght
def get_attacks(cur_season_dataset, start_index, fifa_league_id,
                       season, 
                       team_names_map,
                       secondary_team_names_map,
		       ternary_team_names_map=None):
  home_attacks = []
  away_attacks = []
  url = ('https://www.fifaindex.com/teams/fifa' + str(season) + 
         '/?league='+ str(fifa_league_id))
  # Acting as a browser, otherwise the permission is denied.
  header = {
    "User-Agent": ("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                   "(KHTML, like Gecko) Chrome/50.0.2661.75 Safari/537.36"),
    "X-Requested-With": "XMLHttpRequest"
  }
  r = requests.get(url, headers=header)
  dfs = pd.read_html(r.text)
  fifa_data = dfs[0]
  # Getting only the Attack and Defense stats.
  fifa_stats = fifa_data[['Name', 'ATT']].dropna().reset_index(drop=True)

  for index in range(start_index, cur_season_dataset.shape[0]):
    # Mapping the team name from the results file name to the FIFA name.
    home_team = team_names_map[cur_season_dataset.loc[index, 'HomeTeam']]
    away_team = team_names_map[cur_season_dataset.loc[index, 'AwayTeam']]
    if fifa_stats[fifa_stats['Name'] == home_team].shape[0] == 0:
      # If the mapped team name is not in the FIFA stats, use secondary mapping
      home_team = secondary_team_names_map[cur_season_dataset.loc[index,
                                                                  'HomeTeam']]
      if fifa_stats[fifa_stats['Name'] == home_team].shape[0] == 0:
        home_team = ternary_team_names_map[cur_season_dataset.loc[index, 'HomeTeam']]
    away_team = team_names_map[cur_season_dataset.loc[index, 'AwayTeam']]
    if fifa_stats[fifa_stats['Name'] == away_team].shape[0] == 0:
      away_team = secondary_team_names_map[cur_season_dataset.loc[index,
                                                                  'AwayTeam']]
      if fifa_stats[fifa_stats['Name'] == away_team].shape[0] == 0:
        away_team = ternary_team_names_map[cur_season_dataset.loc[index, 'AwayTeam']]
    home_attack = fifa_stats[fifa_stats['Name'] == home_team
                             ].reset_index().loc[0, 'ATT']
    away_attack = fifa_stats[fifa_stats['Name'] == away_team
                             ].reset_index().loc[0, 'ATT']
    home_attacks.append(home_attack)
    away_attacks.append(away_attack)

  return home_attacks, away_attacks

# Gathers average fauls of teams from given dataset.
#
# cur_season_dataset - dataset to gather average fauls from
# start_index - index of the first match to gather average fauls for
#
# Returns list of home teams average commited fauls and list of away teams average
# commited fauls
def get_avg_fauls(cur_season_dataset, start_index):
  home_commited_fauls = []
  away_commited_fauls = []
  for index in range(start_index, cur_season_dataset.shape[0]):
    home_team = cur_season_dataset.loc[index, 'HomeTeam']
    away_team = cur_season_dataset.loc[index, 'AwayTeam']
    # Taking only already finished matches.
    cur_season_dataset_finished = cur_season_dataset[:index]
    home_team_home_matches = cur_season_dataset_finished[
        cur_season_dataset_finished['HomeTeam'] == home_team]
    home_team_away_matches = cur_season_dataset_finished[
        cur_season_dataset_finished['AwayTeam'] == home_team]
    away_team_home_matches = cur_season_dataset_finished[
        cur_season_dataset_finished['HomeTeam'] == away_team]
    away_team_away_matches = cur_season_dataset_finished[
        cur_season_dataset_finished['AwayTeam'] == away_team]

    home_commited_fauls_sum = home_team_home_matches['HF'].sum()
    home_commited_fauls_sum += home_team_away_matches['AF'].sum()
    away_commited_fauls_sum = away_team_home_matches['HF'].sum()
    away_commited_fauls_sum += away_team_away_matches['AF'].sum()
    home_matches_num = home_team_home_matches.shape[0] + \
                       home_team_away_matches.shape[0]
    away_matches_num = away_team_home_matches.shape[0] + \
                       away_team_away_matches.shape[0]
    # Calculating and appends the averages.
    home_commited_fauls.append(home_commited_fauls_sum / home_matches_num)
    away_commited_fauls.append(away_commited_fauls_sum / away_matches_num)

  return home_commited_fauls, away_commited_fauls

# Gathers number of days from last match of teams from given dataset.
#
# cur_season_dataset - dataset to gather number of days from
# start_index - index of the first match to gather number of days for
#
# Returns list of number of days from last match for home teams and list
# of number of days from last match for away teams.
def get_days_rest(cur_season_dataset, start_index):
  home_days_rest = []
  away_days_rest = []
  # Finding out which format of date to use.
  date = cur_season_dataset.loc[0, 'Date']
  if len(date) == 8:
    date_format = '%d/%m/%y'
  else:
    date_format = '%d/%m/%Y'
  for index in range(start_index, cur_season_dataset.shape[0]):
    home_team = cur_season_dataset.loc[index, 'HomeTeam']
    away_team = cur_season_dataset.loc[index, 'AwayTeam']
    # Date of the current match
    match_date = datetime.strptime(cur_season_dataset.loc[index, 'Date'], 
                                   date_format)
    # Taking only already finished matches.
    cur_season_dataset_finished = cur_season_dataset[:index]
    home_team_matches = cur_season_dataset_finished[
        (cur_season_dataset_finished['HomeTeam'] == home_team) | 
        (cur_season_dataset_finished['AwayTeam'] == home_team)]
    away_team_matches = cur_season_dataset_finished[
        (cur_season_dataset_finished['HomeTeam'] == away_team) | 
        (cur_season_dataset_finished['AwayTeam'] == away_team)]
    # Reversing order of matches to make them accessible through index.
    home_team_matches_ordered = home_team_matches.iloc[::-1].reset_index()
    away_team_matches_ordered = away_team_matches.iloc[::-1].reset_index()

    # Date of the last match of each team
    home_last_date = home_team_matches_ordered.loc[0, 'Date']
    away_last_date = away_team_matches_ordered.loc[0, 'Date']
    # Calculating number of days and appends them.
    home_days_rest.append((match_date - datetime.strptime(home_last_date,
                                                          date_format)).days)
    away_days_rest.append((match_date - datetime.strptime(away_last_date, 
                                                          date_format)).days)
  return home_days_rest, away_days_rest

# Gathers win percentages of teams from given dataset.
#
# cur_season_dataset - dataset to gather percentages from
# start_index - index of the first match to gather percentages for
#
# Returns list of win percentages of home teams and list of win percentages of
# away teams
def get_percentages(cur_season_dataset, start_index):
  home_win_percentage = []
  away_win_percentage = []
  for index in range(start_index, cur_season_dataset.shape[0]):
    home_team = cur_season_dataset.loc[index, 'HomeTeam']
    away_team = cur_season_dataset.loc[index, 'AwayTeam']
    # Taking only already finished matches.
    cur_season_dataset_finished = cur_season_dataset[:index]
    home_team_home_matches = cur_season_dataset_finished[
        cur_season_dataset_finished['HomeTeam'] == home_team]
    home_team_away_matches = cur_season_dataset_finished[
        cur_season_dataset_finished['AwayTeam'] == home_team]
    away_team_home_matches = cur_season_dataset_finished[
        cur_season_dataset_finished['HomeTeam'] == away_team]
    away_team_away_matches = cur_season_dataset_finished[
        cur_season_dataset_finished['AwayTeam'] == away_team]
    
    home_wins = \
      home_team_home_matches[home_team_home_matches['FTR'] == 'H'].shape[0] + \
      home_team_away_matches[home_team_away_matches['FTR'] == 'A'].shape[0]
    away_wins = \
      away_team_home_matches[away_team_home_matches['FTR'] == 'H'].shape[0] + \
      away_team_away_matches[away_team_away_matches['FTR'] == 'A'].shape[0]

    home_matches_num = home_team_home_matches.shape[0] + \
                       home_team_away_matches.shape[0]
    away_matches_num = away_team_home_matches.shape[0] + \
                       away_team_away_matches.shape[0]
    # Calculating the percentages and appends them.
    home_win_percentage.append(home_wins / home_matches_num)
    away_win_percentage.append(away_wins / away_matches_num)

  return home_win_percentage, away_win_percentage

# Gets match result ago matches ago of given team
#
# prev_matches - dataset with all matches of the team
# team_name - name of the team
# ago - number of matches ago of wanted result
#
# Returns result of match ago matches ago
def get_match_result(prev_matches, team_name, ago):
  # Reversing order of matches to make them accessible through index.
  prev_matches_ordered = prev_matches.iloc[::-1].reset_index()
  result_ago = prev_matches_ordered.loc[ago - 1, 'FTR']
  
  # The result for the wanted team depends also on whether the wanted team was
  # home or away team in the wanted match.
  # Win is represented as 1, draw as 0 and loss as -1.
  if result_ago == 'D':
      return 0
  elif result_ago == 'H':
    if prev_matches_ordered.loc[ago - 1, 'HomeTeam'] == team_name:
      return 1
    else:
      return -1
  else:
    if prev_matches_ordered.loc[ago - 1, 'HomeTeam'] == team_name:
      return -1
    else:
      return 1

# Gathers streaks of teams from given dataset. 0 means last match was a draw,
# positive number means streak of wins and negative number means
# streak of losses
#
# cur_season_dataset - dataset to gather streaks from
# start_index - index of the first match to gather streaks for
#
# Returns list of home teams streaks and list of away
# teams streaks.
def get_streaks(cur_season_dataset, start_index):
  home_streaks = []
  away_streaks = []
  for index in range(start_index, cur_season_dataset.shape[0]):
    home_team = cur_season_dataset.loc[index, 'HomeTeam']
    away_team = cur_season_dataset.loc[index, 'AwayTeam']
    # Taking only already finished matches.
    cur_season_dataset_finished = cur_season_dataset[:index]
    home_team_matches = cur_season_dataset_finished[
        (cur_season_dataset_finished['HomeTeam'] == home_team) | 
        (cur_season_dataset_finished['AwayTeam'] == home_team)]
    away_team_matches = cur_season_dataset_finished[
        (cur_season_dataset_finished['HomeTeam'] == away_team) | 
        (cur_season_dataset_finished['AwayTeam'] == away_team)]

    home_last_result = get_match_result(home_team_matches, home_team, 1)
    away_last_result = get_match_result(away_team_matches, away_team, 1)
    home_streak = home_last_result
    away_streak = away_last_result
    ago = 2
    while (ago <= home_team_matches.shape[0]):
      home_ago_result = get_match_result(home_team_matches, home_team, ago)
      if home_last_result == home_ago_result:
        home_streak += home_ago_result
      else:
        break
      ago +=1
    ago = 2
    while (ago <= away_team_matches.shape[0]):
      away_ago_result = get_match_result(away_team_matches, away_team, ago)
      if away_last_result == away_ago_result:
        away_streak += away_ago_result
      else:
        break
      ago +=1
    home_streaks.append(home_streak)
    away_streaks.append(away_streak)
  
  return home_streaks, away_streaks

# Creates dataset from given filename and process it. Gathers the wanted
# features for the current season file.
#
# cur_season_file - filename of the wanted season from football-data website
# fifa_league_id - id of the wanted league on the fifaindex website
# season - last 2 gigits of the year in which the wanted season ends
# team_names_map - mapping from current season file to FIFA team names
# secondary_team_names_map, ternary_team_names_map - mappings to use when team_names_map mapping 
#                            does not work
# return_dates - whether to return dates with the features
# include_stats - wheteher to include shots, corners and falus, which are inaccessible
#		  for the BSA
#
# Returns dataset with game results and the wanted features and number
# of matches per round.
def create_data_single(cur_season_file, fifa_league_id,
                       season, 
                       team_names_map,
                       secondary_team_names_map,
		       ternary_team_names_map=None,
                       return_dates=False,
		       include_stats=True):
  cur_season_dataset = pd.read_csv(cur_season_file)
  
  results = cur_season_dataset[['FTR', 'HomeTeam', 'Date']]
  matches_per_round = int(results['HomeTeam'].unique().size / 2)
  # Starting a few rounds later to have previous matches results
  start_index = 6 * matches_per_round
  results = results[start_index:]
  # Home win is repredsented as 1, away win as -1, draw as 0.
  results['FTR'].replace('H', 1, inplace=True)
  results['FTR'].replace('A', -1, inplace=True)
  results['FTR'].replace('D', 0, inplace=True)
  results.drop('HomeTeam', axis=1, inplace=True)
  
  home_scored_goalss, away_scored_goalss = get_avg_goals(cur_season_dataset,
                                                             start_index)
  results['HG'] = home_scored_goalss
  results['AG'] = away_scored_goalss

  if include_stats:
    home_fired_shots, away_fired_shots = get_avg_shots(cur_season_dataset,
                                                           start_index)
    results['HS'] = home_fired_shots
    results['AS'] = away_fired_shots

    home_corners, away_corners = get_avg_corners(cur_season_dataset,
                                                          start_index)
    results['HC'] = home_corners
    results['AC'] = away_corners

    home_attacks, away_attacks = get_attacks(cur_season_dataset,
                                           start_index, fifa_league_id,
                       season, 
                       team_names_map,
                       secondary_team_names_map,
		       ternary_team_names_map=ternary_team_names_map)
    results['HAT'] = home_attacks
    results['AAT'] = away_attacks

    home_commited_fauls, away_commited_fauls = get_avg_fauls(cur_season_dataset,
                                                           start_index)
    results['HF'] = home_commited_fauls
    results['AF'] = away_commited_fauls

  home_odds, away_odds = get_odds(cur_season_dataset,
                                                           start_index)
  results['HOD'] = home_odds
  results['AOD'] = away_odds

  home_days_rest, away_days_rest = get_days_rest(cur_season_dataset,
                                                 start_index)
  results['HDR'] = home_days_rest
  results['ADR'] = away_days_rest

  home_win_percentage, away_win_percentage = get_percentages(cur_season_dataset,
                                                             start_index)
  results['HWP'] = home_win_percentage
  results['AWP'] = away_win_percentage

  home_streaks, away_streaks = get_streaks(cur_season_dataset, start_index)
  results['HSK'] = home_streaks
  results['ASK'] = away_streaks

  if return_dates == False:
    results = results.drop('Date', axis=1)
  return results, matches_per_round

# Creates an ANN model with 3 dense layers with 64, 32 and 16 neurons. Also applies dropout
# after each dense layer. In each layer relu activation function is used. In the end,
# the output layer with 3 neurons is added, because the problem is a 3-class classification.
# Softmax activation function is applied in the output layer.
#
# input_shape - shape of the input data, used as the shape of the input layer
#
# Returns the created model.
def func_model(input_shape):
    input_layer = Input(shape=input_shape)
    flatten = Flatten()(input_layer)
    dense1 = Dense(64, activation='relu')(flatten)
    dropout1 = Dropout(rate=0.2)(dense1)
    dense2 = Dense(32, activation='relu')(dropout1)
    dropout2 = Dropout(rate=0.2)(dense2)
    dense3 = Dense(16, activation='relu')(dropout2)
    dropout3 = Dropout(rate=0.2)(dense3)
    concat = Concatenate()([dense1, dense2, dense3, dropout1, dropout2, dropout3])
    output_layer = Dense(3, activation='softmax')(concat)
    model = Model(inputs=input_layer, outputs=output_layer)
    return model