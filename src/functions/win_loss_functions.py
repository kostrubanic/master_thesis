# In this file, functions for creating the Win-loss feature set are implemented.

import pandas as pd
from datetime import datetime

# Gathers win, lose and draw percentages of teams from given dataset.
#
# cur_season_dataset - dataset to gather percentages from
# start_index - index of the first match to gather percentages for
#
# Returns list of win percentages of home teams, list of win percentages of
# away teams, list of lose percentages of home teams, list of lose percentages
# of away teams, list of draw percentages of home teams and list of draw
# percentages of away teams
def get_percentages(cur_season_dataset, start_index):
  home_win_percentage = []
  away_win_percentage = []
  home_lose_percentage = []
  away_lose_percentage = []
  home_draw_percentage = []
  away_draw_percentage = []
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
    home_losses = \
      home_team_home_matches[home_team_home_matches['FTR'] == 'A'].shape[0] + \
      home_team_away_matches[home_team_away_matches['FTR'] == 'H'].shape[0]
    away_losses = \
      away_team_home_matches[away_team_home_matches['FTR'] == 'A'].shape[0] + \
      away_team_away_matches[away_team_away_matches['FTR'] == 'H'].shape[0]
    home_draws = \
      home_team_home_matches[home_team_home_matches['FTR'] == 'D'].shape[0] + \
      home_team_away_matches[home_team_away_matches['FTR'] == 'D'].shape[0]
    away_draws = \
      away_team_home_matches[away_team_home_matches['FTR'] == 'D'].shape[0] + \
      away_team_away_matches[away_team_away_matches['FTR'] == 'D'].shape[0]

    home_matches_num = home_team_home_matches.shape[0] + \
                       home_team_away_matches.shape[0]
    away_matches_num = away_team_home_matches.shape[0] + \
                       away_team_away_matches.shape[0]
    # Calculating the percentages and appends them.
    if home_matches_num == 0:
      # In case there was no previous matches, set the percentages
      # 0.4, 0.4 and 0.2.
      home_win_percentage.append(0.4)
      home_lose_percentage.append(0.4)
      home_draw_percentage.append(0.2)
    else:
      home_win_percentage.append(home_wins / home_matches_num)
      home_lose_percentage.append(home_losses / home_matches_num)
      home_draw_percentage.append(home_draws / home_matches_num)
    if away_matches_num == 0:
      # In case there was no previous matches, set the percentages
      # 0.4, 0.4 and 0.2.
      away_win_percentage.append(0.4)
      away_lose_percentage.append(0.4)
      away_draw_percentage.append(0.2)
    else:
      away_win_percentage.append(away_wins / away_matches_num)
      away_lose_percentage.append(away_losses / away_matches_num)
      away_draw_percentage.append(away_draws / away_matches_num)

  return home_win_percentage, away_win_percentage, home_lose_percentage, \
         away_lose_percentage, home_draw_percentage, away_draw_percentage

# Gathers win, lose and draw percentages of teams on same ground from given
# dataset.
#
# cur_season_dataset - dataset to gather percentages from
# start_index - index of the first match to gather percentages for
#
# Returns list of win percentages of home teams on the home ground, list of lose
# percentages of home teams on the home ground, list of draw percentages of home
# teams on the home ground, list of win percentages of away teams on the away
# ground, list of lose percentages of away teams on the away ground and  list of
# draw percentages of away teams on the away ground.
def get_ground_percentages(cur_season_dataset, start_index):
  home_ground_win_percentage = []
  away_ground_win_percentage = []
  home_ground_lose_percentage = []
  away_ground_lose_percentage = []
  home_ground_draw_percentage = []
  away_ground_draw_percentage = []
  for index in range(start_index, cur_season_dataset.shape[0]):
    home_team = cur_season_dataset.loc[index, 'HomeTeam']
    away_team = cur_season_dataset.loc[index, 'AwayTeam']
    # Taking only already finished matches.
    cur_season_dataset_finished = cur_season_dataset[:index]
    home_team_home_matches = cur_season_dataset_finished[
        cur_season_dataset_finished['HomeTeam'] == home_team]
    away_team_away_matches = cur_season_dataset_finished[
        cur_season_dataset_finished['AwayTeam'] == away_team]
    
    home_ground_wins = \
      home_team_home_matches[home_team_home_matches['FTR'] == 'H'].shape[0]
    home_ground_losses = \
      home_team_home_matches[home_team_home_matches['FTR'] == 'A'].shape[0]
    home_ground_draws = \
      home_team_home_matches[home_team_home_matches['FTR'] == 'D'].shape[0]
    away_ground_wins = \
      away_team_away_matches[away_team_away_matches['FTR'] == 'A'].shape[0]
    away_ground_losses = \
      away_team_away_matches[away_team_away_matches['FTR'] == 'H'].shape[0]
    away_ground_draws = \
      away_team_away_matches[away_team_away_matches['FTR'] == 'D'].shape[0]

    home_matches_num = home_team_home_matches.shape[0]
    away_matches_num = away_team_away_matches.shape[0]
    # Calculating the percentages and appends them.
    if home_matches_num == 0:
      # In case there was no previous matches, set the percentages
      # 0.4, 0.4 and 0.2.
      home_ground_win_percentage.append(0.4)
      home_ground_lose_percentage.append(0.4)
      home_ground_draw_percentage.append(0.2)
    else:
      home_ground_win_percentage.append(home_ground_wins / home_matches_num)
      home_ground_lose_percentage.append(home_ground_losses / home_matches_num)
      home_ground_draw_percentage.append(home_ground_draws / home_matches_num)
    if away_matches_num == 0:
      # In case there was no previous matches, set the percentages
      # 0.4, 0.4 and 0.2.
      away_ground_win_percentage.append(0.4)
      away_ground_lose_percentage.append(0.4)
      away_ground_draw_percentage.append(0.2)
    else:
      away_ground_win_percentage.append(away_ground_wins / away_matches_num)
      away_ground_lose_percentage.append(away_ground_losses / away_matches_num)
      away_ground_draw_percentage.append(away_ground_draws / away_matches_num)

  return home_ground_win_percentage, home_ground_lose_percentage, \
         home_ground_draw_percentage, away_ground_win_percentage,  \
         away_ground_lose_percentage, away_ground_draw_percentage

# Gathers home win, away win and draw percentages of the previous encounters of
# the teams on the same ground
#
# cur_season_dataset - dataset to gather percentages from
# prev_seasons_datatsets - datasets of previous seasons to also gather
# percentages from
# start_index - index of the first match to gather percentages for
#
# Returns lists of home win, away win and draw percentages of the previous
# encounters of the teams on the same ground
def get_same_encounters_percentages(cur_season_dataset, 
                                    prev_seasons_datatsets, start_index):
  same_encounters_home_win_percentage = []
  same_encounters_away_win_percentage = []
  same_encounters_draw_percentage = []
  if prev_seasons_datatsets == []:
    all_prev_seasons_dataset = None
  else:
    all_prev_seasons_dataset = pd.concat(prev_seasons_datatsets)
  for index in range(start_index, cur_season_dataset.shape[0]):
    home_team = cur_season_dataset.loc[index, 'HomeTeam']
    away_team = cur_season_dataset.loc[index, 'AwayTeam']
    # Taking only already finished matches.
    cur_season_dataset_finished = cur_season_dataset[:index]
    # Getting dataset of all the previous seasons and the finished part of the
    # current season.
    if prev_seasons_datatsets == []:
      all_finished_dataset = cur_season_dataset_finished
    else:
      all_finished_dataset = \
        pd.concat([all_prev_seasons_dataset, cur_season_dataset_finished])
    same_encounters_matches = all_finished_dataset[
        (all_finished_dataset['HomeTeam'] == home_team) &
        (all_finished_dataset['AwayTeam'] == away_team)]
    
    same_encounters_home_wins = \
      same_encounters_matches[same_encounters_matches['FTR'] == 'H'].shape[0]
    same_encounters_away_wins = \
      same_encounters_matches[same_encounters_matches['FTR'] == 'A'].shape[0]
    same_encounters_draws = \
      same_encounters_matches[same_encounters_matches['FTR'] == 'D'].shape[0]

    same_encounters_num = same_encounters_matches.shape[0]
    # In case there was no previous encounter, set the percentages
    # 0.4, 0.4 and 0.2.
    if same_encounters_num == 0:
      same_encounters_home_win_percentage.append(0.4)
      same_encounters_away_win_percentage.append(0.4)
      same_encounters_draw_percentage.append(0.2)
    else:
      # Calculating the percentages and appends them.
      same_encounters_home_win_percentage.append(
          same_encounters_home_wins / same_encounters_num)
      same_encounters_away_win_percentage.append(
          same_encounters_away_wins / same_encounters_num)
      same_encounters_draw_percentage.append(
          same_encounters_draws / same_encounters_num)

  return same_encounters_home_win_percentage, \
         same_encounters_away_win_percentage, same_encounters_draw_percentage

# Gathers home win, away win and draw percentages of the previous encounters of
# the teams on both grounds
#
# cur_season_dataset - dataset to gather percentages from
# prev_seasons_datatsets - datasets of previous seasons to also gather
# percentages from
# start_index - index of the first match to gather percentages for
#
# Returns lists of home win, away win and draw percentages of the previous
# encounters of the teams on both ground
def get_both_encounters_percentages(cur_season_dataset, 
                                    prev_seasons_datatsets, start_index):
  both_encounters_home_win_percentage = []
  both_encounters_away_win_percentage = []
  both_encounters_draw_percentage = []
  if prev_seasons_datatsets == []:
    all_prev_seasons_dataset = None
  else:
    all_prev_seasons_dataset = pd.concat(prev_seasons_datatsets)
  for index in range(start_index, cur_season_dataset.shape[0]):
    home_team = cur_season_dataset.loc[index, 'HomeTeam']
    away_team = cur_season_dataset.loc[index, 'AwayTeam']
    # Taking only already finished matches.
    cur_season_dataset_finished = cur_season_dataset[:index]
    # Getting dataset of all the previous seasons and the finished part of the
    # current season.
    if prev_seasons_datatsets == []:
      all_finished_dataset = cur_season_dataset_finished
    else:
      all_finished_dataset = \
        pd.concat([all_prev_seasons_dataset, cur_season_dataset_finished])
    same_encounters_matches = all_finished_dataset[
        (all_finished_dataset['HomeTeam'] == home_team) &
        (all_finished_dataset['AwayTeam'] == away_team)]
    # Matches on the away teams ground
    opposite_encounters_matches = all_finished_dataset[
        (all_finished_dataset['HomeTeam'] == away_team) &
        (all_finished_dataset['AwayTeam'] == home_team)]
    
    both_encounters_home_wins = \
      same_encounters_matches[
          same_encounters_matches['FTR'] == 'H'].shape[0] + \
      opposite_encounters_matches[
          opposite_encounters_matches['FTR'] == 'A'].shape[0]
    both_encounters_away_wins = \
      same_encounters_matches[
          same_encounters_matches['FTR'] == 'A'].shape[0] + \
      opposite_encounters_matches[
          opposite_encounters_matches['FTR'] == 'H'].shape[0]
    both_encounters_draws = \
      same_encounters_matches[
          same_encounters_matches['FTR'] == 'D'].shape[0] + \
      opposite_encounters_matches[
          opposite_encounters_matches['FTR'] == 'D'].shape[0]

    both_encounters_num = same_encounters_matches.shape[0] + \
                          opposite_encounters_matches.shape[0]
    # In case there was no previous encounter, set the percentages
    # 0.4, 0.4 and 0.2.
    if both_encounters_num == 0:
      both_encounters_home_win_percentage.append(0.4)
      both_encounters_away_win_percentage.append(0.4)
      both_encounters_draw_percentage.append(0.2)
    else:
      # Calculating the percentages and appends them.
      both_encounters_home_win_percentage.append(
          both_encounters_home_wins / both_encounters_num)
      both_encounters_away_win_percentage.append(
          both_encounters_away_wins / both_encounters_num)
      both_encounters_draw_percentage.append(
          both_encounters_draws / both_encounters_num)

  return both_encounters_home_win_percentage, \
         both_encounters_away_win_percentage, both_encounters_draw_percentage

# Creates datasets from given filenames and process them. Gathers the wanted
# features for the current season file.
#
# cur_season_file - filename of the wanted season from football-data website
# prev_season_file - filename of the previous season from football-data website,
#                    used for feature was in lower league
# other_prev_seasons_files - filenames of the other previous seasons from
#                            football-data website, used for percentages of 
#                            previous encounters
# return_dates - whether to return dates with the features
# drop_draws - whether to ignore the matches, which ended with draw
# skip_rounds - how many rounds from the beginning of the season to skip
# return_names - whether to return names of the teams with the features
#
# Returns dataset with game results and the wanted features and number
# of matches per round.
def create_data_single(cur_season_file, other_prev_seasons_files,
		       return_dates=False, drop_draws=False, skip_rounds=0,
		       return_names=False):
  cur_season_dataset = pd.read_csv(cur_season_file)
  all_prev_seasons_datasets = []
  for file in other_prev_seasons_files:
    dataset = pd.read_csv(file)
    all_prev_seasons_datasets.append(dataset)
  
  results = cur_season_dataset[['FTR', 'HomeTeam', 'AwayTeam', 'Date']]
  matches_per_round = int(results['HomeTeam'].unique().size / 2)
  # Starting a few rounds later to have previous matches results
  start_index = skip_rounds * matches_per_round
  results = results[start_index:]
  # Home win is repredsented as 1, away win as -1, draw as 0.
  results['FTR'].replace('H', 1, inplace=True)
  results['FTR'].replace('A', -1, inplace=True)
  results['FTR'].replace('D', 0, inplace=True)

  home_win_percentage, away_win_percentage, home_lose_percentage, \
  away_lose_percentage, home_draw_percentage, \
  away_draw_percentage = get_percentages(cur_season_dataset, start_index)
  results['HWP'] = home_win_percentage
  results['AWP'] = away_win_percentage
  results['HLP'] = home_lose_percentage
  results['ALP'] = away_lose_percentage
  results['HDP'] = home_draw_percentage
  results['ADP'] = away_draw_percentage

  home_ground_win_percentage, home_ground_lose_percentage, \
  home_ground_draw_percentage, away_ground_win_percentage,  \
  away_ground_lose_percentage, \
  away_ground_draw_percentage = get_ground_percentages(cur_season_dataset,
                                                       start_index)
  results['HGWP'] = home_ground_win_percentage
  results['HGLP'] = home_ground_lose_percentage
  results['HGDP'] = home_ground_draw_percentage
  results['AGWP'] = away_ground_win_percentage
  results['AGLP'] = away_ground_lose_percentage
  results['AGDP'] = away_ground_draw_percentage

  same_encounters_home_win_percentage, \
  same_encounters_away_win_percentage, same_encounters_draw_percentage = \
    get_same_encounters_percentages(cur_season_dataset, 
                                    all_prev_seasons_datasets, start_index)
  results['SEH'] = same_encounters_home_win_percentage
  results['SEA'] = same_encounters_away_win_percentage
  results['SED'] = same_encounters_draw_percentage

  both_encounters_home_win_percentage, \
  both_encounters_away_win_percentage, both_encounters_draw_percentage = \
    get_both_encounters_percentages(cur_season_dataset,
                                    all_prev_seasons_datasets, start_index)
  results['BEH'] = both_encounters_home_win_percentage
  results['BEA'] = both_encounters_away_win_percentage
  results['BED'] = both_encounters_draw_percentage

  if return_dates == False:
    results = results.drop('Date', axis=1)
  if return_names == False:
    results.drop(['HomeTeam', 'AwayTeam'], axis=1, inplace=True)
  if drop_draws:
    results = results[results['FTR'] != 0].reset_index(drop=True)
  return results, matches_per_round

# Calls create_data_single for each given season except the last one.
# Concatenates all the reurned datasets and split them into features X and
# explained variable y.
#
# seasons_files - list of filenames of seasons to go through. Has to be ordered
#                 from the most recent season. The last season is used only 
#                 as previous season for feature was in lower league
# drop_draws - whether to ignore the matches, which ended with draw
# skip_rounds - how many rounds from the beginning of the season to skip
# return_names - whether to return names of the teams with the features
#
# Returns features X and explained variable y.
def create_data(seasons_files, drop_draws=False, skip_rounds=0, return_names=False):
  dfs = []
  for i in range(len(seasons_files)):
    print('Processing ' + seasons_files[i] + ' season file.')
    results, matches_per_round = create_data_single(seasons_files[i],
                                                    seasons_files[i + 1:],
						    drop_draws=drop_draws,
						    skip_rounds=skip_rounds,
						    return_names=return_names)
    dfs.append(results)
  results_all = pd.concat(dfs)
  X = results_all.drop('FTR', axis=1)
  y = results_all['FTR']
  return X, y