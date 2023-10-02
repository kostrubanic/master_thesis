# In this file, functions for the baseline using PCA and Naive Bayes are implemented.
# Here belong functions for gathering the original features according to the paper
# and lists of teams for gathering the ids.
#
# The baseline is implemented according to the paper Predicting The Dutch Football Competition
# Using Public Data: A Machine Learning Approach by N. Tax and Y. Joustra from 2015.
# The authors used many different features achieved from the previous results and from
# the games statistics. I kept the features as similar as possible, but a few ones I had to
# replace. I downloaded the data from https://www.football-data.co.uk/.

import pandas as pd
from datetime import datetime

# List of EPL teams used for getting team id.
teams_epl = ['Reading', 'Fulham', 'Bournemouth', 'Sunderland', 'Swansea', 'Stoke',
         'Wolves', 'West Brom', 'Hull', 'Portsmouth', 'Newcastle', 'Blackburn',
         'Birmingham', 'Leicester', 'West Ham', 'Brentford', 'Brighton',
         'Chelsea', 'Liverpool', 'Man City', 'Charlton', 'Norwich', 'Arsenal',
         'Man United', 'Leeds', 'Blackpool', 'Cardiff', 'Everton', 'Huddersfield',
         'Tottenham', 'Derby', 'Burnley', 'Sheffield United', 'Wigan',
         'Southampton', 'Middlesbrough', 'QPR', 'Aston Villa', 'Crystal Palace',
         'Watford', 'Bolton']

# List of BL teams used for getting team id.
teams_bl = ['Hertha', 'Greuther Furth', 'Stuttgart', 'Hamburg', 'Braunschweig',
         'Duisburg', 'Ingolstadt', 'Freiburg', 'Bochum', 'RB Leipzig', 'Cottbus',
         'Leverkusen', 'Fortuna Dusseldorf', 'Augsburg', 'Aachen', 'Ein Frankfurt',
         'Hannover', 'Hoffenheim', 'Schalke 04', "M'gladbach", 'Werder Bremen',
         'Paderborn', 'Karlsruhe', 'Bielefeld', 'FC Koln', 'Wolfsburg', 'St Pauli',
         'Kaiserslautern', 'Bayern Munich', 'Dortmund', 'Darmstadt', 'Mainz',
         'Nurnberg', 'Hansa Rostock']

# List of LLPD teams used for getting team id.
teams_llpd = ['Santander', 'Numancia', 'Mallorca', 'Recreativo', 'Valladolid',
         'Vallecano', 'Ath Madrid', 'Alaves', 'Zaragoza', 'Osasuna', 'Tenerife',
         'Eibar', 'Celta', 'Gimnastic', 'Barcelona', 'Espanol', 'Granada',
         'Xerez', 'Leganes', 'Real Madrid', 'Sp Gijon', 'Ath Bilbao', 'Las Palmas',
         'Hercules', 'Levante', 'Getafe', 'Betis', 'La Coruna', 'Huesca', 'Malaga',
         'Almeria', 'Murcia', 'Villarreal', 'Cordoba', 'Elche', 'Sociedad',
         'Girona', 'Sevilla', 'Valencia']

# List of SA teams used for getting team id.
teams_sa = ['Torino', 'Sassuolo', 'Crotone', 'Sampdoria', 'Messina', 'Milan',
         'Atalanta', 'Pescara', 'Inter', 'Genoa', 'Palermo', 'Bologna', 'Parma',
         'Bari', 'Roma', 'Chievo', 'Juventus', 'Carpi', 'Frosinone', 'Siena',
         'Empoli', 'Novara', 'Fiorentina', 'Ascoli', 'Napoli', 'Lazio', 'Udinese',
         'Lecce', 'Verona', 'Livorno', 'Cesena', 'Catania', 'Spal', 'Benevento',
         'Cagliari', 'Brescia', 'Reggina']

# List of BSA teams used for getting team id.
teams_bsa = ['Santos', 'Nautico', 'Joinville', 'Avai', 'America MG', 'Ponte Preta',
         'Flamengo RJ', 'Internacional', 'Atletico GO', 'Vitoria', 'Gremio',
         'Fortaleza', 'Parana', 'Corinthians', 'Vasco', 'Atletico-MG',
         'Botafogo RJ', 'Sport Recife', 'Goias', 'Atletico-PR', 'Chapecoense-SC',
         'Figueirense', 'Portuguesa', 'Fluminense', 'Santa Cruz', 'Sao Paulo',
         'Criciuma', 'Novara', 'Ceara', 'Cruzeiro', 'Bahia', 'Coritiba',
         'Palmeiras', 'CSA']

# Gathers team ids from given dataset.
#
# cur_season_dataset - dataset to gather ids of teams from
# start_index - index of the first match to gather ids of
# teams - list of teams to ge ids according to
#
# Returns list of home teams ids and list of away teams ids
def get_ids(cur_season_dataset, start_index, teams):
  home_ids = []
  away_ids = []
  for index in range(start_index, cur_season_dataset.shape[0]):
    home_team = cur_season_dataset.loc[index, 'HomeTeam']
    away_team = cur_season_dataset.loc[index, 'AwayTeam']
    home_id = teams.index(home_team)
    away_id = teams.index(away_team)
    home_ids.append(home_id)
    away_ids.append(away_id)
  return home_ids, away_ids

# Gathers average goals of teams from given dataset.
#
# cur_season_dataset - dataset to gather average goals from
# start_index - index of the first match to gather average goals for
#
# Returns list of home teams average scored goals, list of away teams average
# scored goals, list of home teams average conceded goals and list of away teams
# average conceded goals
def get_avg_goals(cur_season_dataset, start_index):
  home_scored_goalss = []
  away_scored_goalss = []
  home_conceded_goalss = []
  away_conceded_goalss = []
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
    home_conceded_goals_sum = home_team_home_matches['FTAG'].sum()
    home_conceded_goals_sum += home_team_away_matches['FTHG'].sum()
    away_conceded_goals_sum = away_team_home_matches['FTAG'].sum()
    away_conceded_goals_sum += away_team_away_matches['FTHG'].sum()
    home_matches_num = home_team_home_matches.shape[0] + \
                       home_team_away_matches.shape[0]
    away_matches_num = away_team_home_matches.shape[0] + \
                       away_team_away_matches.shape[0]
    # Calculating and appending the averages.
    home_scored_goalss.append(home_scored_goals_sum / home_matches_num)
    away_scored_goalss.append(away_scored_goals_sum / away_matches_num)
    home_conceded_goalss.append(home_conceded_goals_sum / home_matches_num)
    away_conceded_goalss.append(away_conceded_goals_sum / away_matches_num)

  return home_scored_goalss, away_scored_goalss, \
         home_conceded_goalss, away_conceded_goalss

# Gets match results ago matches ago of given teams
#
# home_team_matches - dataset with all matches of home team
# away_team_matches - dataset with all matches of away team
# home_team - name of home team
# away_team - name of away team
# ago - number of matches ago of wanted results
#
# Returns home team result and away team result of match ago matches ago
def get_match_results(home_team_matches, away_team_matches,
                      home_team, away_team, ago):
  # Reversing order of matches to make them accessible through index.
  home_team_matches_ordered = home_team_matches.iloc[::-1].reset_index()
  away_team_matches_ordered = away_team_matches.iloc[::-1].reset_index()
  home_result_ago = home_team_matches_ordered.loc[ago - 1, 'FTR']
  away_result_ago = away_team_matches_ordered.loc[ago - 1, 'FTR']
  
  # The result for the wanted team depends also on whether the wanted team was
  # home or away team in the wanted match.
  # Win is represented as 1, draw as 0 and loss as -1.
  if home_result_ago == 'D':
      home_result = 0
  elif home_result_ago == 'H':
    if home_team_matches_ordered.loc[ago - 1, 'HomeTeam'] == home_team:
      home_result = 1
    else:
      home_result = -1
  else:
    if home_team_matches_ordered.loc[ago - 1, 'HomeTeam'] == home_team:
      home_result = -1
    else:
      home_result = 1
  if away_result_ago == 'D':
    away_result = 0
  elif away_result_ago == 'H':
    if away_team_matches_ordered.loc[ago - 1, 'HomeTeam'] == away_team:
      away_result = 1
    else:
      away_result = -1
  else:
    if away_team_matches_ordered.loc[ago - 1, 'HomeTeam'] == away_team:
      away_result = -1
    else:
      away_result = 1

  return home_result, away_result

# Gathers results of previous matches of teams from given dataset.
#
# cur_season_dataset - dataset to gather results from
# start_index - index of the first match to gather results for
#
# Returns list of home teams results of matches 1 to 5 matches ago and away
# teams results of matches 1 to 5 matches ago.
def get_prev_results(cur_season_dataset, start_index):
  home_results_1_ago = []
  away_results_1_ago = []
  home_results_2_ago = []
  away_results_2_ago = []
  home_results_3_ago = []
  away_results_3_ago = []
  home_results_4_ago = []
  away_results_4_ago = []
  home_results_5_ago = []
  away_results_5_ago = []
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

    # Gathering wanted results of the given teams
    home_results = []
    away_results = []
    for ago in range(1, 6):
      home_result, away_result = get_match_results(home_team_matches,
                                                   away_team_matches,
                                                   home_team, away_team, ago)
      home_results.append(home_result)
      away_results.append(away_result)
    home_results_1_ago.append(home_results[0])
    away_results_1_ago.append(away_results[0])
    home_results_2_ago.append(home_results[1])
    away_results_2_ago.append(away_results[1])
    home_results_3_ago.append(home_results[2])
    away_results_3_ago.append(away_results[2])
    home_results_4_ago.append(home_results[3])
    away_results_4_ago.append(away_results[3])
    home_results_5_ago.append(home_results[4])
    away_results_5_ago.append(away_results[4])

  return home_results_1_ago, away_results_1_ago, home_results_2_ago, \
         away_results_2_ago, home_results_3_ago, away_results_3_ago, \
         home_results_4_ago, away_results_4_ago, home_results_5_ago, \
         away_results_5_ago

# Gathers if teams were in lower league last season.
#
# cur_season_dataset - dataset of matches to gather info for
# previous_season_dataset - dataset of previous season, if a team was not here,
#                           it was in lower league last yeas 
# start_index - index of the first match to gather info for
#
# Returns list of bools if home team was in lower league and list of bools
# if away team was in lower league
def get_was_lower_league(cur_season_dataset,
                         previous_season_dataset, start_index):
  home_was_lower_league = []
  away_was_lower_league = []
  for index in range(start_index, cur_season_dataset.shape[0]):
    home_team = cur_season_dataset.loc[index, 'HomeTeam']
    away_team = cur_season_dataset.loc[index, 'AwayTeam']
    if home_team in previous_season_dataset['HomeTeam'].unique():
      # 0 - team was in lower league, 1 - it was not
      home_was_lower_league.append(0)
    else:
      home_was_lower_league.append(1)
    if away_team in previous_season_dataset['HomeTeam'].unique():
      away_was_lower_league.append(0)
    else:
      away_was_lower_league.append(1)
  return home_was_lower_league, away_was_lower_league

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
    home_win_percentage.append(home_wins / home_matches_num)
    away_win_percentage.append(away_wins / away_matches_num)
    home_lose_percentage.append(home_losses / home_matches_num)
    away_lose_percentage.append(away_losses / away_matches_num)
    home_draw_percentage.append(home_draws / home_matches_num)
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
    home_ground_win_percentage.append(home_ground_wins / home_matches_num)
    home_ground_lose_percentage.append(home_ground_losses / home_matches_num)
    home_ground_draw_percentage.append(home_ground_draws / home_matches_num)
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
  all_prev_seasons_dataset = pd.concat(prev_seasons_datatsets)
  for index in range(start_index, cur_season_dataset.shape[0]):
    home_team = cur_season_dataset.loc[index, 'HomeTeam']
    away_team = cur_season_dataset.loc[index, 'AwayTeam']
    # Taking only already finished matches.
    cur_season_dataset_finished = cur_season_dataset[:index]
    # Getting dataset of all the previous seasons and the finished part of the
    # current season.
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
  all_prev_seasons_dataset = pd.concat(prev_seasons_datatsets)
  for index in range(start_index, cur_season_dataset.shape[0]):
    home_team = cur_season_dataset.loc[index, 'HomeTeam']
    away_team = cur_season_dataset.loc[index, 'AwayTeam']
    # Taking only already finished matches.
    cur_season_dataset_finished = cur_season_dataset[:index]
    # Getting dataset of all the previous seasons and the finished part of the
    # current season.
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

# Gathers average shots of teams from given dataset.
#
# cur_season_dataset - dataset to gather average shots from
# start_index - index of the first match to gather average shots for
#
# Returns list of home teams average fired shots, list of away teams average
# fired shots, list of home teams average conceded shots and list of away teams
# average conceded shots
def get_avg_shots(cur_season_dataset, start_index):
  home_fired_shots = []
  away_fired_shots = []
  home_conceded_shots = []
  away_conceded_shots = []
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
    home_conceded_shots_sum = home_team_home_matches['AS'].sum()
    home_conceded_shots_sum += home_team_away_matches['HS'].sum()
    away_conceded_shots_sum = away_team_home_matches['AS'].sum()
    away_conceded_shots_sum += away_team_away_matches['HS'].sum()
    home_matches_num = home_team_home_matches.shape[0] + \
                       home_team_away_matches.shape[0]
    away_matches_num = away_team_home_matches.shape[0] + \
                       away_team_away_matches.shape[0]
    # Calculating and appends the averages.
    home_fired_shots.append(home_fired_shots_sum / home_matches_num)
    away_fired_shots.append(away_fired_shots_sum / away_matches_num)
    home_conceded_shots.append(home_conceded_shots_sum / home_matches_num)
    away_conceded_shots.append(away_conceded_shots_sum / away_matches_num)

  return home_fired_shots, away_fired_shots, \
         home_conceded_shots, away_conceded_shots

# Gathers average fauls of teams from given dataset.
#
# cur_season_dataset - dataset to gather average fauls from
# start_index - index of the first match to gather average fauls for
#
# Returns list of home teams average commited fauls, list of away teams average
# commited shots, list of home teams average achieved fauls and list of away
# teams average achieved fauls
def get_avg_fauls(cur_season_dataset, start_index):
  home_commited_fauls = []
  away_commited_fauls = []
  home_achieved_fauls = []
  away_achieved_fauls = []
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
    home_achieved_fauls_sum = home_team_home_matches['AF'].sum()
    home_achieved_fauls_sum += home_team_away_matches['HF'].sum()
    away_achieved_fauls_sum = away_team_home_matches['AF'].sum()
    away_achieved_fauls_sum += away_team_away_matches['HF'].sum()
    home_matches_num = home_team_home_matches.shape[0] + \
                       home_team_away_matches.shape[0]
    away_matches_num = away_team_home_matches.shape[0] + \
                       away_team_away_matches.shape[0]
    # Calculating and appends the averages.
    home_commited_fauls.append(home_commited_fauls_sum / home_matches_num)
    away_commited_fauls.append(away_commited_fauls_sum / away_matches_num)
    home_achieved_fauls.append(home_achieved_fauls_sum / home_matches_num)
    away_achieved_fauls.append(away_achieved_fauls_sum / away_matches_num)

  return home_commited_fauls, away_commited_fauls, \
         home_achieved_fauls, away_achieved_fauls

# Gathers average goals from last 5 matches of teams from given dataset.
#
# cur_season_dataset - dataset to gather average recent goals from
# start_index - index of the first match to gather average recent goals for
#
# Returns list of home teams average recent scored goals, list of away teams
# average recent scored goals, list of home teams average recent conceded goals
# and list of away teams average recent conceded goals
def get_avg_recent_goals(cur_season_dataset, start_index):
  home_recent_scored_goals = []
  away_recent_scored_goals = []
  home_recent_conceded_goals = []
  away_recent_conceded_goals = []
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
    # Reversing order of matches to make them accessible through index.
    home_team_recent_matches = home_team_matches.iloc[::-1].reset_index()[:5]
    away_team_recent_matches = away_team_matches.iloc[::-1].reset_index()[:5]

    home_team_recent_home_matches = home_team_recent_matches[
        home_team_recent_matches['HomeTeam'] == home_team]
    home_team_recent_away_matches = home_team_recent_matches[
        home_team_recent_matches['AwayTeam'] == home_team]
    away_team_recent_home_matches = away_team_recent_matches[
        away_team_recent_matches['HomeTeam'] == away_team]
    away_team_recent_away_matches = away_team_recent_matches[
        away_team_recent_matches['AwayTeam'] == away_team]

    home_recent_scored_goals_sum = home_team_recent_home_matches['FTHG'].sum()
    home_recent_scored_goals_sum += home_team_recent_away_matches['FTAG'].sum()
    away_recent_scored_goals_sum = away_team_recent_home_matches['FTHG'].sum()
    away_recent_scored_goals_sum += away_team_recent_away_matches['FTAG'].sum()
    home_recent_conceded_goals_sum = home_team_recent_home_matches['FTAG'].sum()
    home_recent_conceded_goals_sum += home_team_recent_away_matches['FTHG'].sum()
    away_recent_conceded_goals_sum = away_team_recent_home_matches['FTAG'].sum()
    away_recent_conceded_goals_sum += away_team_recent_away_matches['FTHG'].sum()

    # Calculating and appending the averages.
    home_recent_scored_goals.append(home_recent_scored_goals_sum / 5)
    away_recent_scored_goals.append(away_recent_scored_goals_sum / 5)
    home_recent_conceded_goals.append(home_recent_conceded_goals_sum / 5)
    away_recent_conceded_goals.append(away_recent_conceded_goals_sum / 5)

  return home_recent_scored_goals, away_recent_scored_goals, \
         home_recent_conceded_goals, away_recent_conceded_goals

# Creates datasets from given filenames and process them. Gathers the wanted
# features for the current season file.
#
# cur_season_file - filename of the wanted season from football-data website
# prev_season_file - filename of the previous season from football-data website,
#                    used for feature was in lower league
# other_prev_seasons_files - filenames of the other previous seasons from
#                            football-data website, used for percentages of 
#                            previous encounters
# teams - list of teams, used for ids
# return_dates - whether to return dates with the features
# return_names - whether to return names of the teams with the features
# include_shots_fauls - whether to include shots and fauls, which are inaccessible for the BSA
#
# Returns dataset with game results and the wanted features and number
# of matches per round.
def create_data_single(cur_season_file, prev_season_file,
                      other_prev_seasons_files, teams, return_dates=False,
                      return_names=False, include_shots_fauls=True):
  cur_season_dataset = pd.read_csv(cur_season_file)
  prev_season_dataset = pd.read_csv(prev_season_file)
  all_prev_seasons_datasets = [prev_season_dataset]
  for file in other_prev_seasons_files:
    dataset = pd.read_csv(file)
    all_prev_seasons_datasets.append(dataset)
  
  results = cur_season_dataset[['FTR', 'HomeTeam', 'AwayTeam', 'Date']]
  matches_per_round = int(results['HomeTeam'].unique().size / 2)
  # Starting a few rounds later to have previous matches results
  start_index = 6 * matches_per_round
  results = results[start_index:]
  # Home win is repredsented as 1, away win as -1, draw as 0.
  results['FTR'].replace('H', 1, inplace=True)
  results['FTR'].replace('A', -1, inplace=True)
  results['FTR'].replace('D', 0, inplace=True)
  

  home_ids, away_ids = get_ids(cur_season_dataset, start_index, teams)
  results['HID'] = home_ids
  results['AID'] = away_ids

  home_scored_goalss, away_scored_goalss, \
  home_conceded_goalss, away_conceded_goalss = get_avg_goals(cur_season_dataset,
                                                             start_index)
  results['HSG'] = home_scored_goalss
  results['ASG'] = away_scored_goalss
  results['HCG'] = home_conceded_goalss
  results['ACG'] = away_conceded_goalss

  home_results_1_ago, away_results_1_ago, home_results_2_ago, \
  away_results_2_ago, home_results_3_ago, away_results_3_ago, \
  home_results_4_ago, away_results_4_ago, home_results_5_ago, \
  away_results_5_ago = get_prev_results(cur_season_dataset, start_index)
  results['HR1A'] = home_results_1_ago
  results['AR1A'] = away_results_1_ago
  results['HR2A'] = home_results_2_ago
  results['AR2A'] = away_results_2_ago
  results['HR3A'] = home_results_3_ago
  results['AR3A'] = away_results_3_ago
  results['HR4A'] = home_results_4_ago
  results['AR4A'] = away_results_4_ago
  results['HR5A'] = home_results_5_ago
  results['AR5A'] = away_results_5_ago

  home_was_lower_league, away_was_lower_league = \
    get_was_lower_league(cur_season_dataset, prev_season_dataset, start_index)
  results['HWLL'] = home_was_lower_league
  results['AWLL'] = away_was_lower_league

  home_days_rest, away_days_rest = get_days_rest(cur_season_dataset, start_index)
  results['HDFLM'] = home_days_rest
  results['ADFLM'] = away_days_rest

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

  if include_shots_fauls:
    home_fired_shots, away_fired_shots, \
    home_conceded_shots, away_conceded_shots = get_avg_shots(cur_season_dataset,
                                                           start_index)
  
    results['HFS'] = home_fired_shots
    results['AFS'] = away_fired_shots
    results['HCS'] = home_conceded_shots
    results['ACS'] = away_conceded_shots

    home_commited_fauls, away_commited_fauls, \
    home_achieved_fauls, away_achieved_fauls = get_avg_fauls(cur_season_dataset,
                                                           start_index)
    results['HCF'] = home_commited_fauls
    results['ACF'] = away_commited_fauls
    results['HAF'] = home_achieved_fauls
    results['AAF'] = away_achieved_fauls

  home_recent_scored_goals, away_recent_scored_goals, \
  home_recent_conceded_goals, \
  away_recent_conceded_goals = get_avg_recent_goals(cur_season_dataset,
                                                    start_index)
  results['HRSG'] = home_recent_scored_goals
  results['ARSG'] = away_recent_scored_goals
  results['HRCG'] = home_recent_conceded_goals
  results['ARCG'] = away_recent_conceded_goals

  if return_dates == False:
    results = results.drop('Date', axis=1)
  if return_names == False:
    results.drop(['HomeTeam', 'AwayTeam'], axis=1, inplace=True)
  return results, matches_per_round

# Calls create_data_single for each given season except the last one.
# Concatenates all the reurned datasets and split them into features X and
# explained variable y.
#
# seasons_files - list of filenames of seasons to go through. Has to be ordered
#                 from the most recent season. The last season is used only 
#                 as previous season for feature was in lower league
# teams - list of teams, used for ids
# return_names - whether to return names of the teams with the features
#
# Returns features X and explained variable y.
def create_data(seasons_files, teams, return_names=False, include_shots_fauls=True):
  dfs = []
  for i in range(len(seasons_files) - 1):
    print('Processing ' + seasons_files[i] + ' season file.')
    results, matches_per_round = create_data_single(seasons_files[i],
                                                    seasons_files[i + 1],
                                                    seasons_files[i + 2:],
                                                    teams,
                                                    return_names=return_names,
						    include_shots_fauls=include_shots_fauls)
    dfs.append(results)
  results_all = pd.concat(dfs)
  X = results_all.drop('FTR', axis=1)
  y = results_all['FTR']
  return X, y  