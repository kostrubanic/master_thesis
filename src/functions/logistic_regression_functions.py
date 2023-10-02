# In this file, functions for the baseline using logistic regression are implemented.
# Here belong functions for gathering the original features according to the paper,
# functions for evaluation with the threshold and mappings between names in the FIFA game
# names in the football-data.co.uk dataset.
#
# The baseline is implemented according to the paper Predicting Football Match Results
# with Logistic Regression by D. Prasetio and M. Harlili from 2016. The authors used
# stats from video game FIFA, so I used them as well. I downloaded the data from
# https://www.fifaindex.com/teams/. Features “Home Offense”, “Home Defense”, “Away Offense”,
# and “Away Defense” were used. For the game results I used datasets from
# https://www.football-data.co.uk/. It was not said how the draws were treated in the paper.
# I assume only win/loss matches were taken into account for training, because logistic 
# regression can only solve a binary task. So I have taken into account only win/loss matches
# for training as well.

import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np

# Some teams have different names in the dataset and in the game.
# This is a mapping between them.
team_names_map_epl = {
    'Brentford': 'Brentford',
    'Man United': 'Manchester United',
    'Burnley': 'Burnley',
    'Chelsea': 'Chelsea',
    'Everton': 'Everton',
    'Leicester': 'Leicester City',
    'Watford': 'Watford',
    'Norwich': 'Norwich City',
    'Newcastle': 'Newcastle United',
    'Tottenham': 'Tottenham Hotspur',
    'Liverpool': 'Liverpool',
    'Aston Villa': 'Aston Villa',
    'Crystal Palace': 'Crystal Palace',
    'Leeds': 'Leeds United',
    'Man City': 'Manchester City',
    'Brighton': 'Brighton & Hove Albion',
    'Southampton': 'Southampton',
    'Wolves': 'Wolverhampton Wanderers',
    'Arsenal': 'Arsenal',
    'West Ham': 'West Ham United',
    'Fulham': 'Fulham',
    'West Brom': 'West Bromwich Albion',
    'Sheffield United': 'Sheffield United',
    'Bournemouth': 'Bournemouth',
    'Swansea': 'Swansea City',
    'Hull': 'Hull City',
    'Sunderland': 'Sunderland',
    'Stoke': 'Stoke City',
    'Middlesbrough': 'Middlesbrough',
    'Huddersfield': 'Huddersfield Town',
    'Cardiff': 'Cardiff City',
    'QPR': 'Queens Park Rangers',
    'Blackburn': 'Blackburn Rovers',
    'Wigan': 'Wigan Athletic',
    'Blackpool': 'Blackpool',
    'Birmingham': 'Birmingham City',
    'Bolton': 'Bolton Wanderers'
}

# Some teams have different names in different seasons in the FIFA game.
# If the mapping above does not work, this one is used.
secondary_team_names_map_epl = {
    'Bournemouth': 'AFC Bournemouth'
}

team_names_map_bl = {
    'Dortmund': 'Borussia Dortmund',
    'Werder Bremen': 'SV Werder Bremen',
    'Augsburg': 'FC Augsburg',
    'Fortuna Dusseldorf': 'Fortuna Düsseldorf',
    'Ein Frankfurt': 'Eintracht Frankfurt',
    'Mainz': '1. FSV Mainz 05',
    'Leverkusen': 'Bayer 04 Leverkusen',
    'Freiburg': 'SC Freiburg',
    'Greuther Furth': 'SpVgg Greuther Fürth',
    'Bayern Munich': 'FC Bayern Munich',
    'Hamburg': 'Hamburger SV',
    'Nurnberg': '1. FC Nuremberg',
    'M\'gladbach': 'Borussia Mönchengladbach',
    'Hoffenheim': '1899 Hoffenheim',
    'Stuttgart': 'VfB Stuttgart',
    'Wolfsburg': 'VfL Wolfsburg',
    'Hannover': 'Hannover 96',
    'Schalke 04': 'FC Schalke 04',
    'Braunschweig': 'Eintracht Braunschweig',
    'Hertha': 'Hertha BSC Berlin',
    'Paderborn': 'SC Paderborn 07',
    'FC Koln': '1. FC Köln',
    'Ingolstadt': 'FC Ingolstadt 04',
    'Darmstadt': 'SV Darmstadt 98',
    'RB Leipzig': 'RB Leipzig',
}

# Some teams have different names in different seasons in the FIFA game.
# If the mapping above does not work, this one is used.
secondary_team_names_map_bl = {
    'Hoffenheim': 'TSG 1899 Hoffenheim',
    'Bayern Munich': 'FC Bayern München',
    'Freiburg': 'Sport-Club Freiburg',
    'Hertha': 'Hertha BSC',
    'Nurnberg': '1. FC Nürnberg'
}

# Some teams have different names in the dataset and in the game.
# This is a mapping between them.
team_names_map_llpd = {
    'Granada': 'Granada Club de Fútbol',
    'Betis': 'Real Betis Balompié S.A.D.',
    'Sp Gijon': 'Real Sporting de Gijón S.A.D.',
    'Sociedad': 'Real Sociedad de Fútbol S.A.D.',
    'Valencia': 'Valencia Club de Fútbol S.A.D.',
    'Santander': 'Real Racing Club S.A.D.',
    'Ath Bilbao': 'Athletic Club de Bilbao',
    'Vallecano': 'Rayo Vallecano de Madrid S.A.D.',
    'Ath Madrid': 'Club Atlético de Madrid S.A.D.',
    'Osasuna': 'Club Atlético Osasuna',
    'Getafe': 'Getafe Club de Fútbol S.A.D.',
    'Levante': 'Levante Unión Deportiva S.A.D.',
    'Mallorca': 'Real Club Deportivo Mallorca S.A.D.',
    'Espanol': 'R.C.D. Espanyol de Barcelona S.A.D.',
    'Sevilla': 'Sevilla Fútbol Club S.A.D.',
    'Malaga': 'Málaga Club de Fútbol S.A.D.',
    'Zaragoza': 'Real Zaragoza S.A.D.',
    'Real Madrid': 'Real Madrid Club de Fútbol',
    'Barcelona': 'F.C. Barcelona',
    'Villarreal': 'Villarreal Club de Fútbol S.A.D.',
    'Celta': 'Real Club Celta de Vigo',
    'La Coruna': 'Real Club Deportivo de La Coruña',
    'Valladolid': 'Real Valladolid Club de Fútbol',
    'Elche': 'Elche Club de Fútbol',
    'Eibar': 'SD Eibar',
    'Cordoba': 'Córdoba Club de Fútbol',
    'Almeria': 'Unión Deportiva Almería',
    'Las Palmas': 'UD Las Palmas',
    'Leganes': 'CD Leganés',
    'Alaves': 'Deportivo Alavés',
    'Girona': 'Girona CF',
    'Huesca': 'SD Huesca',
}

# Some teams have different names in different seasons in the FIFA game.
# If the mapping above does not work, this one is used.
secondary_team_names_map_llpd = {
    'Malaga': 'Málaga Club de Fútbol',
    'Mallorca': 'Real Club Deportivo Mallorca',
    'Espanol': 'RCD Espanyol de Barcelona',
    'Sevilla': 'Sevilla Fútbol Club',
    'Getafe': 'Getafe Club de Fútbol',
    'Betis': 'Real Betis Balompié',
    'Barcelona': 'FC Barcelona',
    'Sociedad': 'Real Sociedad de Fútbol',
    'Vallecano': 'Rayo Vallecano de Madrid',
    'Zaragoza': 'Real Zaragoza',
    'Ath Madrid': 'Club Atlético de Madrid',
    'Levante': 'Levante Unión Deportiva',
    'Valencia': 'Valencia Club de Fútbol',
    'Villarreal': 'Villarreal Club de Fútbol',
    'Celta': 'RC Celta de Vigo',
    'Granada': 'Granada CF',
    'Real Madrid': 'Real Madrid CF',
    'Sp Gijon': 'Real Sporting de Gijón',
    'La Coruna': 'RC Deportivo de La Coruña',
    'Osasuna': 'CA Osasuna',
    'Girona': 'Girona FC',
    'Valladolid': 'R. Valladolid CF',
    'Alaves': 'D. Alavés',
    'Ath Bilbao': 'Athletic Club'
}

ternary_team_names_map_llpd = {
    'Ath Madrid': 'Atlético Madrid',
    'Espanol': 'RCD Espanyol',
    'Getafe': 'Getafe CF',
    'Levante': 'Levante UD',
    'Villarreal': 'Villarreal CF',
    'Malaga': 'Málaga CF',
    'Vallecano': 'Rayo Vallecano',
    'Sevilla': 'Sevilla FC',
    'Sociedad': 'Real Sociedad',
    'Valencia': 'Valencia CF',
    'Betis': 'Real Betis',
    'Celta': 'RC Celta',
    'Real Madrid': 'Real Madrid'
}

# Some teams have different names in the dataset and in the game.
# This is a mapping between them.
team_names_map_sa = {
    'Chievo': 'Chievo Verona',
    'Juventus': 'Juventus',
    'Lazio': 'Lazio',
    'Napoli': 'Napoli',
    'Bologna': 'Bologna',
    'Spal': 'SPAL',
    'Empoli': 'Empoli',
    'Cagliari': 'Cagliari',
    'Parma': 'Parma',
    'Udinese': 'Udinese',
    'Sassuolo': 'Sassuolo',
    'Inter': 'Inter',
    'Torino': 'Torino',
    'Roma': 'Roma',
    'Atalanta': 'Atalanta',
    'Frosinone': 'Frosinone',
    'Fiorentina': 'Fiorentina',
    'Sampdoria': 'Sampdoria',
    'Genoa': 'Genoa',
    'Milan': 'Milan',
    'Palermo': 'Palermo',
    'Pescara': 'Pescara',
    'Catania': 'Catania',
    'Siena': 'Siena',
    'Verona': 'Hellas Verona',
    'Livorno': 'Livorno',
    'Cesena': 'Cesena',
    'Carpi': 'Carpi',
    'Crotone': 'Crotone',
    'Benevento': 'Benevento',
}

# Some teams have different names in different seasons in the FIFA game.
# If the mapping above does not work, this one is used.
secondary_team_names_map_sa = {
    'Fiorentina': 'ACF Fiorentina',
    'Spal': 'Spal',
}

# Downloads the stats from FIFA game of given league and season. Process the 
# results of given results file. Creates a dataset of the game results and the
# stats using given mappings.
#
# fifa_league_id - id of the wanted league on the fifaindex website
# season - last 2 gigits of the year in which the wanted season ends
# results_file - filename with game results from football-data website
# team_names_map - mapping from results file to FIFA team names
# secondary_team_names_map, ternary_team_names_map - mappings to use when team_names_map mapping 
#                            does not work
# return_dates - whether to return dates with the stats
#
# Returns dataset with game results and the wanted stats and number of matches
# per round.
def create_data_single(fifa_league_id,
                       season, 
                       results_file,
                       team_names_map,
                       secondary_team_names_map,
		       ternary_team_names_map=None,
                       return_dates=False,
                       drop_draws=True):
  # Downloading stats from fifaindex website using given fifa_league_id
  # and season.
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
  fifa_stats = fifa_data[['Name', 'ATT', 'DEF']].dropna().reset_index(drop=True)
  
  # Processing the game results from results file.
  results_data = pd.read_csv(results_file)
  # The only important thigs are team names and the result.
  results = results_data[['HomeTeam', 'AwayTeam', 'FTR', 'Date']]
  matches_per_round = int(results['HomeTeam'].unique().size / 2)
  # Drop the matches, which ended with draw.
  if drop_draws:
    results = results[results['FTR'] != 'D'].reset_index(drop=True)
  # Home win is repredsented as 1, away win as 0, draw as 2.
  results['FTR'].replace('H', 1, inplace=True)
  results['FTR'].replace('A', 0, inplace=True)
  results['FTR'].replace('D', 2, inplace=True)

  # Adding the features to the results.
  for index in range(results.shape[0]):
    # Mapping the team name from the results file name to the FIFA name.
    home_team = team_names_map[results.loc[index, 'HomeTeam']]
    if fifa_stats[fifa_stats['Name'] == home_team].shape[0] == 0:
      # If the mapped team name is not in the FIFA stats, use secondary mapping
      home_team = secondary_team_names_map[results.loc[index, 'HomeTeam']]
      if fifa_stats[fifa_stats['Name'] == home_team].shape[0] == 0:
        home_team = ternary_team_names_map[results.loc[index, 'HomeTeam']]
    away_team = team_names_map[results.loc[index, 'AwayTeam']]
    if fifa_stats[fifa_stats['Name'] == away_team].shape[0] == 0:
      away_team = secondary_team_names_map[results.loc[index, 'AwayTeam']]
      if fifa_stats[fifa_stats['Name'] == away_team].shape[0] == 0:
        away_team = ternary_team_names_map[results.loc[index, 'AwayTeam']]
    # Creating the Home Offense, Home Defense, 
    # Away Offense, and Away Defense features.
    hatt = fifa_stats[fifa_stats['Name'] == home_team].reset_index().loc[0, 'ATT']
    aatt = fifa_stats[fifa_stats['Name'] == away_team].reset_index().loc[0, 'ATT']
    hdef = fifa_stats[fifa_stats['Name'] == home_team].reset_index().loc[0, 'DEF']
    adef = fifa_stats[fifa_stats['Name'] == away_team].reset_index().loc[0, 'DEF']
    results.loc[index, 'HATT'] = hatt
    results.loc[index, 'AATT'] = aatt
    results.loc[index, 'HDEF'] = hdef
    results.loc[index, 'ADEF'] = adef
  # Leaving only the wanted columns.
  if return_dates == True:
    results = results[['FTR', 'HATT', 'AATT', 'HDEF', 'ADEF', 'Date']]
  else:
    results = results[['FTR', 'HATT', 'AATT', 'HDEF', 'ADEF']]
  return results, matches_per_round

# Calls create_data_single with each given season and results file.
# Concatenates all the reurned datasets and split them into features X and
# explained variable y.
#
# fifa_league_id - id of the wanted league on the fifaindex website
# seasons - list of years in which the wanted seasons end (last 2 digits)
# results_files - list of filenames with game results from football-data website
#                 1 for each season
# team_names_map - mapping from results file to FIFA team names
# secondary_team_names_map - mapping to use when team_names_map mapping 
#                            does not work
#
# Returns features X and explained variable y.
def create_data(fifa_league_id,
                seasons, 
                results_files, 
                team_names_map, 
                secondary_team_names_map,
		ternary_team_names_map=None,):
  dfs = []
  for season, results_file in zip(seasons, results_files):
    results, matches_per_round = create_data_single(fifa_league_id,
                                 season, 
                                 results_file,
                                 team_names_map,
                                 secondary_team_names_map,
				 ternary_team_names_map=ternary_team_names_map)
    dfs.append(results)
  results_all = pd.concat(dfs)
  X = results_all.drop('FTR', axis=1)
  y = results_all['FTR']
  return X, y

# Evaluates the given predictions with the given truth values.
#
# preds - given predictions - numbers between 0 and 1
# truths - truth values numbers from set {0, 1, 2}
# threshold - defines how often will draws be predicted
#
# Returns features X and explained variable y.
def evaluate(preds, truths, threshold):
  preds[preds < 0.5 - threshold] = 0 # predicts away win
  preds[preds > 0.5 + threshold] = 1 # predicts home win
  preds[(preds > 0) & (preds < 1)] = 2 # predicts draw
  return 1 - (np.count_nonzero(preds - truths) / len(preds))