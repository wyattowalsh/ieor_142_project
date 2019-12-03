import numpy as np
import pandas as pd
import warnings
from pathlib import Path

def create_datasets():
	"""Creates a dictionary of different types of datasets

	"""
	raw_games = pd.read_csv(Path().resolve().joinpath('data', 'raw', 'game_data.csv'), index_col = 0)
	raw_games['Time'] = pd.to_datetime(raw_games['Time'])
	raw_stadiums = pd.read_csv(Path().resolve().joinpath('data', 'raw', 'stadiums_data.csv'), index_col = 0)

	dataset_1 = create_dataset_1(raw_games,raw_stadiums)
	dataset_1_1 = create_dataset_1_1(dataset_1)
	dataset_2 = create_dataset_2(raw_games)
	dataset_3 = create_dataset_3(raw_games, raw_stadiums)

	datasets = {'dataset_1': dataset_1,
				'dataset_1_1': dataset_1_1,
				'dataset_2': dataset_2,
				'dataset_3': dataset_3}

	return datasets

def load_datasets():
	'''Creates a dictionary of different types of datasets by loading local files

	'''

	try:
		dataset_1 = pd.read_csv(Path().resolve().joinpath('data', 'processed', 'dataset_1.csv'), index_col = 0)
		dataset_1.index = pd.to_datetime(dataset_1.index)
		dataset_1_1 = pd.read_csv(Path().resolve().joinpath('data', 'processed', 'dataset_1_1.csv'), index_col = 0)
		dataset_1_1.index = pd.to_datetime(dataset_1_1.index)
		dataset_2 = pd.read_csv(Path().resolve().joinpath('data', 'processed', 'dataset_2.csv'), index_col = 0)
		dataset_2.index = pd.to_datetime(dataset_2.index)
		dataset_3 = pd.read_csv(Path().resolve().joinpath('data', 'processed', 'dataset_3.csv'), index_col = 0)
		dataset_3.index = pd.to_datetime(dataset_3.index)

		ddatasets = {'dataset_1': dataset_1,
					'dataset_1_1': dataset_1_1,
					'dataset_2': dataset_2,
					'dataset_3': dataset_3}

		return datasets
	except: 
		return create_datasets()

def load_dataset(name):
	try: 
		dataset = pd.read_csv(Path().resolve().parent.parent.joinpath('data', 'processed', '{}.csv'.format(name)), 
		                      index_col = 0)
		dataset.index = pd.to_datetime(dataset.index)
		return dataset
	except:
		warnings.warn('Most likely are overwriting data.')
		datasets = create_datasets()
		names = list(datasets.keys())
		for i in np.arange(len(names)):
			save_dataset(names[i], datasets[names[i]])
		return datasets[name]


def save_dataset(name, data):
	"""

	"""

	data = data.copy()
	to_save = Path().resolve().joinpath('data', 'processed', '{}.csv'.format(name))
	data.to_csv(to_save)


def create_dataset_1(raw_games, raw_stadiums):
	"""Creates a filtered dataset, adds a few new features, removes extreme outliers, and removes redundant features.
	
	Filters out games that were held in old stadiums and years before 2004 since that is the earliest we have popularity data.
	Adds lag feature of attendance, the last attendance versus the same opponent, and the capacities of each stadium.
	Games with attendance over 25,000 people are removed.
	It seems that since there is popularity and capacity data the team names can be removed.
	It also seems that since there is win % data the points scored by each team can be removed as well. 
	"""
	data = raw_games.copy()
	stadiums = raw_stadiums.copy()

	# Remove games that were played in old stadiums
	teams = data['Home'].unique()
	for team in teams:
		earliest = pd.to_numeric(stadiums.loc[stadiums['Home'] == team]['Opened'].values[0])
		data.loc[data['Home'] == team] = data.loc[data['Home'] == team].loc[data['Time'] >= pd.Timestamp(earliest, 7, 1)]
	data = data.dropna()

	# Add features
	data['Last Game'] = np.nan
	data["Last Attendance vs Opp"] = np.nan
	data["Capacity"] = np.nan
	for team in teams:
		data.loc[data['Home'] == team, 'Last Game'] = (data.loc[data['Home'] == team, 'Attendance'].shift(1)) 
		data.loc[data['Home'] == team] = data.loc[data['Home'] == team].sort_values(['Visitor', 'Time'])
		tot_scores = np.array([])
		for vis in data.loc[data['Home'] == team]['Visitor'].unique():
			scores = data.loc[data['Home'] == team].loc[data.loc[data['Home'] == team]['Visitor'] == vis]['Attendance'].values
			scores = np.append(np.array([0]), scores[0:len(scores)-1])
			tot_scores = np.append(tot_scores, scores)
		data.loc[data['Home'] == team, 'Last Attendance vs Opp'] = tot_scores
		data.loc[data['Home'] == team].sort_values('Time')
		data.loc[data['Home'] == team] = data.loc[data['Last Attendance vs Opp'] != 0]
		data.loc[data['Home'] == team, "Capacity"] = stadiums.loc[stadiums['Home'] == team, "Capacity"].values[0]
	data = data.dropna()

	# Remove extreme outliers
	data = data.loc[data['Attendance'] <= 25000]

	# Remove seemingly redundant features and games before 2004
	data = data.drop(['Home', "V PTS", "H PTS", "Visitor", "Match-up"], axis=1)
	data = data.sort_values('Time').reset_index(drop=True).set_index('Time', drop=True)
	data = data.loc[data.index.year >= 2004]
	data = data.dropna()
	data = data[['V Pop', 'H Pop', 'Curr Win %',  'LS Win %','Last Game', 'Last Attendance vs Opp', 
	'Capacity','Playoffs?', 'Last Five', 'Day of Week','Month', 'Rivalry?', 'Attendance']]
	return data

def create_dataset_1_1(dataset_1):
	dataset_1_1 = dataset_1.copy()
	subset_columns = np.append(['H Pop', "LS Win %","Last Game", "Last Attendance vs Opp"], dataset_1_1.columns.values[7:])
	dataset_1_1 = dataset_1_1[subset_columns]
	return dataset_1_1

def create_dataset_2(raw_games):
	"""This dataset contains games from all the years scraped.

	It does not include popularity or capacity data.
	It does include the lagged attendance feature and last attendance versus the same opponent
	It also removes games with attendance over 25,000
	"""
	data = raw_games.copy()

	data['Last Game'] = np.nan
	data["Last Attendance vs Opp"] = np.nan
	teams = data['Home'].unique()
	for team in teams:
		data.loc[data['Home'] == team, 'Last Game'] = (data.loc[data['Home'] == team, 'Attendance'].shift(1)) 
		data.loc[data['Home'] == team] = data.loc[data['Home'] == team].sort_values(['Visitor', 'Time'])
		tot_scores = np.array([])
		for vis in data.loc[data['Home'] == team]['Visitor'].unique():
			scores = data.loc[data['Home'] == team].loc[data.loc[data['Home'] == team]['Visitor'] == vis]['Attendance'].values
			scores = np.append(np.array([0]), scores[0:len(scores)-1])
			tot_scores = np.append(tot_scores, scores)
		data.loc[data['Home'] == team, 'Last Attendance vs Opp'] = tot_scores
		data.loc[data['Home'] == team].sort_values('Time')
		data.loc[data['Home'] == team] = data.loc[data['Last Attendance vs Opp'] != 0]
	data = data.dropna()

	# Remove extreme outliers
	data = data.loc[data['Attendance'] <= 25000]

	# Remove seemingly redundant or irrelevant features
	data = data.drop(['Home', "V PTS", "H PTS", "H Pop", "V Pop", "Visitor", "Match-up"], axis=1)
	data = data.sort_values('Time').reset_index(drop=True).set_index('Time', drop=True)
	data = data.dropna()
	return data

def create_dataset_3(raw_games, raw_stadiums):
	"""Contains games since 1990.

	It does not include popularity data. 
	It does include the lagged attendance feature, last attendance versus the same opponent, and stadium capacities
	It also removes games with attendance over 25,000
	"""

	data = raw_games.copy()
	stadiums = raw_stadiums.copy()

	# Remove games that were played in old stadiums
	teams = data['Home'].unique()
	for team in teams:
		earliest = pd.to_numeric(stadiums.loc[stadiums['Home'] == team]['Opened'].values[0])
		data.loc[data['Home'] == team] = data.loc[data['Home'] == team].loc[data['Time'] >= pd.Timestamp(earliest, 7, 1)]
	data = data.dropna()

	# Add features
	data['Last Game'] = np.nan
	data["Last Attendance vs Opp"] = np.nan
	data["Capacity"] = np.nan
	for team in teams:
		data.loc[data['Home'] == team, 'Last Game'] = (data.loc[data['Home'] == team, 'Attendance'].shift(1)) 
		data.loc[data['Home'] == team] = data.loc[data['Home'] == team].sort_values(['Visitor', 'Time'])
		tot_scores = np.array([])
		for vis in data.loc[data['Home'] == team]['Visitor'].unique():
			scores = data.loc[data['Home'] == team].loc[data.loc[data['Home'] == team]['Visitor'] == vis]['Attendance'].values
			scores = np.append(np.array([0]), scores[0:len(scores)-1])
			tot_scores = np.append(tot_scores, scores)
		data.loc[data['Home'] == team, 'Last Attendance vs Opp'] = tot_scores
		data.loc[data['Home'] == team].sort_values('Time')
		data.loc[data['Home'] == team] = data.loc[data['Last Attendance vs Opp'] != 0]
		data.loc[data['Home'] == team, "Capacity"] = stadiums.loc[stadiums['Home'] == team, "Capacity"].values[0]
	data = data.dropna()

	# Remove extreme outliers
	data = data.loc[data['Attendance'] <= 25000]

	# Remove seemingly redundant or irrelevant features
	data = data.drop(['Home', "V PTS", "H PTS", "H Pop", "V Pop", "Visitor", "Match-up"], axis=1)
	data = data.sort_values('Time').reset_index(drop=True).set_index('Time', drop=True)
	data = data.dropna()
	return data

def get_dataset_name(dataset):
	'''Returns the name associated with the dataset in a dictionary of datasets.

	With the assumption that each dataset is unique.
	'''
	datasets = load_datasets()
	return list(datasets.keys())[list(datasets.values()).index(dataset)]

def get_number_numerical(name):
	if name == 'dataset_1':
		numerical_features = 7
	elif name == 'dataset_1_1':
		numerical_features = 4
	# elif name == 'dataset_2':
	# 	numerical_features = ["Last Game", "Last Attendance vs Opp"]
	# elif name == 'dataset_3':
	# 	numerical_features = ["Last Game", "Capacity", "Last Attendance vs Opp"]
	return numerical_features




