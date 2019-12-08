import numpy as np
import pandas as pd
import warnings
from pathlib import Path

def load_dataset(name): 
	'''

	'''

	try: 
		dataset = pd.read_csv(Path().resolve().joinpath('data', 'processed', '{}.csv'.format(name)), 
		                      index_col = 0)
		dataset.index = pd.to_datetime(dataset.index)
		return dataset
	except:
		warnings.warn('{} does not exist'.format(name))

def load_datasets(names = ['dataset_1',"dataset_2", "dataset_3"]):
	'''

	'''

	datasets = {}
	for i,name in enumerate(names):
		datasets[str(i+1)] = load_dataset(name)
		if datasets[str(i+1)] is None:
			warnings.warn('{} does not exist'.format(name))
			return
	return datasets

def save_dataset(name, data):
	"""

	"""

	data = data.copy()
	to_save = Path().resolve().joinpath('data', 'processed', '{}.csv'.format(name))
	data.to_csv(to_save)

def create_datasets():
	dataset()
	dataset_1()
	dataset_2()
	dataset_3()

def get_number_numerical(name):
	if name == 'dataset_1':
		numerical_features = 7
	elif name == 'dataset_2':
		numerical_features = 7
	elif name == 'dataset_3':
		numerical_features = 7
	# elif name == 'dataset_2':
	# 	numerical_features = ["Last Game", "Last Attendance vs Opp"]
	# elif name == 'dataset_3':
	# 	numerical_features = ["Last Game", "Capacity", "Last Attendance vs Opp"]
	return numerical_features

def dataset():
	"""This dataset contains games from all the years scraped.

	It does not include popularity or capacity data.
	It does include the lagged attendance feature and last attendance versus the same opponent
	It also removes games with attendance over 25,000
	"""
	data = pd.read_csv(Path().resolve().joinpath('data', 'raw', 'game_data.csv'), index_col = 0)
	data = data.sort_values('Time').reset_index(drop=True).set_index('Time', drop=True)
	data.index = pd.to_datetime(data.index)
	# Remove extreme outliers
	data = data.loc[data['Attendance'] <= 25000]

	data['Last Game'] = np.nan
	data["Last Attendance vs Opp"] = np.nan
	teams = data['Home'].unique()
	for team in teams:
		data.loc[data['Home'] == team, 'Last Game'] = (data.loc[data['Home'] == team, 'Attendance'].shift(1)) 
		data.loc[data['Home'] == team] = data.loc[data['Home'] == team].sort_values(['Visitor', 'Time'])
		tot_scores = np.array([])
		for visitor in data.loc[data['Home'] == team]['Visitor'].unique():
			scores = data.loc[data['Home'] == team].loc[data.loc[data['Home'] == team]['Visitor'] == visitor]['Attendance'].values
			scores = np.append(np.array([0]), scores[0:len(scores)-1])
			tot_scores = np.append(tot_scores, scores)
		data.loc[data['Home'] == team, 'Last Attendance vs Opp'] = tot_scores
	data = data.dropna()

	# Remove seemingly redundant or irrelevant features
	data = data.drop(["V PTS", "H PTS", "Match-up"], axis=1)
	data = data[['V Pop', 'H Pop', 'Curr Win %',  'LS Win %','Last Game', 'Last Attendance vs Opp', 
	'Capacity', "Home", "Visitor",'Playoffs?', 'Last Five', 'Day of Week','Month', 'Rivalry?', 'Attendance']]
	data = data.dropna()
	save_dataset('dataset', data)
	return


def dataset_1():
	'''

	'''
	data = load_dataset('dataset')
	data = data.drop(['V Pop', "H Pop", 'Rivalry?'], axis = 1)
	save_dataset('dataset_1', data)
	return

def dataset_2():
	'''

	''' 

	data = load_dataset('dataset')
	stadiums = pd.read_csv(Path().resolve().joinpath('data', 'raw', 'stadiums_data.csv'), index_col = 0)

	# Remove games that were played in old stadiums
	teams = data['Home'].unique()
	for team in teams:
		earliest = pd.to_numeric(stadiums.loc[stadiums['Home'] == team]['Opened'].values[0])
		data.loc[data['Home'] == team] = data.loc[data['Home'] == team].loc[data.loc[data['Home'] == team].index >= pd.Timestamp(earliest, 7, 1)]

	data = data.drop(["Home"], axis = 1)
	data = data.dropna()
	save_dataset('dataset_2', data)
	return 

def dataset_3():
	data = load_dataset('dataset_2')
	data = data.loc[data.index.year >= 2004]	
	save_dataset('dataset_3', data)


