import numpy as np
import pandas as pd
import src.data.datasets as ds
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def split(dataset):
	"""Uses scikit-learn's train_test_split to split the given dataset and does one hot encoding for categorical features.

	30% of smallest dataset 
	Shuffle = False since our data has a time component and thus we want to test on most recent data.
	Adds a column for each unique category in a given column with a boolean datatype
	Returns X_train, X_test, y_train, y_test.
	"""

	dataset = dataset.copy()
	dataset = pd.get_dummies(dataset, columns=["Home", "Visitor", "Day of Week", 'Month', "Last Five"])
	X_train, X_test, y_train, y_test = train_test_split(dataset.drop(["Attendance"], axis=1), 
														dataset["Attendance"], shuffle=False, test_size = 3340)
	train = pd.concat([X_train,y_train],axis = 1)
	return X_train, X_test, y_train, y_test, train

def split_subset(name):
	'''

	'''

	dataset = ds.load_dataset(name)
	X_train_0, X_test_0, y_train_0, y_test_0, train_0 = split(dataset)
	splits = [X_train_0, X_test_0, y_train_0, y_test_0, train_0]
	return splits

def split_subsets(names):
	'''

	'''

	sets = dict.fromkeys(names)
	for name in names:
		dataset = ds.load_dataset(name)
		X_train_0, X_test_0, y_train_0, y_test_0, train_0 = split(dataset)
		splits = [X_train_0, X_test_0, y_train_0, y_test_0, train_0]
		sets[name] = splits
	return sets

def standardize(name, X):
	"""Splits dataset, one hot encodes categorical variables, and standardizes numerical features.

	Default value for test proportion is 0.25, which should perform well for our data.
	Shuffle = False since our data has a time component and thus we want to test on most recent data.
	Adds a column for each unique category in a given column with a boolean datatype
	Centers and scales numerical features
	Returns X_train, X_test, y_train, y_test
	"""

	X = X.copy()
	scaler = StandardScaler()
	number_numerical = ds.get_number_numerical()[name]
	X.iloc[:,0:number_numerical] = scaler.fit_transform(X.iloc[:,0:number_numerical])

	return X
