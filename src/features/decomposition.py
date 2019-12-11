import numpy as np
import pandas as pd
import src.data.datasets as ds
import src.data.train_test_split as split
import src.models.metrics as metrics
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression


def pca(name, X_train, X_test):
	"""

	"""

	X_train = X_train.copy()
	X_test = X_test.copy()
	num_numerical = ds.get_number_numerical(name)
	X_train_s = split.standardize(name, X_train)
	X_test_s = split.standardize(name, X_test)
	X_train_s_numerical = X_train_s.iloc[:,0:num_numerical]
	X_train_s_categorical = X_train_s.iloc[:,num_numerical:]
	X_test_s_numerical = X_test_s.iloc[:,0:num_numerical]
	X_test_s_categorical = X_test_s.iloc[:,num_numerical:]
	estimator = PCA(0.95)
	X_train_s_numerical_reduced = pd.DataFrame(estimator.fit_transform(X_train_s_numerical), 
	                                           index = X_train_s_categorical.index)
	X_test_s_numerical_reduced = pd.DataFrame(estimator.transform(X_test_s_numerical), 
	                                          index = X_test_s_categorical.index)
	X_train_s = pd.concat([X_train_s_numerical_reduced, X_train_s_categorical], axis = 1)
	X_test_s = pd.concat([X_test_s_numerical_reduced, X_test_s_categorical], axis = 1)
	return X_train_s, X_test_s


def pca_cv(name, X_train, X_test, y_train, y_test, save = False):
	X_train = X_train.copy()
	X_test = X_test.copy()
	y_train = y_train.copy()
	y_test = y_test.copy()
	num_numerical = ds.get_number_numerical(name)
	X_train_s = split.standardize(name, X_train)
	X_test_s = split.standardize(name, X_test)
	X_train_s_numerical = X_train_s.iloc[:,0:num_numerical]
	X_train_s_categorical = X_train_s.iloc[:,num_numerical:]
	X_test_s_numerical = X_test_s.iloc[:,0:num_numerical]
	X_test_s_categorical = X_test_s.iloc[:,num_numerical:]
	df = pd.DataFrame()
	ols = LinearRegression()
	for i in np.arange(1,num_numerical):
		pca = PCA(i)
		X_train_s_numerical_reduced = pd.DataFrame(pca.fit_transform(X_train_s_numerical), 
	                                     	  index = X_train_s_categorical.index)
		X_test_s_numerical_reduced = pd.DataFrame(pca.transform(X_test_s_numerical), 
	                                          index = X_test_s_categorical.index)
		X_train_s = pd.concat([X_train_s_numerical_reduced, X_train_s_categorical], axis = 1)
		X_test_s = pd.concat([X_test_s_numerical_reduced, X_test_s_categorical], axis = 1)

		model = ols.fit(X_train_s, y_train)
		preds = model.predict(X_test_s)
		preds = metrics.apply_metrics('{}: {} dimensions'.format(name, i), y_test, preds.ravel())
		df = pd.concat([df, preds], axis = 0)

	if save:
		to_save = Path().resolve().joinpath('models', 'cross_validation_outcomes', 'pca', '{}_pca_cv.csv'.format(name))
		df.to_csv(to_save)

	return df
