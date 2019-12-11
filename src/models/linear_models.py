import numpy as np
import pandas as pd
import src.data.train_test_split as split
import src.models.metrics as metrics
from functools import partial
from sklearn.linear_model import ElasticNetCV, LassoCV, LinearRegression, RidgeCV, HuberRegressor
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.svm import LinearSVR

def collect_all_statistics(names):
	"""

	"""

	splits = split.split_subsets(names)
	df = pd.DataFrame()
	for name in names:
		stats = collect_statistics(name, splits[name][0], splits[name][1], splits[name][2], splits[name][3])
		df = pd.concat([df, stats], axis = 0)

	return df


def collect_statistics(name, X_train, X_test, y_train, y_test):
	'''Runs multiple variations of all linear models and outputs a dataframe with statistics
	

	Variations include: numerical predictor standardization, time series cross-validator, feature selection
	'''
	X_train = X_train.copy()
	X_test = X_test.copy()
	y_train = y_train.copy()
	y_test = y_test.copy()
	X_train_s = split.standardize(name, X_train)
	X_test_s = split.standardize(name, X_test)

	variation_strings = ['', ' with Standardized Features']
	variation_training = [X_train, X_train_s]
	variation_test = [X_test, X_test_s]
	models = [linear, ridge, lasso, elastic_net, huber, support_vector_machine]
	model_names = ["OLS", "Ridge", "Lasso", "Elastic Net", "Huber", "LinearSVR"]
	regression_statistics = pd.DataFrame()
	for i,model in enumerate(models):
		built_models = list(map(partial(model, y_train = y_train), variation_training))
		model_name = model_names[i]
		for j, variation in enumerate(variation_strings):
			model_metrics = metrics.apply_metrics('{} {}{}'.format(name, model_name, variation), 
			                                      y_test, built_models[j].predict(variation_test[j]))
			regression_statistics = pd.concat([regression_statistics, model_metrics], axis = 0)
			
	for i, model in enumerate(models[1:]):
		built_models = list(map(partial(model, cv =TimeSeriesSplit(5), y_train = y_train), variation_training))
		model_name = model_names[i]
		if model_name == 'OLS':
			continue
		for j, variation in enumerate(variation_strings):
			model_metrics = metrics.apply_metrics('{} {}{} TimeSeriesSplit'.format(name, model_name, variation), 
			                                      y_test, built_models[j].predict(variation_test[j]))
			regression_statistics = pd.concat([regression_statistics, model_metrics], axis = 0)
	return regression_statistics

def linear(X_train, y_train):
	'''Outputs a fitted Linear Regression Model.

	Inputs can be standardized or not
	'''

	X_train = X_train.copy()
	y_train = y_train.copy()
	model = LinearRegression().fit(X_train, y_train)
	return model

def ridge(X_train, y_train, cv = 5):
	'''Outputs a fitted Ridge Regression Model with a penalty term tuned through cross validation.

	'''

	X_train = X_train.copy()
	y_train = y_train.copy()
	alphas = np.linspace(1e-4, (1e6)+1, 50)
	model = RidgeCV(alphas=alphas, fit_intercept=True, cv=cv).fit(X_train, y_train)
	return model

def lasso(X_train, y_train, cv = 5):
	'''Outputs a fitted Lasso Regression Model with a penalty term tuned through cross validation.

	Inputs must be standardized.
	Number of folds in cross validation is by default 5.
	n_jobs = -1 allows for all local processors to be utilized.
	'''

	X_train = X_train.copy()
	y_train = y_train.copy()
	model = LassoCV(n_alphas=100, verbose = 0, cv=5, n_jobs=-1, copy_X = True).fit(X_train, y_train)
	return model

def elastic_net(X_train, y_train, cv = 5):
	'''Outputs a fitted Elastic Net Regression Model with tuning parameters found through cross validation.
	
	Inputs must be standardized.
	l1_ratios are spread out on a log scale as recommended by package authors.
	Number of folds in cross validation is by default 5.
	n_jobs = -1 allows for all local processors to be utilized.
	# '''
	# if np.any(X_train.mean(axis = 0) > 1):
	# 	raise ValueError('Numerical features must be standardized')

	X_train = X_train.copy()
	y_train = y_train.copy()
	l1_ratios = np.geomspace(1e-6,1,100)
	model = ElasticNetCV(l1_ratio=l1_ratios, n_alphas=100, cv = 5, verbose = 0, 
	                     n_jobs = -1).fit(X_train, y_train)
	return model

def huber(X_train, y_train, cv = 5):
	'''

	'''

	X_train = X_train.copy()
	y_train = y_train.copy()
	to_score, scoring = metrics.create_metrics()
	param_grid = {'epsilon': np.linspace(1, 2, 25),
	'alpha': np.linspace(1e-6, 1e6+1, 25)}
	model = HuberRegressor(max_iter = 250)
	model_cv = GridSearchCV(model, param_grid= param_grid, scoring = to_score, 
	                        n_jobs = -1, pre_dispatch = 4, cv = cv, 
	                        refit = 'Mean Absolute Error').fit(X_train, y_train)
	return model_cv

def support_vector_machine(X_train, y_train, cv = 5):
	"""

	"""

	X_train = X_train.copy()
	y_train = y_train.copy()
	to_score, scoring = metrics.create_metrics()
	param_grid = {'C': [2e-5,2e-3,2e-1,2e1,2e3,2e5,2e7,2e9,2e11]}
	model = LinearSVR(dual = False, random_state = 18, loss = 'squared_epsilon_insensitive')
	model_cv = GridSearchCV(model, param_grid= param_grid, scoring = to_score, 
	                        n_jobs = -1,  pre_dispatch = 4, cv = cv,
	refit = 'Mean Absolute Error')
	fitted_model = model_cv.fit(X_train, y_train)
	return fitted_model







