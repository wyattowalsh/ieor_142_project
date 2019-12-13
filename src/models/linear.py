import numpy as np
import pandas as pd
from pathlib import Path
import src.data.datasets as ds
import src.data.train_test_split as split
import src.models.metrics as metrics
from sklearn.linear_model import ElasticNetCV, LassoCV, LinearRegression, RidgeCV, HuberRegressor
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.svm import LinearSVR

def baseline_creation(save = False):
	'''

	'''
	file_names = ['dataset_1', 'dataset_2', 'dataset_3']
	names = ['Dataset 1', "Dataset 2", "Dataset 3"]
	sets = split.split_subsets(file_names)

	df_avgs = pd.DataFrame()
	df_ols = pd.DataFrame()
	for i, file_name in enumerate(file_names):
		y_bar = np.mean(sets[file_name][2])
		preds = np.ones(len(sets[file_name][3])) * y_bar
		avg_score = metrics.apply_metrics('{} Average'.format(names[i]), sets[file_name][3], preds)
		df_avgs = pd.concat([df_avgs, avg_score], axis = 0)
		ols_score = linear(file_name)[1]
		df_ols = pd.concat([df_ols, ols_score], axis = 0)
	
	if save == True:
		to_save_avgs = Path().resolve().joinpath('models', 'baseline', '{}.csv'.format('averages'))
		df_avgs.to_csv(to_save_avgs)
		to_save_ols = Path().resolve().joinpath('models', 'baseline', '{}.csv'.format('OLS'))
		df_ols.to_csv(to_save_ols)

	return df_avgs, df_ols

def collect_all_statistics(save = False):
	"""

	"""
	names = list(ds.get_names().keys())
	df = pd.DataFrame()
	for name in names:
		stats = collect_statistics(name)
		df = pd.concat([df, stats], axis = 0)

	if save:
		to_save = Path().resolve().joinpath('models', 'linear', '{}.csv'.format('performance_outcomes_all'))
		df.to_csv(to_save)	
	return df


def collect_statistics(name):
	'''Runs multiple variations of all linear models and outputs a dataframe with statistics
	

	Variations include: numerical predictor standardization, time series cross-validator, feature selection
	'''

	results = pd.DataFrame()
	models = [linear, ridge, lasso, elastic_net, huber, support_vector_machine]
	for model in models:
		if model == linear:
			results = pd.concat([results, model(name)[1]], axis = 0)
			continue
		for val in [False, True]:
			results = pd.concat([results, model(name, standardize = val)[1]], axis = 0)
			results = pd.concat([results, model(name, standardize = val, cv = TimeSeriesSplit(5))[1]], axis = 0)
	return results

def linear(name):
	'''Outputs a fitted Linear Regression Model.

	Inputs can be standardized or not
	'''

	display_name = ds.get_names()[name]
	X_train, X_test, y_train, y_test, train = split.split_subset(name)
	model = LinearRegression().fit(X_train, y_train)
	performance = metrics.apply_metrics('{} OLS'.format(display_name), y_test, model.predict(X_test))
	return model, performance

def ridge(name, cv = 5, standardize = False):
	'''Outputs a fitted Ridge Regression Model with a penalty term tuned through cross validation.

	'''

	if cv == 5:
		cv_type = 'K-Fold'
	else:
		cv_type = "Time Series Split"
	display_name = ds.get_names()[name]
	X_train, X_test, y_train, y_test, train = split.split_subset(name)
	if standardize:
		X_train = split.standardize(name, X_train)
		X_test = split.standardize(name, X_test)
	alphas = np.linspace(1e-4, (1e6)+1, 50)
	model = RidgeCV(alphas=alphas, fit_intercept=True, cv=cv).fit(X_train, y_train)
	if standardize:
		performance = metrics.apply_metrics('{} standardized {} Ridge'.format(display_name, cv_type), 
		                                    y_test, model.predict(X_test))
	else:
		performance = metrics.apply_metrics('{} {} Ridge'.format(display_name, cv_type), y_test, model.predict(X_test))
	return model, performance

def lasso(name, cv = 5, standardize = False):
	'''Outputs a fitted Lasso Regression Model with a penalty term tuned through cross validation.

	Inputs must be standardized.
	Number of folds in cross validation is by default 5.
	n_jobs = -1 allows for all local processors to be utilized.
	'''

	if cv == 5:
		cv_type = 'K-Fold'
	else:
		cv_type = "Time Series Split"
	display_name = ds.get_names()[name]
	X_train, X_test, y_train, y_test, train = split.split_subset(name)
	if standardize:
		X_train = split.standardize(name, X_train)
		X_test = split.standardize(name, X_test)
	model = LassoCV(n_alphas=100, verbose = 0, cv=5, n_jobs=-1, copy_X = True, random_state = 18).fit(X_train, y_train)
	if standardize:
		performance = metrics.apply_metrics('{} standardized {} Lasso'.format(display_name, cv_type),
		y_test, model.predict(X_test))
	else:
		performance = metrics.apply_metrics('{} {} Lasso'.format(display_name, cv_type), y_test, model.predict(X_test))
	return model, performance

def elastic_net(name, cv = 5, standardize = False):
	'''Outputs a fitted Elastic Net Regression Model with tuning parameters found through cross validation.
	
	Inputs must be standardized.
	l1_ratios are spread out on a log scale as recommended by package authors.
	Number of folds in cross validation is by default 5.
	n_jobs = -1 allows for all local processors to be utilized.
	# '''
	# if np.any(X_train.mean(axis = 0) > 1):
	# 	raise ValueError('Numerical features must be standardized')

	if cv == 5:
		cv_type = 'K-Fold'
	else:
		cv_type = "Time Series Split"
	display_name = ds.get_names()[name]
	X_train, X_test, y_train, y_test, train = split.split_subset(name)
	if standardize:
		X_train = split.standardize(name, X_train)
		X_test = split.standardize(name, X_test)
	l1_ratios = np.geomspace(1e-6,1,100)
	model = ElasticNetCV(l1_ratio=l1_ratios, n_alphas=100, cv = 5, verbose = 0, 
	                     n_jobs = -1, random_state = 18).fit(X_train, y_train)
	if standardize:
		performance = metrics.apply_metrics('{} standardized Elastic Net'.format(display_name, cv_type),
		y_test, model.predict(X_test))
	else:
		performance = metrics.apply_metrics('{} Elastic Net'.format(display_name, cv_type), y_test, model.predict(X_test))
	return model, performance

def huber(name, standardize = False, cv = 5):
	'''

	'''

	if cv == 5:
		cv_type = 'K-Fold'
	else:
		cv_type = "Time Series Split"

	display_name = ds.get_names()[name]
	X_train, X_test, y_train, y_test, train = split.split_subset(name)
	if standardize:
		X_train = split.standardize(name, X_train)
		X_test = split.standardize(name, X_test)
	to_score, scoring = metrics.create_metrics()
	param_grid = {'epsilon': np.linspace(1 + 1e-5, 2, 10),
	'alpha': np.linspace(1e-6, 10, 10)}
	model = HuberRegressor(max_iter = 500, warm_start = True)
	model_cv = GridSearchCV(model, param_grid= param_grid, scoring = to_score, 
	                        n_jobs = -1, pre_dispatch = 16, cv = cv,
	                        refit = False).fit(X_train, y_train)
	performance = pd.DataFrame()
	variations = get_model_variants(HuberRegressor, model_cv)
	for variation in variations:
		model = variation.fit(X_train, y_train).predict(X_test)
		if standardize:
			performance = pd.concat([performance, metrics.apply_metrics('{} standardized {} Huber'.format(display_name, cv_type),
			 y_test, model)], axis = 0)
		else:
			performance = pd.concat([performance, metrics.apply_metrics('{} {} Huber'.format(display_name, cv_type),
		 	y_test, model)], axis = 0)
		
	return model_cv, performance

def support_vector_machine(name, standardize = False, cv = 5):
	"""

	"""

	if cv == 5:
		cv_type = 'K-Fold'
	else:
		cv_type = "Time Series Split"
	display_name = ds.get_names()[name]
	X_train, X_test, y_train, y_test, train = split.split_subset(name)
	if standardize:
		X_train = split.standardize(name, X_train)
		X_test = split.standardize(name, X_test)
	to_score, scoring = metrics.create_metrics()
	param_grid = {'C': [2e-5,2e-3,2e-1,2e1,2e3,2e5,2e7,2e9,2e11],
	'loss': ['epsilon_insensitive', 'squared_epsilon_insensitive']}
	model = LinearSVR(dual = False, random_state = 18)
	model_cv = GridSearchCV(model, param_grid= param_grid, scoring = to_score, 
	                        n_jobs = -1,  pre_dispatch = 16, cv = cv, 
	                        refit = False).fit(X_train, y_train)
	performance = pd.DataFrame()
	variations = get_model_variants(LinearSVR, model_cv)
	for variation in variations:
		model = variation.fit(X_train, y_train).predict(X_test)
		if standardize:
			performance = pd.concat([performance, metrics.apply_metrics('{} standardized {} Linear SVM'.format(display_name, cv_type),
		 	y_test, model)], axis = 0)
		else: 
			performance = pd.concat([performance, metrics.apply_metrics('{} {} Linear SVM'.format(display_name, cv_type),
		 y_test, model)], axis = 0)

	return model_cv, performance

def get_model_variants(model, cv):
	'''

	'''
	cv_results = cv.cv_results_
	results = pd.DataFrame.from_dict(cv_results)
	bestr2 = results.loc[results['rank_test_$R^2$'] == 1, 'params'].values[0]
	
	# bestevs = results.loc[results['rank_test_Explained Variance Score'] == 1, 'params'].values[0]
	# evs = model(bestevs)
	bestmae = results.loc[results['rank_test_Mean Absolute Error'] == 1, 'params'].values[0]
	bestrmse = results.loc[results['rank_test_Root Mean Square Error'] == 1, 'params'].values[0]
	dict_list = [bestr2,bestmae, bestrmse]
	unique_dict_list = [dict(t) for t in {tuple(sorted(d.items())) for d in dict_list}]
	models = []
	for item in unique_dict_list:
		models = models + [model(**item)]
	# bestmape = results.loc[results['rank_test_Mean Absolute Percent Error'] == 1, 'params'].values[0]
	# mape = model(bestmape)
	
	return models 








