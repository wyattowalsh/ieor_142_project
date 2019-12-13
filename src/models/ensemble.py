import numpy as np
import pandas as pd
import src.data.train_test_split as split
import src.data.datasets as ds
import src.models.linear as linear
import src.models.metrics as metrics
from pathlib import Path
from sklearn.ensemble import AdaBoostRegressor, ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, TimeSeriesSplit
from sklearn.tree import DecisionTreeRegressor

def get_all_results(save=False):
	'''

	'''
	model_names = ['random_forest', 'adaboost', 'gradient_boosting', 'extra_trees']
	results = pd.DataFrame()
	for name in model_names:
		df = pd.read_csv(Path().resolve().joinpath('models', 'ensemble', '{}.csv'.format(name)), index_col = 0)
		results = pd.concat([results, df], axis = 0)

	if save:
		to_save = Path().resolve().joinpath('models', 'ensemble', 'performance_outcomes_all.csv')
		results.to_csv(to_save)

	return results

def get_results(model, save = False):
	'''

	'''
	names = ['dataset_1', 'dataset_2', 'dataset_3']
	models = [random_forest_grid_cv, adaboost_grid_cv, gradient_boosting_grid_cv, extra_trees_grid_cv]
	model_names = ['random_forest', 'adaboost', 'gradient_boosting', 'extra_trees']
	model_dict = dict(zip(models, model_names))	
	results = pd.DataFrame()
	for name in names:
		result_k_fold = model(name)[1]
		result_tss = model(name, cv = TimeSeriesSplit(5))[1]
		results = pd.concat([results,result_k_fold, result_tss], axis = 0)

	if save:
		to_save = Path().resolve().joinpath('models', 'ensemble', '{}.csv'.format(model_dict[model]))
		results.to_csv(to_save)

	return results

def get_rf_randomized_results(tss = False):
	'''

	'''

	names = ['dataset_1', 'dataset_2', 'dataset_3'] 
	results_dict = {}
	results_df = {}
	if tss:
		for name in names:
		    results_dict[name] = random_forest_randomized_cv(name = TimeSeriesSplit(5)).cv_results_
		    results_df[name] = pd.DataFrame.from_dict(results_dict[name])
		    to_save = Path().resolve().joinpath('models', 'cross_validation_outcomes', 
		                                        'ensemble', 'random_forest', 
		                                        '{}_time_series_split.csv'.format(name))
		    results_df[name].to_csv(to_save)

		return results_df
	else:
		for name in names:
		    results_dict[name] = random_forest_randomized_cv(name).cv_results_
		    results_df[name] = pd.DataFrame.from_dict(results_dict[name])
		    to_save = Path().resolve().joinpath('models', 'cross_validation_outcomes', 
		                                        'ensemble', 'random_forest', 
		                                        '{}_k_fold.csv'.format(name))
		    results_df[name].to_csv(to_save)

		return results_df

def random_forest_randomized_cv(name, n_iter = 25, cv = 5):
	'''Conducts a randomized search of cross validation for given parameters of the random forest and returns results.

	Implements scoring criteria based off of custom dictionary.
	'''

	X_train, X_test, y_train, y_test, train = split.split_subset(name)

	random_grid_0 = {'n_estimators': np.linspace(start=100, stop= 1000, num=10, dtype=int),
	'min_samples_split': [2, 4, 8, 16, 32],
	'min_samples_leaf': [1, 2, 4, 8, 16]}



	to_score = metrics.create_metrics()[0]
	rf = RandomForestRegressor(n_jobs = -1,random_state = 18, max_features= None, bootstrap = False)
	rf_cv = RandomizedSearchCV(estimator=rf, param_distributions= random_grid_0, n_jobs = -1, n_iter= n_iter, cv=cv,
	pre_dispatch = 16, scoring= to_score, random_state = 18, refit = False).fit(X_train, y_train)

	return rf_cv

def random_forest_grid_cv(name, cv = 5):
	'''Conducts a grid search over all possible combinations of given parameters and returns result.

	Uses parameters closely clustered around the best randomized search results.
	Also returns back best fitted model by specified criteria (MAE).
	'''

	if cv == 5:
		cv_type = 'K-Fold'
	else:
		cv_type = "Time Series Split"

	display_name = ds.get_names()[name]
	X_train, X_test, y_train, y_test, train = split.split_subset(name)
	to_score = metrics.create_metrics()[0]
	param_grid = {'n_estimators': np.arange(400,1851,50),
				   'min_samples_split': [2, 4],
				   'min_samples_leaf': [1]}
				   
	rf = RandomForestRegressor(n_jobs = -1,random_state = 18, max_features= None, bootstrap = False)
	rf_cv = GridSearchCV(n_jobs = -1, estimator=rf, param_grid=param_grid, scoring = to_score, pre_dispatch = 16,
						 refit = False, cv = cv).fit(X_train, y_train)

	performance = pd.DataFrame()
	variations = linear.get_model_variants(RandomForestRegressor, rf_cv)
	for variation in variations:
		model = variation.fit(X_train, y_train).predict(X_test)
		performance = pd.concat([performance, metrics.apply_metrics('{} {} Random Forest'.format(display_name, cv_type),
	 y_test, model)], axis = 0)

	return rf_cv, performance 

def get_ada_randomized_results(tss =  False):
	'''

	'''

	names = ['dataset_1', 'dataset_2', 'dataset_3'] 
	results_dict = {}
	results_df = {}
	if tss:
		for name in names:
		    results_dict[name] = adaboost_randomized_cv(name).cv_results_
		    results_df[name] = pd.from_dict(results_dict[name])
		    to_save = Path().resolve().joinpath('models', 'cross_validation_outcomes', 
		                                        'ensemble', 'adaboost',
		                                        '{}_time_series_split.csv'.format(name))
		    results_df[name].to_csv(to_save)

		return results_df
	else:
		for name in names:
		    results_dict[name] = adaboost_randomized_cv(name, cv = 5).cv_results_
		    results_df[name] = pd.from_dict(results_dict[name])
		    to_save = Path().resolve().joinpath('models', 'cross_validation_outcomes', 
		                                        'ensemble', 'adaboost', 
		                                        '{}_k_fold.csv'.format(name))
		    results_df[name].to_csv(to_save)

		return results_df

def adaboost_randomized_cv(name, n_iter = 25, cv = 5):
	"""Conducts a randomized search of cross validation for given parameters of AdaBoost and returns results.

	"""

	X_train, X_test, y_train, y_test, train = split.split_subset(name)
	to_score = metrics.create_metrics()[0]

	param_grid = {'base_estimator': [DecisionTreeRegressor(max_depth=2, random_state = 18),
	DecisionTreeRegressor(max_depth=3,random_state = 18), 
	DecisionTreeRegressor(max_depth=4,random_state = 18),
	DecisionTreeRegressor(max_depth=5,random_state = 18)],
	'n_estimators': np.linspace(50, 1000, 10, dtype=int),
	'loss': ['linear', 'square', 'exponential']} 

	adaboost = AdaBoostRegressor(random_state = 18)
	adaboost_cv = RandomizedSearchCV(estimator=adaboost, param_distributions = param_grid, 
	                                 n_iter = n_iter, n_jobs=-1, pre_dispatch = 16, cv=cv, 
	                                 refit=False, random_state = 18,
									 scoring = to_score).fit(X_train, y_train)


	return adaboost_cv

def adaboost_grid_cv(name, cv = 5):
	'''Conducts a grid search over all possible combinations of given parameters and returns the result.

	Uses parameters closely clustered around the best randomized search results.
	Also returns back best fitted model by specified criteria (MAE).
	'''

	if cv == 5:
		cv_type = 'K-Fold'
	else:
		cv_type = "Time Series Split"

	X_train, X_test, y_train, y_test, train = split.split_subset(name)
	to_score = metrics.create_metrics()[0]

	param_grid = {'base_estimator': [DecisionTreeRegressor(max_depth=2, random_state = 18),
	DecisionTreeRegressor(max_depth=3,random_state = 18), 
	DecisionTreeRegressor(max_depth=4,random_state = 18),
	DecisionTreeRegressor(max_depth=5,random_state = 18)],
	'n_estimators': np.linspace(50, 1000, 10, dtype=int)} 

	adaboost = AdaBoostRegressor(random_state = 18)
	adaboost_cv = GridSearchCV(estimator=adaboost, parm_grid=param_grid, scoring = to_score, 
						 refit = False, cv = cv, n_jobs=-1, pre_dispatch = 16).fit(X_train, y_train)

	display_name = ds.get_names()[name]
	performance = pd.DataFrame()
	variations = linear.get_model_variants(AdaBoostRegressor, adaboost_cv)
	for variation in variations:
		model = variation.fit(X_train, y_train).predict(X_test)
		performance = pd.concat([performance, metrics.apply_metrics('{} {} AdaBoost'.format(display_name, cv_type),
	 y_test, model)], axis = 0)

	return adaboost_cv, performance 

def get_gb_randomized_results(tss = False):
	'''

	'''

	names = list(ds.get_names().keys())
	results_dict = {}
	results_df = {}
	if tss:
		for name in names:
		    results_dict[name] = adaboost_grid_cv(name).cv_results_
		    results_df[name] = pd.from_dict(results_dict[name])
		    to_save = Path().resolve().joinpath('models', 'cross_validation_outcomes', 
		                                        'ensemble', 'gradient_boosting',
		                                        '{}_time_series_split.csv'.format(name))
		    results_df[name].to_csv(to_save)

		return results_df
	else:
		for name in names:
		    results_dict[name] = adaboost_grid_cv(name, cv = 5).cv_results_
		    results_df[name] = pd.from_dict(results_dict[name])
		    to_save = Path().resolve().joinpath('models', 'cross_validation_outcomes', 
		                                        'ensemble', 'gradient_boosting',
		                                        '{}_k_fold.csv'.format(name))
		    results_df[name].to_csv(to_save)

		return results_df

def gradient_boosting_randomized_cv(name, n_iter = 25, cv =5):
	"""Conducts a randomized search of cross validation for given parameters of Gradient Boosting and returns results.

	"""	

	X_train, X_test, y_train, y_test, train = split.split_subset(name)
	to_score = metrics.create_metrics()[0]
	param_grid = {'loss' : ['ls', 'lad', 'huber'] ,
	'learning_rate': np.arange(0.1,1.01, 0.1),
	'n_estimators': np.linspace(100, 1000, 10, dtype=int),
	'min_samples_split': [2, 4, 8, 16, 32, 64],
	'min_samples_leaf': [1, 2, 4, 8, 16],
	'max_depth': [2,3,4,5,10],
	'alpha': np.linspace(1e-6, 10, 10)}

	gradient_boosting = GradientBoostingRegressor(random_state = 18)
	gradient_boosting_cv = RandomizedSearchCV(estimator= gradient_boosting, n_jobs = -1, pre_dispatch = 16,
	                                          param_distributions = param_grid, n_iter = n_iter, cv=cv, 
	                                          refit=False,
	                                          scoring = to_score).fit(X_train, y_train)
	return gradient_boosting_cv

def gradient_boosting_grid_cv(name, cv = 5):
	"""Conducts a grid search over all possible combinations of given parameters and returns the result

	Uses parameters closely clustered around the best randomized search results.
	Also returns back best fitted model by specified criteria (MAE).
	"""

	if cv == 5:
		cv_type = 'K-Fold'
	else:
		cv_type = "Time Series Split"

	X_train, X_test, y_train, y_test, train = split.split_subset(name)
	to_score = metrics.create_metrics()[0]

	param_grid = {'loss' : ['ls', 'lad', 'huber'] ,
	'learning_rate': np.arange(0.1,1.01, 0.1),
	'n_estimators': np.linspace(20, 2000, 10, dtype=int),
	'learning_rate': np.linspace(0.01, 1, 10)} 

	gradient_boosting = GradientBoostingRegressor(random_state = 18)
	gradient_boosting_cv = GridSearchCV(n_jobs = -1, estimator= gradient_boosting, param_grid = param_grid,
										 cv= cv, refit = False, scoring = to_score, 
										 pre_dispatch = 16).fit(X_train, y_train)

	display_name = ds.get_names()[name]
	performance = pd.DataFrame()
	variations = linear.get_model_variants(GradientBoostingRegressor, gradient_boosting_cv)
	for variation in variations:
		model = variation.fit(X_train, y_train).predict(X_test)
		performance = pd.concat([performance, metrics.apply_metrics('{} {} Gradient Boosting'.format(display_name, cv_type),
	 y_test, model)], axis = 0)

	return gradient_boosting_cv, performance 

def get_et_randomized_results(tss = False):
	'''

	'''

	names = list(ds.get_names().keys())
	results_dict = {}
	results_df = {}
	if tss:
		for name in names:
		    results_dict[name] = adaboost_grid_cv(name).cv_results_
		    results_df[name] = pd.from_dict(results_dict[name])
		    to_save = Path().resolve().joinpath('models', 'cross_validation_outcomes', 
		                                        'ensemble', 'extra_trees',
		                                        '{}_time_series_split.csv'.format(name))
		    results_df[name].to_csv(to_save)

		return results_df
	else:
		for name in names:
		    results_dict[name] = adaboost_grid_cv(name, cv = 5).cv_results_
		    results_df[name] = pd.from_dict(results_dict[name])
		    to_save = Path().resolve().joinpath('models', 'cross_validation_outcomes', 
		                                        'ensemble', 'extra_trees',
		                                        '{}_k_fold.csv'.format(name))
		    results_df[name].to_csv(to_save)

		return results_df

def extra_trees_randomized_cv(name, n_iter = 25, cv = 5):
	"""

	"""

	X_train, X_test, y_train, y_test, train = split.split_subset(name)
	to_score = metrics.create_metrics()[0]
	extra_trees = ExtraTreesRegressor(random_state = 18, n_jobs = -1, max_features= None, bootstrap = False)

	random_grid = {'n_estimators': np.linspace(start=100, stop= 1000, num=10, dtype=int),
	'min_samples_split': [2, 4, 8, 16, 32],
	'min_samples_leaf': [1, 2, 4, 8, 16]}

	extra_trees_cv = RandomizedSearchCV(estimator=extra_trees, param_distributions=random_grid,
										n_iter=n_iter, cv=cv, n_jobs=-1, pre_dispatch = 16, 
										refit=False, 
										scoring = to_score).fit(X_train, y_train)

	return extra_trees_cv

def extra_trees_grid_cv(name, cv = 5):
	"""

	"""

	if cv == 5:
		cv_type = 'K-Fold'
	else:
		cv_type = "Time Series Split"

	X_train, X_test, y_train, y_test, train = split.split_subset(name)
	to_score = metrics.create_metrics()[0]
	extra_trees = ExtraTreesRegressor(n_jobs = -1, random_state = 18, max_features= None, bootstrap = False)
	param_grid = {'n_estimators': np.linspace(start=200, stop=2000, num=10, dtype=int),
				   'max_features': np.arange(1, len(X_train.columns.values)+1),
				   'min_samples_split': [2, 5, 10]}

	extra_trees_cv = GridSearchCV(n_jobs = -1, estimator= extra_trees, param_grid = param_grid, pre_dispatch = 16,
	 cv= cv, refit = False, scoring = to_score).fit(X_train, y_train)

	display_name = ds.get_names()[name]
	performance = pd.DataFrame()
	variations = linear.get_model_variants(ExtraTreesRegressor, extra_trees_cv)
	for variation in variations:
		model = variation.fit(X_train, y_train).predict(X_test)
		performance = pd.concat([performance, metrics.apply_metrics('{} Extra Trees'.format(display_name, cv_type),
	 y_test, model)], axis = 0)

	return extra_trees_cv, performance






