
import numpy as np
import src.data.train_test_split as split
import src.models.metrics as metrics
from sklearn.ensemble import AdaBoostRegressor, ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, TimeSeriesSplit


def random_forest_randomized_cv(name, n_iter = 50, cv = TimeSeriesSplit(5)):
	'''Conducts a randomized search of cross validation for given parameters of the random forest and returns results.

	Implements scoring criteria based off of custom dictionary.
	'''

	X_train, X_test, y_train, y_test, train = split.split_subset(name)

	random_grid_0 = {'n_estimators': np.linspace(start=50, stop=2000, num=50, dtype=int),
	'criterion': ['mae','mse'], 'max_features': np.arange(1, len(X_train.columns.values)+1, 3),
	'min_samples_split': [2, 4, 8, 16, 32, 64], 'min_samples_leaf': [1, 2, 4, 8, 16],
	'bootstrap' : [True, False]}

	random_grid_1 = {'n_estimators': np.arange(400,1851,50),
				   'max_features': np.arange(13,24),
				   'min_samples_split': [2, 4, 16],
				   'min_samples_leaf': [1, 4],
				   'bootstrap' : [False]}

	to_score = metrics.create_metrics()[0]
	rf = RandomForestRegressor(n_jobs=-1, random_state = 18, criterion = 'mae')
	rf_cv = RandomizedSearchCV(estimator=rf, param_distributions= random_grid_0, n_iter= n_iter, cv=cv, pre_dispatch = 16, 
	iid=False, scoring= to_score, random_state = 18).fit(X_train, y_train)

	return rf_cv

def random_forest_grid_cv(X_train, y_train, cv = TimeSeriesSplit(5)):
	'''Conducts a grid search over all possible combinations of given parameters and returns result.

	Uses parameters closely clustered around the best randomized search results.
	Also returns back best fitted model by specified criteria (MAE).
	'''

	X_train = X_train.copy()
	y_train = y_train.copy()
	to_score = metrics.create_metrics()[0]
	param_grid = {'n_estimators': np.arange(400,1851,50),
				   'max_features': [13,14],
				   'min_samples_split': [2, 4],
				   'min_samples_leaf': [1],
				   'bootstrap' : [False]}
				   
	rf = RandomForestRegressor(n_jobs = -1, random_state = 18, criterion = 'mae')
	rf_cv = GridSearchCV(estimator=rf, param_grid=param_grid, scoring = to_score, pre_dispatch = 16,
						 refit = "Mean Absolute Error", cv = cv).fit(X_train, y_train)
	return rf_cv

def adaboost_randomized_cv(X_train, y_train, n_iter = 200, cv = TimeSeriesSplit(5)):
	"""Conducts a randomized search of cross validation for given parameters of AdaBoost and returns results.

	"""

	X_train = X_train.copy()
	y_train = y_train.copy()
	to_score = metrics.create_metrics()[0]
	param_grid = {'n_estimators': np.linspace(20, 2000, 10, dtype=int),
				  'learning_rate': np.linspace(0.01, 1, 10)} 
	adaboost = AdaBoostRegressor(random_state = 18)
	adaboost_cv = RandomizedSearchCV(estimator=adaboost, param_distributions = param_grid, 
	                                 n_iter = n_iter, n_jobs=-1, pre_dispatch = 16, cv=cv, 
	                                 refit='Mean Absolute Error', iid=False, random_state = 18,
									 scoring = to_score).fit(X_train, y_train)


	return adaboost_cv

def adaboost_grid_cv(X_train, y_train, cv = TimeSeriesSplit(5)):
	'''Conducts a grid search over all possible combinations of given parameters and returns the result.

	Uses parameters closely clustered around the best randomized search results.
	Also returns back best fitted model by specified criteria (MAE).
	'''

	X_train = X_train.copy()
	y_train = y_train.copy()
	to_score = metrics.create_metrics()[0]
	param_grid = {'base_estimator': [],
	'n_estimators': np.linspace(20, 2000, 10, dtype=int),
	'learning_rate': np.linspace(0.01, 1, 10)} 

	adaboost = AdaBoostRegressor(random_state = 18)
	adaboost_cv = GridSearchCV(estimator=adaboost, parm_grid=param_grid, scoring = to_score, 
						 refit = "Mean Absolute Error", cv = cv, 
						 n_jobs=-1, pre_dispatch = 6,).fit(X_train, y_train)

	return adaboost_cv

def gradient_boosting_randomized_cv(X_train, y_train, n_iter = 200, cv = TimeSeriesSplit(5)):
	"""Conducts a randomized search of cross validation for given parameters of Gradient Boosting and returns results.

	"""	

	X_train = X_train.copy()
	y_train = y_train.copy()
	to_score = metrics.create_metrics()[0]
	param_grid = {'loss' : ['ls', 'lad', 'huber'] ,
	'learning_rate': [1e-3,1e-2,1e-1,1,10],
	'n_estimators': np.linspace(20, 2000, 10, dtype=int),
	'learning_rate': np.linspace(0.001, 1, 10)} 

	gradient_boosting = GradientBoostingRegressor(random_state = 18)
	gradient_boosting_cv = RandomizedSearchCV(estimator= gradient_boosting, n_jobs = -1, pre_dispatch = 16,
	                                          param_distributions = param_grid, n_iter = n_iter, cv=cv, 
	                                          refit='Mean Absolute Error', iid=False,
	                                          scoring = to_score).fit(X_train, y_train)

	return gradient_boosting_cv

def gradient_boosting_grid_cv(X_train, y_train, cv = TimeSeriesSplit(5)):
	"""Conducts a grid search over all possible combinations of given parameters and returns the result

	Uses parameters closely clustered around the best randomized search results.
	Also returns back best fitted model by specified criteria (MAE).
	"""

	X_train = X_train.copy()
	y_train = y_train.copy()
	to_score = metrics.create_metrics()[0]
	param_grid = {'loss' : ['ls', 'lad', 'huber'] ,
	'learning_rate': 5,
	'n_estimators': np.linspace(20, 2000, 10, dtype=int),
	'learning_rate': np.linspace(0.01, 1, 10)} 

	gradient_boosting = GradientBoostingRegressor(random_state = 18)
	gradient_boosting_cv = GridSearchCV(estimator= gradient_boosting, param_grid = param_grid,
										 cv= cv, refit = "Mean Absolute Error",
										iid = False, scoring = to_score).fit(X_train, y_train)

	return gradient_boosting_cv

def extra_trees_randomized_cv(X_train, y_train, n_iter = 200, cv = TimeSeriesSplit(5)):
	"""

	"""

	X_train = X_train.copy()
	y_train = y_train.copy()
	to_score = metrics.create_metrics()[0]
	extra_trees = ExtraTreesRegressor()

	random_grid = {'n_estimators': np.linspace(start=200, stop=2000, num=10, dtype=int),
				   'max_features': np.arange(1, len(X_train.columns.values)+1),
				   'min_samples_split': [2, 5, 10]}

	extra_trees_cv = RandomizedSearchCV(estimator=extra_trees, param_distributions=random_grid,
										n_iter=n_iter, cv=cv, n_jobs=-1, pre_dispatch = 6, 
										refit='Mean Absolute Error',iid=False, 
										scoring = to_score).fit(X_train, y_train)

	return extra_trees_cv

def extra_trees_grid_cv(X_train,y_train, cv = TimeSeriesSplit(5)):
	"""

	"""

	X_train = X_train.copy()
	y_train = y_train.copy()
	to_score = metrics.create_metrics()[0]
	extra_trees = ExtraTreesRegressor()
	param_grid = {'n_estimators': np.linspace(start=200, stop=2000, num=10, dtype=int),
				   'max_features': np.arange(1, len(X_train.columns.values)+1),
				   'min_samples_split': [2, 5, 10]}

	extra_trees_cv = GridSearchCV(estimator= extra_trees, param_grid = param_grid, n_jobs=-1, pre_dispatch = 6,
	 cv= cv, refit = "Mean Absolute Error", iid = False, scoring = to_score).fit(X_train, y_train)

	return extra_trees_cv






