import numpy as np
import src.models.metrics as metrics
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.neighbors import KNeighborsRegressor

def k_neighbors_randomized_cv(X_train,y_train, n_iter = 250, cv = 5):
	"""

	"""

	to_score = metrics.create_metrics()[0]
	param_grid = {'n_neighbors': np.arange(2**1,2**11, 20, dtype = int),
	'weights': ['uniform', 'distance'],
	'algorithm': ['ball_tree', 'kd_tree', 'brute'],
	'leaf_size': [2**1,2**2,2**3,2**4,2**5,2**6,2**7,2**8, 2**9, 2**10],
	'p': [1,2,3]}	

	model = KNeighborsRegressor(n_jobs = -1)
	model_cv = RandomizedSearchCV(estimator=model, param_distributions=param_grid, n_iter= n_iter, cv=cv, 
							   refit= "Mean Absolute Error", iid=False, scoring= to_score).fit(X_train, y_train)

	return model_cv