import numpy as np
import pandas as pd
from sklearn.metrics import explained_variance_score, mean_absolute_error, make_scorer, mean_squared_error, r2_score

def create_metrics():
	'''Creates metrics that functions and the user can use.

	'''

	def mean_absolute_percentage_error(y_true, y_pred): 
		return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
	def root_mean_square_error(y_true, y_pred):
		return np.sqrt(mean_squared_error(y_true, y_pred))
	def mean_absolute_error_custom(y_true, y_pred):
		return np.abs(mean_absolute_error(y_true,y_pred))
	r2 = make_scorer(r2_score)
	evs = make_scorer(explained_variance_score)
	mae_custom = make_scorer(mean_absolute_error_custom, greater_is_better = False)
	rmse_custom = make_scorer(root_mean_square_error, greater_is_better = False)
	mape_custom = make_scorer(mean_absolute_percentage_error, greater_is_better = False)
	to_score = {'R^2': r2, "Explained Variance Score": evs,
			   "Mean Absolute Error": mae_custom, "Root Mean Square Error": rmse_custom,
			   "Mean Absolute Percent Error": mape_custom}
	scoring = {'R^2': r2_score, "Explained Variance Score": explained_variance_score,
			   "Mean Absolute Error": mean_absolute_error_custom, 
			   "Root Mean Square Error": root_mean_square_error,
			   "Mean Absolute Percent Error": mean_absolute_percentage_error}
	return to_score, scoring

def apply_metrics(name, y_true, y_pred):
	"""

	"""
	
	y_true = y_true.copy()
	y_pred = y_pred.copy()
	scoring = create_metrics()[1]
	keys = list(scoring.keys())
	scores = {}
	for metric in keys:
		score = scoring[metric](y_true, y_pred)
		scores[metric] = score
	scores = pd.DataFrame(scores.values(), index = scores.keys(), columns = [name]).transpose()
	return scores


