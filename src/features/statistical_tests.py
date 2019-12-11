import src.data.datasets as ds
import src.data.train_test_split as split
import numpy as np
import pandas as pd
from sklearn.feature_selection import f_regression, mutual_info_regression
from statsmodels.stats.outliers_influence import variance_inflation_factor



def collect_tests(name):
	'''Aggregates statistical test data for a dataset into a single DataFrame for notebook presentation.

	'''
	sets = split.split_subsets(name)
	name = name[0]
	X_train = sets[name][0]
	y_train = sets[name][2]
	vif = find_vifs(name, X_train).sort_index(0)
	sig = find_numerical_significance(name, X_train, y_train).sort_index(0)
	return pd.concat([vif, sig], axis = 1)

def find_vifs(name, X_train, tolerance = 5):
	'''Iteratively drops features from set of numerical training features based off of VIF scores.

	'''

	X_train = X_train.copy()
	number_numerical = ds.get_number_numerical()
	train_num = X_train.iloc[:,0:number_numerical[name]]
	vif = pd.DataFrame(index = train_num.columns)
	vif_x = train_num.copy()
	cols = vif_x.columns.values
	for i in np.arange(len(train_num.columns)):
		try:
			vifs = [variance_inflation_factor(vif_x.values, i) for i in range(vif_x.shape[1])]
			vifs_df = pd.DataFrame(vifs, columns=["VIF Round {}".format(i)], index = cols)
			if max(vifs) > tolerance:
				loc = np.where(vifs == max(vifs))[0][0]
				vif_x = vif_x.drop([cols[loc]], axis = 1)
				cols = np.delete(cols,loc, axis =0)
			else: 
				vif = pd.concat([vif, vifs_df], axis = 1)
				break
			vif = pd.concat([vif, vifs_df], axis = 1)
		except:
			break
	return vif  

def find_numerical_significance(name, X_train, y_train):
	'''Returns P-value of hypothesis test where H0 is that the feature has no effect on the outcome and 
	an estimate of the mutual information 

	'''

	X_train = X_train.copy()
	y_train = y_train.copy()
	number_numerical = ds.get_number_numerical()
	train_num = X_train.iloc[:,0:number_numerical[name]]

	f_r = pd.DataFrame(f_regression(train_num, y_train)[1], index = train_num.columns.values, 
					   columns = ["Numerical Feature Significance (P-Value) {}".format(name)])
	mir = pd.DataFrame(mutual_info_regression(train_num, y_train, random_state = 18), index = train_num.columns.values, 
					   columns = ["Estimated Mutual Information {}".format(name)])
	df = pd.concat([f_r,mir], axis =1)

	return df
   
