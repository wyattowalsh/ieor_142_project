import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import lasso_path

def plot_lasso_coefs(X_train, y_train):
	alphas, coefs, dual_gaps = lasso_path(X_train, y_train, eps=0.001, n_alphas=100)
	sns.set_style("whitegrid")
	sns.set_palette(sns.color_palette('bright',12))
	fig, ax = plt.subplots(figsize=(60, 25))
	for i in range(len(coefs)):
		sns.lineplot(x = alphas, y = coefs[i,:], ax = ax)
	ax.set_xlabel('Alpha Parameter', fontsize = 40)
	ax.set_ylabel('Coefficient Value', fontsize = 40) 
	ax.tick_params(labelsize=30)
	ax.set_title(label = "Lasso Regression Coefficient Path", fontsize = 50)
	ax.legend(X_train.columns.values,loc="upper right", fontsize=25)
	plt.show()

