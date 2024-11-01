import numpy as np
import sklearn
from scipy.linalg import khatri_rao
from sklearn import svm
from sklearn import linear_model
# You are allowed to import any submodules of sklearn that learn linear models e.g. sklearn.svm etc
# You are not allowed to use other libraries such as keras, tensorflow etc
# You are not allowed to use any scipy routine other than khatri_rao

# SUBMIT YOUR CODE AS A SINGLE PYTHON (.PY) FILE INSIDE A ZIP ARCHIVE
# THE NAME OF THE PYTHON FILE MUST BE submit.py

# DO NOT CHANGE THE NAME OF THE METHODS my_fit, my_map etc BELOW
# THESE WILL BE INVOKED BY THE EVALUATION SCRIPT. CHANGING THESE NAMES WILL CAUSE EVALUATION FAILURE

# You may define any new functions, variables, classes here
# For example, functions to calculate next coordinate or step length

################################
# Non Editable Region Starting #
################################
def my_fit( X_train, y0_train, y1_train ):
################################
#  Non Editable Region Ending  #
################################
	X_mapped = my_map(X_train)
	mod0 = svm.LinearSVC(loss = 'hinge',C=10,tol = 0.001,max_iter = 2100)
	mod1 = svm.LinearSVC(C=10,max_iter = 2100)
	mod0.fit(X_mapped,y0_train)
	mod1.fit(X_mapped,y1_train)
	w0 = mod0.coef_
	b0 = mod0.intercept_
	w1 = mod1.coef_
	b1 = mod1.intercept_
	# Use this method to train your models using training CRPs
	# X_train has 32 columns containing the challenge bits
	# y0_train contains the values for Response0
	# y1_train contains the values for Response1
	# THE RETURNED MODELS SHOULD BE TWO VECTORS AND TWO BIAS TERMS
	# If you do not wish to use a bias term, set it to 0
	return w0, b0, w1, b1

################################
# Non Editable Region Starting #
################################
def my_map(X):
################################
#  Non Editable Region Ending  #
################################
	n_sam, n_col = X.shape
	ans = np.copy(X)
	prod_X = np.ones(n_sam)

	for i in range(n_col - 1, 0, -1):
		prod_X *= X[:, i]
		ans = np.column_stack((ans, prod_X))
	return ans
