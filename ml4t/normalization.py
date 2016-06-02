import numpy as np

def mean_normalization(trainX, testX):
	"""Returns the features normalized by the mean and standard deviation of
	feature values in the training set."""
	trnX = ( trainX - trainX.mean(axis=0) )/trainX.std(axis=0)
	tstX = ( testX - trainX.mean(axis=0) )/trainX.std(axis=0)
	return trnX, tstX
	
def max_normalization(trainX, testX):
	"""Returns the features normalized by the maximum 
	feature values in the training set."""
	trnX = trainX / trainX.max(axis=0)
	tstX = testX / trainX.max(axis=0)
	return trnX, tstX