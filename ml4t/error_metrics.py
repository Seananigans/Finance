import math
import numpy as np


def rmse(expected, predictions):
	"""@parameters: expected values and predicted values."""
	return math.sqrt(((expected - predictions) ** 2).sum()/expected.shape[0])

def mape(expected, predictions):
	return np.abs((expected - predictions)/expected).mean()
