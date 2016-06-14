import numpy as np
import pandas as pd

class Lag(object):
    def __init__(self, window=1):
        self.window = window
        self.name = "Lag_{}".format(window)

    def addEvidence(self, data):
        self.data = data

    def getIndicator(self):
    	lag = self.data.shift(self.window).pct_change()

        lag.columns = ["Lag{}_".format(self.window)+x for x in lag.columns]

        return lag
