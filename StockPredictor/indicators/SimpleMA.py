import numpy as np
import pandas as pd

class SimpleMA(object):
    def __init__(self, window=20):
        self.window = window
        self.name = "SMA_{}".format(window)

    def addEvidence(self, data):
        self.data = data

    def getIndicator(self):
        sma = self.data/pd.rolling_mean(self.data, self.window) - 1
        sma.columns = [self.name+"_"+x for x in sma.columns]

        return sma
