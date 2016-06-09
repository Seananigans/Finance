import numpy as np
import pandas as pd

class ExponentialMA(object):
    def __init__(self, window=20):
        self.window = window
        self.name = "EMA_{}".format(window)

    def addEvidence(self, data):
        self.data = data

    def getIndicator(self):
        ema = self.data/pd.ewma(self.data, span=self.window) - 1
        ema.columns = [self.name+"_"+x for x in ema.columns]

        return ema
