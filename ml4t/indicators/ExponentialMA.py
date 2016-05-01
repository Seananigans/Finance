import numpy as np
import pandas as pd

class ExponentialMA(object):
    def __init__(self, window=20):
        self.window = window

    def addEvidence(self, data):
        self.data = data

    def getIndicator(self):
        ema = self.data/pd.ewma(self.data, span=self.window) - 1
        ema.columns = ["EMA_"+x for x in ema.columns]

        return ema
