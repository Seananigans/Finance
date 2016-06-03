import numpy as np
import pandas as pd

class Bollinger(object):
    def __init__(self, window=20):
        self.window = window
        self.name = "Bollinger_{}".format(window)

    def addEvidence(self, data):
        self.data = data

    def getIndicator(self):
        mva = pd.rolling_mean(self.data, self.window)
        sd = pd.rolling_std(self.data, self.window)
        boll = (self.data - mva) / (2*sd)
        
        boll.columns = ["Bollinger_"+x for x in boll.columns]

        return boll
