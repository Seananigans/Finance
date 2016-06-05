import numpy as np
import pandas as pd

class RSI(object):
    def __init__(self, window=14):
        self.window = window
        self.name = "RSI_{}".format(window)

    def addEvidence(self, data):
        self.data = data
        
    def getIndicator(self):
        gain = (self.data-self.data.shift(1)).fillna(0)
        gain.columns = ["RSI_{}".format(self.window) for x in gain.columns]
        return pd.rolling_apply(gain, self.window , self.rsi_calc)

    def rsi_calc(self, prices):
        avg_gain = prices[prices>0].sum()/self.window
        avg_loss = -prices[prices<0].sum()/self.window
        rs = avg_gain/avg_loss
        return 100 - 100/(1+rs)
