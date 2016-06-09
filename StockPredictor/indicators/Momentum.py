import numpy as np
import pandas as pd

class Momentum(object):
    def __init__(self, window=5):
        self.window = window
        self.name = "Momentum_{}".format(window)

    def addEvidence(self, data):
        self.data = data

    def getIndicator(self):
    	mom = (self.data/self.data.shift(self.window)) - 1
        mom.columns = [self.name+"_"+x for x in mom.columns]

        return mom
