import numpy as np
import pandas as pd
import datetime as dt

class Weekdays(object):
    def __init__(self, window=1):
        pass

    def addEvidence(self, data):
        self.data = data

    def getIndicator(self):
        daynames = [
            "Sunday",
            "Monday",
            "Tuesday",
            "Wednesday",
            "Thursday",
            "Friday",
            "Saturday"]
    	wkday = [d.weekday() for d in self.data.index]
    	week = np.zeros((self.data.shape[0],7))
    	for i in range(7):
            week[:,i] = np.array(wkday)==i
        day_indicator = pd.DataFrame(week,
                                     index= self.data.index,
                                     columns=daynames)
    	return day_indicator
