import os
import pandas as pd

spy_list = ["AAPL","ABT","ABBV","ACN","ADBE","ATVI","AYI","MMM","IBM"]
fhand = pd.read_csv("spy_list.csv")
spy_list = list(fhand.Symbols)
for i in spy_list:
	os.system('python learner_strategy.py {} 10 0.04'.format(i))
	os.system('python marketsim.py')