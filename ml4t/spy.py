import os
import pandas as pd

#########
print """
Sorry. Currently not in use.
"""
exit()
#########

fhand = pd.read_csv("spy_list.csv")
spy_list = list(fhand.Symbols)
for i in spy_list:
	os.system('python learner_strategy.py {} 5 0.05'.format(i))
	os.system('python marketsim.py')