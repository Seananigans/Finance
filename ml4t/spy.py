import os
import pandas as pd
from random import sample

###########
##print """
##Sorry. Currently not in use.
##"""
##exit()
###########

fhand = pd.read_csv("spy_list.csv")
spy_list = list(fhand.Symbols)
# spy_list = sample(spy_list,50)

for _, i in enumerate(spy_list):
	# Learner Strategy args = [1]symbol [2]horizon [3]threshold [4]num_shares [5]shorting?
	os.system('python learner_strategy.py {} 5 0.01 10 F'.format(i))
	# Market Simulator args = [1]create a plot [2]portfolio start value
	os.system('python marketsim.py T 1000')
