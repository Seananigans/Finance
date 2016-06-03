import re
import numpy as np
import sys

try:
        fhand = open(sys.argv[1])
except IndexError:
        fhand = open('spytest.txt')

port_returns = []
spy_returns = []
port_sharpe = []
spy_sharpe = []

for line in fhand:
	if re.search("Cumulative Return of Fund:", line):
		returns = float(line.split()[-1])
		port_returns.append(returns)
	elif re.search("Cumulative Return of ", line):
		returns = float(line.split()[-1])
		spy_returns.append(returns)
	if re.search("Sharpe Ratio of Fund:", line):
		returns = float(line.split()[-1])
		port_sharpe.append(returns)
	elif re.search("Sharpe Ratio of ", line):
		returns = float(line.split()[-1])
		spy_sharpe.append(returns)

print "Average Portfolio Return: {}".format(np.mean(port_returns))
print "Average SPY Return: {}".format(np.mean(spy_returns))
print "Average Portfolio Sharpe Ratio: {}".format(np.nanmean(port_sharpe))
print "Average SPY Sharpe Ratio: {}".format(np.nanmean(spy_sharpe))
fhand.close()
