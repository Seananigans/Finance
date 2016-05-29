"""
Test for neural net learners.
"""

import numpy as np
import math
import matplotlib.pyplot as plt
import pandas as pd
from learners import LinRegLearner as lrl
from learners import KNNLearner as knn
from learners import BagLearner as bag
##from learners import NeuralRegLearner as net
from learners import ANNRegLearner as net

def mean_subtraction(trainX, testX):
    trnX = ( trainX - trainX.mean(axis=0) )
    tstX = ( testX - trainX.mean(axis=0) )
    return trnX, tstX
    
def mean_normalization(trainX, testX):
    trnX = ( trainX - trainX.mean(axis=0) )/trainX.std(axis=0)
    tstX = ( testX - trainX.mean(axis=0) )/trainX.std(axis=0)
    return trnX, tstX
    
def max_normalization(trainX, testX):
    trnX = trainX / trainX.max(axis=0)
    tstX = testX / trainX.max(axis=0)
    return trnX, tstX
    
def plot_histogram(trainY):
    mns = trainY.mean(axis=0)
    sds = trainY.std(axis=0)
    plt.hist(trainY)
    plt.xlabel("Daily Returns")
    plt.ylabel("Counts")
    mean_lines = [plt.axvline(mn, color="k", lw=3) for mn in mns]
    upper_std_lines = [plt.axvline(mn + sd, color="r", lw=2) for sd in sds]
    lower_std_lines = [plt.axvline(mn - sd, color="r", lw=2) for sd in sds]
    std_lines = upper_std_lines+lower_std_lines
    lines = mean_lines+std_lines
    #Create Labels
    labels = [None for i in lines]
    labels[:mns.shape[0]] = [
        "Avg. {} Day\nReturn:   {}%".format(5,
                                            round(mn*100,2)) for mn in mns]
    labels[mns.shape[0]:] = [
        "Std. Dev.\nof Returns: {}%".format(round(sd*100,2)) for sd in sds]
    plt.legend(lines,#[mean_line, std_line],
               labels)
    plt.show()
    
if __name__=="__main__":

	# get actual data
	filename= "simData/example.csv"
	
	df = pd.read_csv(filename, index_col='Date',
					parse_dates=False, na_values=['nan'])
					
	data = df.values
	cols = [col for col in df.columns if not col.startswith("Returns")]
	outcome = [col for col in df.columns if col.startswith("Returns")]
	
	# compute how much of the data is training and testing
	train_rows = int(math.floor(0.6* df.shape[0]))
	test_rows = int(df.shape[0] - train_rows)
	
	# separate out training and testing data
	trainX = df.ix[:train_rows,0:-1]
	trainY = df.ix[:train_rows,-1]
	testX = df.ix[train_rows:,0:-1]
	testY = df.ix[train_rows:,-1]
	
	trainX, testX = mean_normalization(trainX, testX)
	
	# create learners list and train them
	opt_var =  [10**j for j in range(-3,3)]
	learners = [ net.ANNRegLearner(lmbda=i, use_trained=False) for i in opt_var]
    
	cors, rmsestrain, rmsestest = [], [], []
	best_cor = 0
	best_rmse = np.inf
	
	if testY.min()<=-0.99 or testY.min()<=0.01: 
		no_cor = True
	else: no_cor = False
	
	for i, learner in enumerate(learners):
		print learner.name
		learner.addEvidence(trainX.values, trainY) # train it

		# evaluate in sample
		predYtrain = learner.query(trainX) # get the predictions
		rmse = math.sqrt(((trainY.values - predYtrain) ** 2).sum()/trainY.shape[0])
		print
		print "In sample results"
		print "RMSE: ", rmse
		predYtrain = predYtrain.reshape(trainY.values.shape)
		# Calculate TRAINING Mean Absolute Percent Error
		mape = (np.abs(trainY - predYtrain)/trainY).mean()
		print "MAPE: ", mape
		if not no_cor:
			c = np.corrcoef(predYtrain, y=trainY.values)
			print "corr: ", c[0,1]
		rmsestrain.append(rmse)

		# evaluate out of sample
		predY = learner.query(testX) # get the predictions
		rmse = math.sqrt(((testY.values - predY) ** 2).sum()/testY.shape[0])
		print
		print "Out of sample results"
		print "RMSE: ", rmse
		# Calculate TEST Mean Absolute Percent Error
		mape = (np.abs(testY - predY)/testY).mean()
		print "MAPE: ", mape
		if testY.min()==-1.0:
			predY[predY>0]=1.0
			predY[predY<=0]=-1.0
		elif testY.min()==0.0:
			predY[predY>0.5]=1.0
			predY[predY<=0.5]=0.0
		if no_cor:
			accuracy = float((testY.values == predY).mean())
			print "Accuracy: ", accuracy
		predY = predY.reshape(testY.values.shape)
		if not no_cor:
			c = np.corrcoef(predY, y=testY.values)
			print "corr: ", c[0,1]
		print
		if not no_cor and c[0,1]>best_cor and rmse<best_rmse:
			learner.network.save("network_models/{}stocknet.txt".format("COMPANY_NAME"))
			best_cor = c[0,1]
			best_rmse = rmse
		if not no_cor: cors.append(c[0,1])
		else: cors.append(accuracy)
		rmsestest.append(rmse)
		predicted = pd.DataFrame(predY,
                                         columns=["Predicted"],
                                         index=df.ix[train_rows:,:].index)
		predicted = predicted.join(testY)
		
		if i%2==0 and testY.min() not in [-1., 0.]:
			plt.figure(1)
			plt.subplot(211)
			pre, = plt.plot(predicted[['Predicted']].values)
			act, = plt.plot(predicted[outcome].values)
			plt.xticks(range(0,predicted.shape[0],50),predicted.index)
			plt.legend([pre, act], ["Predicted", "Actual"])
			plt.xlabel("Date")
			plt.ylabel("Returns")
			plt.subplot(212)
			# Correlation between predicted and actual
# 			predY, testY = mean_subtraction(predY, testY)
			plt.scatter(predY, testY)
			plt.axvline(predY.mean(), color='r')
			plt.axhline(testY.mean(), color='r')
			plt.scatter([predY.mean()], [testY.mean()], color='r')
			plt.plot([testY.min(), testY.max()],[testY.min(), testY.max()], color='g')
# 			plt.plot([predY.min(), predY.max()],[predY.min(), predY.max()], color='r')
			plt.xlabel("Predicted Returns")
			plt.ylabel("Actual Returns")
			plt.show()
		df1 = pd.DataFrame(testX,
						   columns=cols,
						   index=df.ix[train_rows:,:].index)
		df1 = df1.join(predicted)
		df1.to_csv("test.csv", index_label="Date")
	
	if len(cors)>2:
		plt.plot(range(len(cors)), cors)
		plt.ylabel("Correlation")
		plt.xlabel("Model Complexity")
		plt.xticks(range(len(cors)),opt_var)
		plt.show()
		# Plot testing & training error on the same plot to 
		# show how error behaves with different models.
		testerr, = plt.plot(range(len(cors)), rmsestest, label="Test Error")
		trainerr, = plt.plot(range(len(cors)), rmsestrain, label="Training Error")
		plt.legend([testerr,trainerr], ["Test Error", "Train Error"])
		plt.xlabel("Model Complexity")
		plt.ylabel("RMSE")
		plt.xticks(range(len(cors)),opt_var)
		plt.show()
