import matplotlib.pyplot as plt

def plot_histogram(trainY):
	"""Plots a histogram of the input with vertical lines 
	indicating the mean and +/- 1 standard deviation."""
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
		"Avg. {} Day\nReturn:	{}%".format(5,
											round(mn*100,2)) for mn in mns]
	labels[mns.shape[0]:] = [
		"Std. Dev.\nof Returns: {}%".format(round(sd*100,2)) for sd in sds]
	plt.legend(lines,#[mean_line, std_line],
			   labels)
	plt.show()