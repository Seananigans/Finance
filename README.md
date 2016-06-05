# Finance
1) What is this code for?
* This code is to establish a base in trading stocks using Machine Learning.
* Included are python files that serve a variety of funcitons
    * Backtesting Market Simulator (marketsim.py)
    * Machine Learning classes (learners/BagLearner.py, learners/KNNLearner.py, etc)
    * Getting and reading data (util.py)
    * Optimizing a portolio of stocks (optimize.py)
    * Analyzing the returns of a portfolio or stock (analyze.py)
    * A few files for testing (test.py, testlearner.py)
    * Indicator Classes for different indicators (indicators)
    * Strategy Classes for different strategies (strategies/bollinger_strategy.py, learner_strategy.py)
* Soon to come:
    * A Reinforcement learning agent that develops its own strategy for buying and selling stocks (Q_Learner.py)

2) How to get this code?
* Clone the repository from github:
   * `$git clone https://github.com/Seananigans/Finance`
* Build:
   * TODO: Add setup.py
   * Till then:
      * Use `$pip install *library` to install the following libraries:
         * numpy
         * scipy
         * pandas
         * scikit-learn
         * matplotlib
3) How do I use this code?
	1) Open a command window and navigate to the StockPredictor folder.
	2) Every new business day, run `python populate_spy.py` to populate the webdata folder with the past years worth of data up to the current day.
		* Data is retrieved for all ticker symbols in the S&P 500 using the yahoo finance API provided in the pandas\_datareader library.
	3) Run `python -W ignore predict_future.py 5` to predict returns for the companies with the top 10 highest return values.
		* The `5` in `python -W ignore predict_future.py 5` is the number of trading days into the future you wish to predict. This can be adjusted to be any number of days ahead you wish to predict.
		* More returns can be investigated in the `results.csv` file in the StockPredictor Folder.
		* While the program trains a simple linear regression model to predict future returns, the feature variables it uses were selected using the `test_indicators.py` program to minimize error.

4) How do I contribute?
* Simply click on the issues tab and tell me what you think needs changing.
