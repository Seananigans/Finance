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
* The easiest way to run the code is by running `spy.py` in the ml4t folder. `spy.py` runs a BagLearner on current data from the yahoo finance API. This will query Adjusted Closing prices over the past year for all ticker symbols in the S&P 500.

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

3) How do I contribute?
* Simply click on the issues tab and tell me what you think needs changing.
