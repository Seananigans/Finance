# -*- coding: utf-8 -*-
"""
Created on Wed Feb 24 17:59:24 2016
@author: victor
"""

import numpy as np
import pandas as pd

class Bollinger():
    
    def __init__(self, window_length = 20, dev_factor=2):
        self.window = window_length
        self.dev = dev_factor
    
    def addPriceSeries(self, historical):
        self.historical = historical
        
    def getIndicator(self):
        #Compute rolling mean and std    
        ma = pd.rolling_mean(self.historical, window = self.window)
        sd = pd.rolling_std(self.historical, window = self.window)
        
        #Normalized indicator
        bollinger_ind = (self.historical - ma) / ( 2 * sd)
                
        #Rename dataframe
        return bollinger_ind.rename( columns=lambda x: "Bollinger_" + x)
        
        
def test_run():
    """Driver function."""
    from util import get_data
    
    # Define input parameters
    start_date = '2007-12-31'
    end_date = '2009-12-31'
    stock_symbol = ["IBM", "AAPL", "GE", "GLD", "SPY"]

    #Get stock quotation
    dates =  pd.date_range(start_date, end_date)
    stock_prices = get_data(stock_symbol, dates, addSPY=False)
    
    indicator = Bollinger()
    indicator.addPriceSeries( stock_prices )
    
    print indicator.getIndicator()
    

if __name__ == "__main__":
    test_run()
