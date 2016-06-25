import pandas as pd
from pandas.tseries.offsets import BDay
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
import math, os, sys

# Import dataset retrieval
from dataset_construction import create_input, create_output
# Import normalization
from helpers.normalization import mean_normalization

fhand = pd.read_csv("spy_list.csv")
spy_list = list(fhand.Symbols)
results = pd.DataFrame()
results = results.append({'Date': np.nan, "Return Date": np.nan, 'Symbol': np.nan, 'Return': np.nan, 'Test Error (RMSE)': np.nan}, ignore_index=True)

df = create_input(spy_list[0])
for sym in spy_list[1:]:
	if sym == "ADT": continue
	df = df.join(create_input(sym, indicators=[], store=False))

df.corr().to_csv("StockCorrelations.csv")
