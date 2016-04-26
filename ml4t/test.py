import datetime as dt
import pandas as pd
from util import *

start_date = dt.datetime(2004,1,1)
end_date = dt.datetime(2006,1,1)
dates = pd.date_range(start_date, end_date)
symbols = ['AXP']

table = get_data(symbols, dates, False)
print table.shift(1)