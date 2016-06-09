import time
from dataset_construction import populate_webdata

if __name__=="__main__":
	start_time = time.time()
	populate_webdata(replace=True)
	print
	print "%s seconds to download S&P 500 data." % (time.time() - start_time)
	print