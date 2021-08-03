import pandas as pd, numpy as np, matplotlib.pyplot as plt

data = pd.read_csv('IXIC.csv')

# print(data[:10])
close = np.array(data['Close'])
date  = np.array(data['Date']) 
# filter = list(map(lambda x: True if x%1.5<=0 else False, range(len(account)-1)))
filter = list(map(lambda x: True if x%6<=0 else False, range(len(close))))

plt.plot(close[filter], '-x')
# data.plot('Date','Close')
# print(date[:10])
# print(len(date[:10]))
l = len(date[filter])

# plt.xticks(range(len(dates_data)), dates_data, rotation='vertical')

plt.xticks(range(l),date[filter], rotation='vertical')

plt.show()