

from os import name
from scipy.stats import pearsonr
import pandas as pd, numpy as np, joblib
import yfinance as yf
import matplotlib.pyplot as plt

from datetime import datetime
# datetime object containing current date and time
now = datetime.now()
# dd/mm/YY H:M:S
dt_string = now.strftime("%Y-%m-%d-%H-%M-%S")

def insider(data):
    data_dict = {}
    names = [i for i in data[0].split('\t')]
    for i in names:
        data_dict[i] = []
    print('data_dict', data_dict)
    print("names:", names)
    print(data[1].split('\t'))

    for k, line in enumerate(data[1:]):

        # print(line)
        if 'opbk' in line.lower():
            print(line)

        for j, val in enumerate(line.split('\t')):
            data_dict[names[j]].append(val)

        
    data_DF = pd.DataFrame(data_dict) 
    print("data_dict:\n", data_DF)
    
    # Remove entries that hasn't got 'Feb' in them
    print("len(data_DF):", len(data_DF))
    
    # data_DF = data_DF[data_DF['Date'].str.contains('Feb')]

    print("len(data_DF):", len(data_DF))
    data_DF = data_DF[data_DF['SEC Form 4\n'].str.contains('Feb 16')<0.5]
    print("len(data_DF):", len(data_DF))
    data_DF = data_DF[data_DF['SEC Form 4\n'].str.contains('Feb 15')<0.5]
    # data_DF = data_DF[data_DF['SEC Form 4\n'].str.contains('Feb 14')<0.5]
    print("len(data_DF):", len(data_DF))

    print("data_DF.iloc(1):\n", data_DF.iloc[0])
    print("len(data_DF):", len(data_DF))
    m = []
    close = []
    save = True

    if save:
        for i in range(len(data_DF)):
            line = data_DF.iloc[i]
            print('yoyoyoyo')
            # print(line)
            if 'feb' in line['SEC Form 4\n'].lower():
                print('feb')
                month = '02'
                startDate = f"2021-{month}-"+ line['SEC Form 4\n'][4:6]

                # print(start_date)
                # endDate = "2021-02-16" # 2021-02-07 for first dataset
                endDate = f"2021-02-{int(startDate[-2:])+6}" # 2021-02-07 for first dataset
                stock_data = yf.download(line['Ticker'],startDate,endDate)        
                # print('feb', 'stock_data:\n',stock_data)
                m.append((stock_data.Close[-1]-stock_data.Close[0])/stock_data.Close[0])
                close.append(stock_data.Close[0])

            elif 'jan' in line['Date'].lower():
                
                print('jan')
                month = '01'
                startDate = f"2021-{month}-"+ line['Date'][4:6]
                print('startDate:', startDate)
                endDate = "2021-02-07"
                stock_data = yf.download(line['Ticker'], startDate, endDate)        
                m.append((stock_data.Close[-1]-stock_data.Close[0])/stock_data.Close[0])
                close.append(stock_data.Close[0])
                
                # print(stock_data)

        joblib.dump(m,f'data/insider_price_change_pct_{dt_string}')
        joblib.dump(close,f'data/close_insider_{dt_string}')

    else: 
        m = np.array(joblib.load('data/'+'insider_price_change_pct'))
        close = np.array(joblib.load('data/'+'close_insider'))

    for i, val in enumerate(data_DF['#Shares']):
        data_DF['#Shares'][i] = float(val.replace(',', ''))

    # print("\ndata_DF after non feb removed")
    # print(data_DF)

    data_DF['insider_price_change_pct'] = m
    data_DF['close'] = close

    print(data_DF[['insider_price_change_pct', '#Shares']])
    data_DF = data_DF.drop_duplicates('Ticker')

    print("len(m), data_DF['insider_price_change_pct']:", len(m), len(data_DF['insider_price_change_pct']))

    print("######\n mean:", np.mean(data_DF['insider_price_change_pct']))
    print("######\n median:", np.median(data_DF['insider_price_change_pct']))
    print(m[:5])
    print("pearsonr(close, m):", pearsonr(data_DF['close'], data_DF['insider_price_change_pct']))

        
    # data_DF['#Shares'] = pd.to_numeric(data_DF['#Shares'])
    data_DF = data_DF.sort_values(by=['#Shares'])
    print("len(data_DF['insider_price_change_pct'][data_DF['insider_price_change_pct']>0])/len(data_DF['insider_price_change_pct']):", len(data_DF['insider_price_change_pct'][data_DF['insider_price_change_pct']>0])/len(data_DF['insider_price_change_pct']))

    print("type(data_DF['#Shares'][0]):",type(data_DF['#Shares'][0]))
    # print('data_DF sorted:\n', data_DF)
    if 0:
        fig, (ax1,ax2) = plt.subplots(1,2, figsize=[20,12]) 
        ax1.hist(m, bins=50)
        # ax2.scatter(data_DF['insider_price_change_pct'], data_DF['#Shares'])
        ax2.hist2d(close, data_DF['insider_price_change_pct'], bins= 50)
        plt.show()

    # startDate = '2021-02-03'
    # endDate = '2021-02-06'

    # data = yf.download('OPBK',startDate,endDate)

    # print(data)

def get_label(df_row):
    print('df_row', df_row)
    start_date = '2021 ' + df_row[0]['Date']
    start_date = datetime.datetime.strptime(start_date, "%Y %b %d")
    start_date = str(start_date).split(' ')[0]

    end_date = str(datetime.date.fromtimestamp(df_row[0]['time_stamp']))
    
    if start_date == end_date:
        return 0

    company = df_row[0]['Ticker']
    
    import yfinance as yf

    change = yf.download(company, start_date, end_date)
    return change.Close[-1]-change.Close[-1]

if __name__=='__main__':
    # with open('insider_data.txt') as fil:
    #     data0 = fil.readlines()

    # insider(data0)

    with open('insider_data_21-02-09_21-02-16.txt') as fil:
        data0 = fil.readlines()

        insider(data0)