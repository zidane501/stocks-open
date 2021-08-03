
import matplotlib.pyplot as plt, numpy as np, pandas as pd
import joblib, os
from scipy import stats
from helper_functions import get_local_minima_and_maxima
# Import the yfinance. If you get module not found error the run !pip install yfiannce from your Jupyter notebook
import yfinance as yf  

def make_data(startDate, endDate, filename = "historic_data_from|{startDate}|to|{endDate}|"):
    """[Makes an array of companies closing data from startdate to enddate, and saves it in to a Pandas Dataframe,
    with company names as row names and dates as column names ]

    Arguments:
        startDate {string} -- [start date on the form '2016-01-01']
        endDate {string} -- [end date on the form '2020-01-25']
        filename {string} -- [Default filename: "historic_data_from|{startDate}|to|{endDate}|"]
    """
    nasdaq_closing_days_files = sorted(os.listdir('./nasdaq_data'))

    # print("nasdaq_closing_days_files[-1]:",nasdaq_closing_days_files[-1])
    pd_day = pd.read_csv('./nasdaq_data/' + nasdaq_closing_days_files[-1]).sort_values(by=['Symbol'])
    companies = pd_day['Symbol']
    # print(companies)

    data = pd.DataFrame()
    # startDate = '2016-01-01'; endDate = '2020-01-25'
    i = 0
    for company in companies: #[615+1911+82+381+219:]: list of error companies
        print(company, i)
        i+=1
        # list of error companies:
        if company in ['CGO', 'CHW', 'RGP', 'SASR', 'TTEK', 'WNEB']: 
            continue

        yf_data = yf.download(company,startDate,endDate)
        data[company] = yf_data.Close
        


    # filename = f"historic_data_from|{startDate}|to|{endDate}|"
    joblib.dump(data.transpose(), "data/" + filename)
    # data_load = joblib.load(filename, mmap_mode='r')
    # print("data_load:",data_load, ":data_load")

startDate = '2006-01-01'; endDate = '2020-05-25'
fname = f"historic_nasdaq_data_from|{startDate}|to|{endDate}|"
make_data(startDate, endDate, filename = fname)