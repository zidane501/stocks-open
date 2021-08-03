# Import the plotting library
# import helper_functions
import matplotlib.pyplot as plt, numpy as np
import joblib
import pandas as pd
from scipy import stats
from helper_functions import get_local_minima_and_maxima
import lightgbm as lgb
# Import the yfinance. If you get module not found error the run !pip install yfiannce from your Jupyter notebook
import yfinance as yf  

from datetime import datetime
# datetime object containing current date and time
now = datetime.now()
# dd/mm/YY H:M:S
dt_string = now.strftime("%Y-%m-%d-%H-%M-%S")

dat = pd.read_csv('/home/lau/Lau/Code/python/Stockmarket/nasdaq_data/NASDAQ_20200910.csv')
print(dat.keys())
biz = dat['Symbol']
#%%% 
print('hi')
#%%
# biz = [#"AKRX",
# "GILD",
# "SHIP",
# "AAPL",
# "MSFT",
# # "TOPS",
# "MRNA",
# "CSCO",
# "CMCSA",
# "INTC",
# "BBBY",
# "BCRX",
# "AXAS",
# "SIRI",
# "HBAN",
# "EBAY",
# "SBUX",
# "NVDA",
# "AGNC",
# "NYMT",
# "NFLX",
# "JBLU",
# "PLUG",
# "ATVI",
# "FITB",
# "ATHX",
# "QCOM",
# "AMAT",
# "MBRX",
# "PENN",
# "AMZN",
# "NVAX",
# "WYNN",
# "SMRT",
# "PTEN",
# "CREE",
# "MYL" ,
# "ETFC",
# "MXIM",
# "FCEL",
# "GPRO",]


# guess local minimas:
def guessing_minimas(pandasData, accountHolding):
    """ Guesses the local minimas (buying points). Guesses the selling points as first negative movement after minima (sellingpoint).
    calculates moneymade if you bought one share at the minimas and sold at the sellingpoint.
    Guesses_max gives the deceleration of the curve after 3 points of deceleration (sugestion for sellingpoint, but isnt used for anything)
    pandasData should be numpy array or pandas dataseries or python list, 1 dim list
    """
    guesses_max  = []
    guesses_min  = []
    sellingpoint = []
    buy  = False
    sell = False
    wait_days = 0
    price_diff         = 0
    price_dif_data     = [(0,0)]
    price_diff_pct     = 0
    price_dif_data_pct = [(0,0)]
    buyingprice  = 0
    sellingprice = 0
    nbuys        = 0
    buyingmoney  = 0
    sellingmoney = 0
    accountDevellopment = [(0,accountHolding)] # Devellopment of account update after sales
    p1_data,p2_data,p3_data,p4_data,p5_data,p6_data,p7_data = [],[],[],[],[],[],[]
    error       = 0 # Check wheter to stop trading for a while (if the latest n was errors (n not determined yet))
    quarantine  = 0 # used with error
    slopeData      = []
    slope5_data    = []
    slope30_data   = []
    decc_coeff              = []
    curve_descend_bool_data = []
    d4curve_data            = []
    correct_guesses         = []
    correct_guess_freq      = []
    # labels = np.array(pandasData[:]); labels = (labels[:-1]-labels[1:])>0
    labels = get_local_minima_and_maxima(pandasData)["minimas"]>0
    # print("get_local_minima_and_maxima(pandasData)['minimas']:",get_local_minima_and_maxima(pandasData)["minimas"])
    # lgb_model = lgb.Booster(model_file='model.txt') 
    for i in range(len(pandasData)):
        # For the first four indexes, where it is out of range

        if i < 7:
            # The first 7 numbers are and should be useless, because they will be negative indices, meaning they are chsen from the back
            p7,p6,p5,p4,p3,p2,p1,p0 = pandasData[i-7], pandasData[i-6], pandasData[i-5], pandasData[i-4], pandasData[i-3], pandasData[i-2], pandasData[i-1], pandasData[i]
            p1_data.append(0);p2_data.append(0);p3_data.append(0);p4_data.append(0);p5_data.append(0);p6_data.append(0);p7_data.append(0)
        else: 
            p7,p6,p5,p4,p3,p2,p1,p0 = pandasData[i-7], pandasData[i-6], pandasData[i-5], pandasData[i-4], pandasData[i-3], pandasData[i-2], pandasData[i-1], pandasData[i]
            p1_data.append(p1);p2_data.append(p2);p3_data.append(p3);p4_data.append(p4);p5_data.append(p5);p6_data.append(p6);p7_data.append(p7)
        
        # if buy==True: wait_days+=1
        
        # if quarantine:
        #     quarantine -= 1
        #     continue

        dval   = pandasData[i-1]-pandasData[i]

        # print("p3, np.arange(5), pandasData[i-6:i-1]:",p3, np.arange(5), pandasData[i-3:i-1].to_list())
        slope5, intercept5, r_value5, p_value5, std_err5  = stats.linregress(np.arange(5), np.array(pandasData[i-6:i-1].to_list())) if i > 6 else 0,0,0,0,0
        slope5 = slope5.slope if i > 6 else 0
        slope30, intercept30, r_value30, p_value30, std_err30 = stats.linregress(np.arange(30), pandasData[i-31:i-1].to_list()) if i > 30 else 0,0,0,0,0
        slope30 = slope30.slope if i > 30 else 0
        # if i > 6: print("dir(slope5) line 97",slope5.slope)# dir(slope5))
        slopeData.append((dval, slope5, slope30))
        slope5_data.append(slope5)
        slope30_data.append(slope30)

        v0=p0-p1
        v1=p1-p2
        acc = v0-v1 

        if buy == True: #and wait_days==1: # and p0-p1<0:
            # print("(p7-p0)/7 > 0.25::",(p7-p0)/7)
            sellingprice = p0
            sellingpoint.append(sellingprice)

            price_change_pct = (sellingprice-buyingprice)*100/buyingprice
            price_diff_pct += price_change_pct 
            # print("price_diff_pct:", price_change_pct)
            if price_change_pct > 0:
                correct_guesses.append(1)
                np_corrguesses = np.array(correct_guesses)
                correct_guess_freq.append(len(np_corrguesses[np_corrguesses>0])/len(correct_guesses))
            else:
                correct_guesses.append(-1)
                np_corrguesses = np.array(correct_guesses)
                correct_guess_freq.append(len(np_corrguesses[np_corrguesses>0])/len(correct_guesses))

            # Stop selling n days bc many mistakes
            if (sellingprice-buyingprice)*100/buyingprice<-4.0 or buyingprice < 20.0:
                error +=1
                if error > 2:
                    quarantine = 5

            # print("(sellingprice-buyingprice)/buyingprice*100:",(sellingprice-buyingprice)*100/buyingprice)
            # price_diff += sellingprice-buyingprice
            # price_dif_data.append((i, price_diff))
            # price_dif_data_pct.append((i, price_diff_pct))

            sellingmoney = nbuys*sellingprice
            # print("nbuys", nbuys, ", buyingprice:", buyingprice)
            # print("nbuys", nbuys, ", sellingprice:", sellingprice)
            # print("sellingmoney", sellingmoney)
            # print("buyingmoney", buyingmoney)
            
            # print("accountHolding", accountHolding)
            n_corr = 1
            if (len(correct_guesses)>2) and ((np.sum(correct_guesses[-4:-1])) > 0.0): # Dont include the latest correct_guess, it is for this buy and should ofcourse not affect wehther to buy or not.

                accountHolding += sellingmoney - buyingmoney
                accountDevellopment.append((i, accountHolding))
            # print("accountHolding after update", accountHolding)
            buy = False
            wait_days = 0
            # print("")
            # print("buyingprice:", buyingprice)
            # print("sellingprice:", sellingprice)
            # print("money:", money)

        else: sellingpoint.append(0)
        
       


        # j = 0
        # latest = [p7,p6, p5, p4, p3, p2, p1]
        # latest_pct = [(latest[i]-latest[i+1])/latest[i] for i in np.arange(7)[:6]]
        # for i in np.arange(7)[:6]:
        #     if latest[i]-latest[i+1] < 0:
        #         j +=1

        ####################################
        ##################################
         # deceleration_coefficient_max
        deceleration_coefficient_max = (p2/p1)/(p1/p0) if p2-p1 < 0 and (p1-p0) < 0 else 1
        if deceleration_coefficient_max < 1:
            guesses_max.append(p0) if i > 3 else guesses_max.append(None)
        else: guesses_max.append(0)

        # deceleration_coefficient_min
        decc_coef0_bool               = p3-p2 > 0 and (p2-p1) > 0
        deceleration_coefficient_min0 = (p2/p1)/(p3/p2)
        decc_coef_bool                = p2-p1 > 0 and (p1-p0) > 0
        deceleration_coefficient_min  = (p1/p0)/(p2/p1)
        decc_coeff.append(deceleration_coefficient_min)
        
        curve_descend_bool = 1 if p3-p0>0 else 0; curve_descend_bool_data.append(curve_descend_bool)
        d4curve = p3-p0; d4curve_data.append(d4curve) # Difference btw this and the 4th datapoint
        
        if (deceleration_coefficient_min < 1 or deceleration_coefficient_min0 < 1) and curve_descend_bool and decc_coef_bool and decc_coef0_bool:
            guesses_min.append(p0) if i > 4 else guesses_min.append(0) 
            # if p0<20.0:
                # break

            buy = True
            
            buyingprice = p0
            nbuys = np.floor((accountHolding/3.8)/buyingprice)
            buyingmoney = (nbuys)*buyingprice

        else: guesses_min.append(0)
        ################################
        ####################################
    
    return {"guesses_min": guesses_min, "guesses_max": guesses_max, "sellingpoint": sellingpoint, 
            "price_dif_data": price_dif_data, "price_diff":price_diff, "accountHolding": accountHolding,
            "accountDevellopment": accountDevellopment, "price_dif_data_pct": price_dif_data_pct,
            "slopeData": slopeData, "slope5_data": slope5_data, "slope30_data": slope30_data, "decc_coeff": decc_coeff,
            "d4curve_data": d4curve_data, "curve_descend_bool_data": curve_descend_bool_data, "decc_coef0_bool": decc_coef0_bool, "decc_coef_bool": decc_coef0_bool,
            "p1_data":p1_data,"p2_data":p2_data,"p3_data":p3_data,"p4_data":p4_data,"p5_data":p5_data,"p6_data":p6_data,"p7_data":p7_data,
            "labels": labels, "correct_guess_freq": correct_guess_freq, "correct_guesses": correct_guesses,  
            }
            # Slopedata is linear regression of the latest numbers
# Get the data of the stock AAPL
total_price_diff = 0
accountmoney = 10000
company_data = {}
compName = 'NVAX' # USed to only run for that company name and plotting

print('debug1')
j = -1
correct_freq = pd.DataFrame()

good_company_list = [] # company where the download doesn't fail
save = True

if save:
    for company in biz[:500]:
        # if company != compName: continue
        # if company=="TOPS": continue
        
        print("\n",company)
        startDate = '2014-01-01'; endDate = '2020-04-25'
        
        data = yf.download(company,startDate,endDate)
        if np.array(data.index.to_list()).any() and data.Close[-1]>10: # pandas to list to np.array  
            
            j+= 1
            print('j+= 1 =',j)
            good_company_list.append(company)
        
            print("len(data.Close):",len(data.Close))
            # if company=="AAPL": print(data)
            
            company_data[company] =  guessing_minimas(data.Close, accountmoney)
            company_data[company]["dataClose"] = data.Close
            
            accountmoney = company_data[company]["accountHolding"]

            if np.array(company_data[company]['correct_guess_freq']).any():
                print(f"\ncompany_data[{company}]['correct_guesses']:", company_data[company]['correct_guesses'])
                print(f"\ncompany_data[{company}]['correct_guess_freq'][-1]:", company_data[company]['correct_guess_freq'][-1])
            # correct_freq[company] = company_data[company]['correct_guess_freq'][-1]
            
            # Slope Analysis (linear regression)
            slopeData = company_data[company]["slopeData"]
            dval_data, slope5_data, slope30_data = np.array([i[0] for i in slopeData]),np.array([i[1] for i in slopeData]), np.array([i[2] for i in slopeData])
            # print("dval_data",dval_data)
            # print("slope5_data",slope5_data)
            # print("slope30_data",slope30_data)
            # print("np.sum(np.logical_and(dval_data>0, slope30_data>0)): ", np.sum(np.logical_and(dval_data>0, slope30_data>0)))
            # print("np.sum(np.logical_and(dval_data>0, slope5_data>0))/len(dval_data): ", np.sum(np.logical_and(dval_data>0, slope5_data<0))/len(dval_data))
            # print("np.sum(np.logical_and(dval_data>0, slope30_data>0))/len(dval_data): ", np.sum(np.logical_and(dval_data>0, slope30_data<0))/len(dval_data))
            print("\nnp.sum(np.logical_and(dval_data>0, slope30_data>0))/len(dval_data): ", np.sum(np.logical_and(dval_data[30:]>0, slope30_data[30:]>0))/float(len(dval_data[30:])))
            print("np.sum(np.logical_and(dval_data>0, slope5_data>0))/len(dval_data): ", np.sum(np.logical_and(dval_data[5:]>0, slope5_data[5:]>0))/float(len(dval_data[5:])))

            print("\naccountmoney:", accountmoney)
            account_diff = accountmoney - company_data[good_company_list[j-1]]["accountHolding"] if j>1 else None #and good_company_list[j-1] != "TOPS" else None
            print("Money made by this company:",account_diff)
            total_price_diff += company_data[company]["price_diff"] 

            print (company, "price_diff for all sales:", company_data[company]["price_diff"])

    print("total_price_diff for all companies:", total_price_diff)

    print("\ncorrect_freq:", correct_freq)
    ##### Write data to file
    
    # filename = f"closingDataDict|{startDate}|{endDate}|02-05-2020"
    filename = f"data/closingDataDict1_{dt_string}"
    with open(filename, 'w'):
        joblib.dump(company_data, filename)
else: 
    company_data = joblib.load('data/' + f"closingDataDict1")
#####
ntrades = 0
for comp in company_data:
    ntrades += len(company_data[comp]["price_dif_data"])

print("ntrades:", ntrades) 
# print ("Trades pr day: ", ntrades*1.0/len(data.Close))

if 0:

    compData = company_data[compName]
    closeData, guesses_min, guesses_max, sellingpoint, price_dif_data, price_diff, accountDevellopment, price_dif_data_pct = compData["dataClose"], compData["guesses_min"], compData["guesses_max"], compData["sellingpoint"], compData["price_dif_data"], compData["price_diff"],compData["accountDevellopment"], compData["price_dif_data_pct"]
    slopeData = compData["slopeData"]

    print("len(data):", len(data))

    dData = np.array([0.0 if i == 0 else closeData[i]-closeData[i-1] for i in range(len(data))])
    negData = np.array([0.0 if i == 0 or closeData[i]-closeData[i+1]< 0 else closeData[i] for i in range(len(closeData[:-1]))])
    posData = np.array([0.0 if i == 0 or closeData[i]-closeData[i+1]> 0 else closeData[i] for i in range(len(closeData[:-1]))])

    local_minimas = get_local_minima_and_maxima(closeData)["minimas"]

    x_price_dif_data = [i[0] for i in price_dif_data]
    y_price_dif_data = [i[1] for i in price_dif_data]

    x_price_dif_data_pct = [i[0] for i in price_dif_data_pct]
    y_price_dif_data_pct = [i[1] for i in price_dif_data_pct]

    x_accDev, y_accDev = np.array([x for x,y in accountDevellopment]), np.array([y for x,y in accountDevellopment])
    y_accDev = (y_accDev-y_accDev[0])/100.0

    dval_data, slope5_data, slope30_data = np.array([i[0] for i in slopeData]),np.array([i[1] for i in slopeData]), np.array([i[2] for i in slopeData])
    # print("guesses_min:",guesses_min, "guesses_min")
    # print("np.sum(np.logical_and(dval_data>0, slope30_data>0))/len(dval_data): ", np.sum(np.logical_and(dval_data>0, slope30_data>0))/len(dval_data))
    # print("np.sum(np.logical_and(dval_data>0, slope5_data>0))/len(dval_data): ", np.sum(np.logical_and(dval_data>0, slope5_data>0))/len(dval_data))

    # Plot the close price of the AAPL
    # data.Close.plot()

    fig, ax = plt.subplots()
    ax.plot(range(len(closeData[:])),np.array(closeData[:]), "-", label=compName)

    # ax.scatter(range(len(negData)), negData, marker=".", c="red", label = "negData")
    # ax.scatter(range(len(posData)), posData, marker=".", c="cyan", label = "posData")
    # ax.scatter(range(len(guesses_min)), guesses_min, marker=".", c="black", label = "buy")
    # print("local_minimas:", local_minimas)
    # ax.scatter(np.arange(len(local_minimas)), local_minimas, marker=".", c="pink")
    # ax.scatter(range(len(local_maximas)), local_maximas, marker="o", c="red")
    # ax.scatter(range(len(guesses_max)), guesses_max, marker="o", c="blue")
    ax.scatter(range(len(guesses_min)), guesses_min, marker="x", c="red", label = "buyingpoint")
    ax.scatter(range(len(sellingpoint)), sellingpoint, marker=".", c="blue", label = "sellingpoint")
    ax.plot(x_price_dif_data, y_price_dif_data,"-x", label = "Sell-buy price dev")
    ax.plot(x_price_dif_data_pct, y_price_dif_data_pct,"-x", label = "Sell-buy price dev in pct")
    ax.plot(x_accDev, y_accDev, "-x",label = "accDev")

    # ax.scatter(range(len(data)), , marker="o")
    ax.legend()
    plt.show()