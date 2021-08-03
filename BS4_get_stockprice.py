import urllib, selenium
from urllib.request import Request, urlopen
from bs4 import BeautifulSoup
import requests
import webbrowser
import pandas as pd, numpy as np, joblib
import datetime
import time, yfinance as yf
import os, re 
from selenium import webdriver
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from selenium.common.exceptions import TimeoutException

# import dryscrape

def get_stock_val(company):
    """ Returns stock price of a company in normal trading hours. Afterhours trading has a different function.

    Args:
        company (str): Company you want the stockprice of

    Returns:
        float: stockprice value
    """
    text = f'{company} stock'
    text = urllib.parse.quote_plus(text)

    url = 'https://google.com/search?q=' + text

    response = requests.get(url)

    #with open('output.html', 'wb') as f:
    #    f.write(response.content)
    #webbrowser.open('output.html')

    soup = BeautifulSoup(response.text, 'lxml')
    for g in soup.find_all():
        # print(g.text)
        if 'aktiekurs' in g.text.lower():
            text_stock = g.text.lower().split('aktiekurs')
            first_comma = len(text_stock[1].split(',')[0])
            
            val = text_stock[1][:first_comma+3]
            val = float(val.replace(',','.'))

            break
        # print('-----')

    print(val) 
    return val
# get_stock_val('otlk')

def get_stock_val_afterHours(company='otlk'):
    
    text = f'{company} afterhours'
    text = urllib.parse.quote_plus(company)

    url = 'https://google.com/search?q=' + text
    # url = f'https://www.marketwatch.com/investing/stock/{company}'

    response = requests.get(url)    

    soup = BeautifulSoup(response.text, 'lxml')

    after_hours_val = ""
    
    f = soup.find_all('div', {'class':"BNeawe tAd8D AP7Wnd"})
    
    for i in f:
        if 'Efterbørs' in i.text:
            line = i.text.split('Efterbørs: ')

            after_hours_val = line[1].split(" ")[0] 

            print(line[1].split(" ")[0])

            break

    return float(after_hours_val.replace(',','.'))
            # break

    # print(val) 



def get_insider_stock_data():
    """saves the stocks of insider trading (https://finviz.com/insidertrading.ashx?tc=1) 
    into a pandas dataframe. Requires a dataframe to load from. Therefore 
    run these two lines only before first call to the function: 
    1. df = pandas.Dataframe(columns=['Ticker', 'Owner', 'Relationship', 'Date', 'Transaction', 'Cost', '#Shares', 'Value', '#Shares', 'Total', 'SEC Form 4', 'time_stamp')
    2. joblib.dump(df, 'data/insider_scraped_data')
    After that it appends scraped data to the dataframe after each run and removes duplicates. 
    """ 
    url = 'https://finviz.com/insidertrading.ashx?tc=1'


    req = Request(url, headers={'User-Agent':'Mozilla/5.0'})
    webpage = urlopen(req).read()
    #with open('output.html', 'wb') as f:
    #    f.write(response.content)
    #webbrowser.open('output.html')

    soup = BeautifulSoup(webpage, 'html.parser')
    point1 = 10*500

    html_code = soup.prettify()
    html_code = html_code.split('\n')

    print("len(html_code):", len(html_code))

    # for line in html_code:
    #     if 'TWO' in line:
    #         pass
    #         # print(line) 

    # print(soup.prettify()[point1:point1 + 10*5*200])
    # for container in soup.findAll('a','tab-link'):
    #     print("container:", container.text)
    #     break

    # test cron_job
    dir_path = os.path.dirname(os.path.realpath(__file__))+ "/data/"
    a = joblib.load(dir_path + 'cron_test_data')
    a.append(a[-1]+1)
    joblib.dump(a, dir_path+ 'cron_test_data')
    
    containers1 = soup.findAll('tr','insider-buy-row-1')
    containers2 = soup.findAll('tr','insider-buy-row-2')

    print("len(containers):", len(containers1))
    # df = pd.DataFrame({'Ticker': [], 'Owner': [], 'Relationship': [], 'Date': [], 'Transaction': [], 'Cost': [], '#Shares': [], 'Value': [], '#Shares Total': [],'SEC Form 4': [], 'time_stamp': []})
    
    df = joblib.load(dir_path + 'insider_scraped_data')

    # This might give problem in the very first call of the function
    # df['time_stamp'] = list(map(convert_time_stamp_to_secs, df['SEC Form 4'])) 

    col_name = df.columns

    for container in containers1+containers2:
        children = [child.text for child in container.contents]
        children = children + [convert_time_stamp_to_secs(children[-1])]
        
        ds = pd.DataFrame([children], columns=col_name)
        
        # print("ds")
        # print(ds)
        # ds['time_stamp'] = list(map(convert_time_stamp_to_secs,df['SEC Form 4']))

        df = pd.concat([df, ds], ignore_index=True)

    # print("len(df):", len(df))
    
    df = df.drop_duplicates(keep='last')
    df = df.sort_values(by=['time_stamp'])#, ignore_index=True)
    # print("len(df):", len(df))
    joblib.dump(df, dir_path + 'insider_scraped_data')

    print("df")
    print("len(df):", len(df))
    print(df[:30])

    print(a) # crontab test

    return df


def convert_time_stamp_to_secs(time_stamp):
    """Calculates the timestamp in seconds. Used to make the 'SEC Form 4' col in
    insider dataframe into a number you can sort by in a pandas dataframe:
    Fx: 
        convert_time_stamp_to_secs("Feb 12 05:40 PM")
        returns 1613104800.0
    
    Args:
        time_stamp (str): formatted like this "Feb 12 05:40 PM" 

    Returns:
        float: time_stamp in seconds
    """
    # time_stamp = "Feb 12 05:40 PM"
     
    now = datetime.datetime.now()
    year = now.strftime("%Y")

    time_stamp = year +" "+ time_stamp

    d = datetime.datetime.strptime(time_stamp, "%Y %b %d %I:%M %p")

    local_time = time.mktime(d.timetuple())

    return local_time

def get_news(company):
    """Gets the latest newsheadline for a company. Source: stocktwits.com

    Args:
        company (str): abbrevation of company name used on the stockexchange. Fx 'tsla' for Tesla.

    Returns:
        (tuple): (date, headline)
        date (str): date and time of the news article headline
        headline (str): news article headline
    """
    url = f'https://stocktwits.com/symbol/{company}'

    option = webdriver.ChromeOptions()
    option.add_argument('headless')

    driver = webdriver.Chrome('/home/lau/Lau/Code/python/Stockmarket/chromedriver/chromedriver', options=option)  # Optional argument, if not specified will search path.

    driver.set_window_position(0, 0)
    driver.set_window_size(2048, 2048)

    driver.get(url)

    button = driver.find_element(By.XPATH, '//button[text()="Accept All"]')
    button.click()
    
    headline = driver.find_element(By.XPATH, '//h4[contains(@class,"st_3laqg_N")]')
    print(headline.text)

    news_date = driver.find_element(By.XPATH, '//div[contains(@class,"st_2c4HJX4")]')
    
    print("date.text:", news_date.text)

    date = news_date.text.split("• ")[1]
    
    return date, headline.text

def save_news(all_companies_news):
    companies = 'tsla', 'aapl', 'otlk'

    for company in companies:
        date, headline = get_news(company)
    
        all_companies_news[company] = {date: headline}
        print("all_companies_news:", all_companies_news)
        
    

def selenium_webdriver_test():

    url = 'https://stocktwits.com/symbol/OTLK'

    option = webdriver.ChromeOptions()
    # option.add_argument('headless')

    driver = webdriver.Chrome('/home/lau/Lau/Code/python/Stockmarket/chromedriver/chromedriver', options=option)  # Optional argument, if not specified will search path.

    driver.set_window_position(0, 0)
    driver.set_window_size(2048, 2048)

    driver.get(url)

    button = driver.find_element(By.XPATH, '//button[text()="Accept All"]')
    button.click()
    # accept_bool = wait(driver,(By.XPATH, "//button[text()='I Accept']"), "Accept")
    # if accept_bool: button.click()
    # driver.refresh();

    h4_news = driver.find_element(By.XPATH, '//h4[contains(@class,"st_3laqg_N")]')
    print(h4_news.text)

    news_date = driver.find_element(By.XPATH, '//div[contains(@class,"st_2c4HJX4")]')
    print(news_date.text)

    button = driver.find_element(By.XPATH, '//h3[contains(@class,"st_BVAeM0C")]')
    # button = driver.find_element(By.XPATH, '//button[text()="Latest OTLK news"]')
    # st_BVAeM0C st_3jzPQQ_ st_3MXcQ5h st_CjvTpBY st_1jzr122 st_jGV698i st_PLa30pM st_cUBEAH8
    # button = wait(driver, (By.XPATH, "//*[ contains( text(), 'Latest’ )]"), 'Latest')
    print(button)
    button.click()
    button = driver.find_element(By.XPATH, '//div[contains(@class,"st_2QAa_wF")]')
    button.click()
    # button = driver.find_element(By.XPATH, '//i[contains(@class,"lib_3V-0nyQ")]')
    # button.click()
    
    span_elem = driver.find_element(By.XPATH, '//span[contains(@class,"st_1uctJiU")]')
    text = span_elem.text
    print('span date:', text)
    span_elem = driver.find_element(By.XPATH, '//span[contains(@class,"st_2P7TMkL")]')
    text = span_elem.text
    print(text)
# //body//div[@id='app']//div[@class='lightMode']//div[@class='st_1V-YOhj']
    if 0:
        j = 0
        while(j == 0):
            h3s = driver.find_elements_by_tag_name("h3") # Finds the right headlines
            print("len(h3s):", len(h3s))
            for i in h3s: # loop over h3s
                print('\n'+ i.text)
                if 'latest' in i.text.lower(): # if text in column press element 
                    print('|'+i.text+"|")
                    print('\n', i)
                    i.click()
                    j = 1
                    break
# Latest OTLK news\nMore News
    # h1s = driver.find_element_by_partial_link_text("Latest OTLK news") # Finds the right headlines

    button = wait(driver, (By.XPATH, "//*[ contains( text(), ‘Get Latest OTLK news’ )]", ))

    button.click()

    # source = driver.page_source
    # print("'more news' in source.lower():", 'more news' in source.lower()) 
    # for line in source.split('\n'):
         
    #      if 'more news' in line.lower():
    #          for ele in line.split('><'):
    #             if 'more news' in ele.lower():
    #                 print(ele)

    # spans = driver.find_elements_by_tag_name("span")
    # time.sleep(1)
    # divs = driver.find_elements_by_partial_link_text('Ticker Report') # Finds the right date
    # print(divs[0].text)
    # print(divs[1].text)
    # st_2QAa_wF st_1jzr122

    print("h1s:", h1s)
    h3 = wait(driver, "Latest")
    # h3.click()
    # for i in h3s:
    #     print('\n'+ i.text)
    #     if 'latest' in i.text.lower(): 
    #         print(i.text)
    #         print('\n', i)
    #         i.click()
    #         break
    # time.sleep(1)
    h1s = driver.find_element_by_xpath("Latest OTLK news") # Finds the right headlines
    for i in h1s:
        print('\n'+ i.text)
        string = "Outlook Therapeutics (NASDAQ:OTLK) Posts  Earnings Results, Misses Expectations By"
        if string.lower() in i.text.lower(): 
            print(i.text)
            print('\n', i)
            # i.click()
            break


    # time.sleep(5)
    driver.close()

def wait(driver, locator, text):
    delay = 20 # seconds
    try:
        # myElem = WebDriverWait(driver, delay).until(EC.presence_of_element_located((By.CLASS_NAME , 'st_2WMHmmI st_bPzIqx3 st_3PPn_WF')))
        myElem = WebDriverWait(driver, delay).until(EC.text_to_be_present_in_element(locator, text))
        
        print("Page is ready!")
    except TimeoutException:
        print("Loading took too much time!")
        myElem = 0
    print('myElem:', myElem)
    return(myElem)

def phantomjs_test():
    url = 'https://stocktwits.com/symbol/OTLK'
    
    options = webdriver.ChromeOptions();
    options.add_argument('headless')
    
    driver = webdriver.PhantomJS()
    
    driver.get(url)
    
    soupFromJokesCC = BeautifulSoup(driver.page_source) #page_source fetches page after rendering is complete

    soup_pretty = soupFromJokesCC.prettify()

    driver.save_screenshot('screen.png') # save a screenshot to disk

    driver.quit()

    print('more news' in soup_pretty)

def dryscrape_test():
    url = 'https://stocktwits.com/symbol/OTLK'
    
    session = dryscrape.Session()
    session.visit(url)
    
    response = session.body()
    soup = BeautifulSoup(response)
    elements = soup.find_all('span', {'class': 'st_2WMHmmI st_bPzIqx3 st_3PPn_WF'})
    print(elements)
    # Result:
    #<p id="intro-text">Yay! Supports javascript</p>

def change_in_sec_vs_buy_date(df_row):
    """
    Column for machinelearning. Makes a column of change in stock value between 
    end-of-day when insider bought stock and when the buy was filed with the SEC

    Args:
        df_row (Pandas series): Row (df.loc[n]) from pandas DataFrame that is saved in get_insider_stock_data()

    Returns:
        float: Pct change in price between sec filing and data of actual purchase
    """
    # print('df_row', df_row)
    start_date = '2021 ' + df_row['Date']
    start_date = datetime.datetime.strptime(start_date, "%Y %b %d")
    start_date = str(start_date).split(' ')[0]
        
    end_date = str(datetime.date.fromtimestamp(df_row['time_stamp']))
    
    # if the start date is from last year:
    if int(start_date[5:7].lstrip("0"))> int(end_date[5:7].lstrip("0")) or (int(start_date[5:7].lstrip("0"))== int(end_date[5:7].lstrip("0")) and int(start_date[8:10].lstrip("0")) > int(end_date[8:10].lstrip("0"))):
        start_date = "2020-" + start_date[5:] 

    print(start_date, end_date)

    if start_date == end_date:
        return 0

    company = df_row['Ticker']

    change = yf.download(company, start_date, end_date)
    return (change.Close[0]-change.Open[-1])/change.Close[0]


def make_insider_labels(df_row, deltadays):
    """[summary]

    Args:
        df_row (Pandas series): Row (df.loc[n]) from pandas DataFrame that is saved in get_insider_stock_data()

    Returns:
        float: Pct change in price between sec filing and delta (= 3 days ahead)
    """
    company = df_row['Ticker']

    start_date  = datetime.date.fromtimestamp(df_row['time_stamp'])
    delta       = datetime.timedelta(days = deltadays)
    end_date    = start_date + delta
    
    print(df_row)
    print("\nend_date:", end_date.strftime("%Y %b %d %a %I:%M %p"), '\n')
    if end_date.strftime("%a") in ["Sat", "Sun"]:
        delta       = datetime.timedelta(days = deltadays+2)
        end_date    = start_date + delta

    start_date = start_date.strftime("%Y-%m-%d")
    end_date   = end_date.strftime("%Y-%m-%d") 
    
    change = yf.download(company, start_date, end_date)
    print("type(change.Open)",change.Open)
    return (change.Close[0]-change.Open[-1])/change.Close[0]

if __name__=='__main__':
    # convert_time_stamp_to_secs()
    # get_insider_stock_data()
    # get_stock_val_afterHours()
    # get_stock_val('otkl')
    # get_news()
    # selenium_webdriver_test()
    save_news({})
    # phantomjs_test()
    # dryscrape_test()

