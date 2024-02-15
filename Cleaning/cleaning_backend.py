from nicegui import events, ui, app
from nicegui.events import ValueChangeEventArguments
import pandas as pd 
import Signatures.Path as p
import Signatures.RandSig as rs
import io
from threading import Thread
from io import StringIO
# import ProcessPoolFunction
from concurrent.futures import ProcessPoolExecutor  
from multiprocessing import Process, Manager
import plotly.graph_objects as go
import asyncio
from pandas import read_csv, DataFrame, concat, merge, to_datetime, Grouper, DateOffset, Timedelta
from numpy import unique, zeros, mean, std, ceil, concatenate, average
import numpy as np
from warnings import filterwarnings
import sys
from logging import getLogger, StreamHandler
import numba
import time

logger = getLogger(__name__)
logger.setLevel("DEBUG")

async def start_stream(log):
    """Start a 'stream' of console outputs."""
    # Create buffer
    string_io = StringIO()
    # Standard ouput like a print
    sys.stdout = string_io
    # Errors/Exceptions
    sys.stderr = string_io
    # Logmessages
    stream_handler = StreamHandler(string_io)
    stream_handler.setLevel("DEBUG")
    logger.addHandler(stream_handler)
    while 1:
        await asyncio.sleep(2)  # need to update ui
        # Push the log component and reset the buffer
        log.push(string_io.getvalue())
        string_io.truncate(0)

def handle_upload(e: events.UploadEventArguments):
    ui.notify(f'Uploaded {e.name}')
    global global_data
    global_data = pd.read_csv(io.BytesIO(e.content.read()), header=None, names=['Date1', 'A', 'Var3', 'Var4', 'B', 'Var6', 'C', 'D', 'E', 'F', 'G', 'Time', 'Midpoint'])

def download_pdf():
    name = global_data['A'][0]
    clean = f"{name}_clean.xlsx"
    time_bucket = f"{name}_time_bucket.xlsx"
    volume_bucket = f"{name}_trades_bucket.xlsx"
    summary = f"{name}_summary.xlsx"
    try:
        ui.download(clean)
    except:
        pass
    try:
        ui.download(time_bucket)
    except:
        pass
    try:
        ui.download(volume_bucket)
    except:
        pass
    try:
        ui.download(summary)
    except:
        pass


def button_callback(trim, K, Gamma, price_cleaning, volume_cleaning, start, end, running_sum_n, skip_lines, time_bucket, time_bucket_size, volume_bucket, volume_bucket_size, log):
    global global_data
    manager = Manager()
    progress = manager.Value('i', 0)  # Shared variable to track progress
    executor = ProcessPoolExecutor()
    future = executor.submit(clean_data, global_data, trim, K, Gamma, price_cleaning, volume_cleaning, start, end, running_sum_n, skip_lines, time_bucket,  time_bucket_size, volume_bucket, volume_bucket_size, progress)
    # Start a separate process to print the progress
    monitor_thread = Thread(target=print_progress, args=(progress,))  # Start monitoring thread
    monitor_thread.start()

def print_progress(progress):
    process_dict = {0: "Cleaning data",
                        1: "Inferring trade direction: 0%",
                        11: "Inferring trade direction: 20%",
                        12: "Inferring trade direction: 40%",
                        13: "Inferring trade direction: 60%",
                        14: "Inferring trade direction: 80%",
                        15: "Inferring trade direction: 100%",
                        2: "Cleaning data: Price",
                        3: "Cleaning data: Volume",
                        4: "Calculating liquidity measures: 0%",
                        41: "Calculating liquidity measures: 20%",
                        42: "Calculating liquidity measures: 40%",
                        43: "Calculating liquidity measures: 60%",
                        44: "Calculating liquidity measures: 80%",
                        45: "Calculating liquidity measures: 100%",
                        5: "Splitting data into buckets: Time",
                        6: "Splitting data into buckets: Volume",
                        7: "Cleaning finished", 
                        10: "There was no quote before the trade"}
    previous = 'none'
    while True:
        if process_dict[progress.value] != previous:
            print(f"Progress: {process_dict[progress.value]}")
            previous = process_dict[progress.value]
        time.sleep(2)  # Sleep for a while to reduce CPU usage
def clean_data(global_data, trim, K, Gamma, price_cleaning, volume_cleaning, start, end, running_sum_n, skip_lines, time_bucket, time_bucket_size, volume_bucket, volume_bucket_size, progress):
    filterwarnings("ignore")
    #================================
    # Importing and cleaning data   #
    #================================
    progress.value = 0
    Quote_Raw_Data = global_data
    Quote_Raw_Data = Quote_Raw_Data.iloc[skip_lines:]
    #Count rows of Quote_Raw_Data
    n = len(Quote_Raw_Data)
    #Add indicator variable to data
    a = list(range(1,n+1))
    Quote_Raw_Data["Ind"] = a
    #Extract only trades from data
    Quote_Raw_Data2 = Quote_Raw_Data[Quote_Raw_Data.C=="Trade"]
    #Find the time offset
    time_offset = Quote_Raw_Data2["Var6"].iloc[0]

    #Correct mistakes where one trade was saved has manytrades
    Quote_Raw_Data3 = Quote_Raw_Data2.groupby(["B","D"],as_index=False)["E"].sum()
    Quote_Raw_Data4 = Quote_Raw_Data2.groupby(["B","D"],as_index=False)[["A","C","F","G","Ind"]].min()
    Quote_Raw_Data5 = merge(Quote_Raw_Data3,Quote_Raw_Data4,on=["B","D"])

    #Extract Quotes from data
    Quote_Raw_Data6 = Quote_Raw_Data[Quote_Raw_Data.C=="Quote"]
    Quote_Raw_Data6 = Quote_Raw_Data6[["A","B","C","D","E","F","G","Ind"]]

    #Merge corrected trades and quotes
    Quote_Raw_Data7 = concat([Quote_Raw_Data5,Quote_Raw_Data6],ignore_index=True)
    Quote_Raw_Data7 = Quote_Raw_Data7.sort_values(by=['Ind'])
    Quote_Raw_Data7 = Quote_Raw_Data7.reset_index(drop=True)
    #
    def infer (Dataset_name,Name_col,Time_stamp_col,Type_col,Price_col,Volume_col,Bid_col,Ask_col, Ind, progress):
        #Rename important collumns
        Dataset_name.rename(columns = {Name_col: "Name"}, inplace = True)
        Dataset_name.rename(columns = {Time_stamp_col: "Date_Time"}, inplace = True)
        Dataset_name.rename(columns = {Type_col: "Type"}, inplace = True)
        Dataset_name.rename(columns = {Price_col: "Price"}, inplace = True)
        Dataset_name.rename(columns = {Volume_col: "Volume"}, inplace = True)
        Dataset_name.rename(columns = {Bid_col: "Bid"}, inplace = True)
        Dataset_name.rename(columns = {Ask_col: "Ask"}, inplace = True)
        Dataset_name.rename(columns = {Ind: "Ind"}, inplace = True)
#
        
        # find index of first trade
        first_trade = Dataset_name[Dataset_name['Type']=='Trade'].index[0]
        while first_trade == 0:
            Dataset_name = Dataset_name.iloc[1:]
            Dataset_name = Dataset_name.reset_index(drop=True)
            first_trade = Dataset_name[Dataset_name['Type']=='Trade'].index[0]
        #Make sure there is a trade midpoint before the first trade
        if Dataset_name["Ask"].isnull()[first_trade-1]:
            Dataset_name["Ask"][first_trade-1] = Dataset_name["Bid"][first_trade-1]
        Dataset_name["M"] = (Dataset_name["Bid"] + Dataset_name["Ask"])/2

        #Getting indexis of trades only with trade prices
        trades = Dataset_name[Dataset_name['Type']=='Trade']
        trades.dropna(subset=['Price', 'Volume'],inplace=True)
        trades = trades.loc[(trades['Volume'] != 0)]
        # Multiply each trade price by 100 if it is less than 5% of the max trade
        maxim = trades.Price.max()
        trades.Price = trades.Price.apply(lambda x: x*100 if x < 0.05*maxim else x)
        #Do the same in the dataset itself
        Dataset_name.Price = Dataset_name.Price.apply(lambda x: x*100 if x < 0.05*maxim else x)
        trades_ind = trades.index
        n3 = len(trades_ind)

        #Create a numpy array to use numba njit for loops
        np_trades_ind = trades_ind.to_numpy()
        np_median = Dataset_name.M.to_numpy()
        np_bid = Dataset_name.Bid.to_numpy()
        np_ask = Dataset_name.Ask.to_numpy()
        np_price = Dataset_name.Price.to_numpy()
        def infer_loop(np_trades_ind, np_median, np_bid, np_ask, np_price,  n3):
            # Loop to infer trade direction
            np_trade_direction = np.zeros(len(np_price))
            for i in range(n3):
                if int(100*i/n3) == 20:
                    progress.value = 11
                elif int(100*i/n3) == 40:
                    progress.value = 12
                elif int(100*i/n3) == 60:
                    progress.value = 13
                elif int(100*i/n3) == 80:
                    progress.value = 14
                elif int(100*i/n3) == 100:
                    progress.value = 15
                # Define Y a variable used to index throughout loop
                y = 0

                # Find closest spread midpoint
                while np.isnan(np_median[np_trades_ind[i]-y]):
                    y = y + 1
                    if np_trades_ind[i]-y < 0:
                        progress.value = 10
                        break

                # If trade price is above midpoint it is a buy
                np_median[np_trades_ind[i]] = np_median[np_trades_ind[i]-y]
                np_bid[np_trades_ind[i]] = np_bid[np_trades_ind[i]-y]
                np_ask[np_trades_ind[i]] = np_ask[np_trades_ind[i]-y]

                if np_price[np_trades_ind[i]] > np_median[np_trades_ind[i]]:
                    np_trade_direction[np_trades_ind[i]] = 1
                # If trade price is below midpoint it is a sell
                elif np_price[np_trades_ind[i]] < np_median[np_trades_ind[i]]:
                    np_trade_direction[np_trades_ind[i]] = -1
                # If trade price is equal to midpoint apply tick rule
                else:
                    ind = 1
                    # Find previous trade price that differs from this trade price
                    temp = np_price[np_trades_ind[i]]
                    while temp == np_price[np_trades_ind[i]-ind]:
                        ind = ind + 1
                        if np_trades_ind[i]-ind < 0:
                            break
                    # If trade price is higher than previous it is a buy
                    if temp > np_price[np_trades_ind[i]-ind]:
                        np_trade_direction[np_trades_ind[i]] = 1
                    # If trade price is lower than previous it is a sell
                    else:
                        np_trade_direction[np_trades_ind[i]] = -1
            # Extract only trades
            trade_indices = np.where(np_trade_direction != 0)[0]
            return np_trade_direction[trade_indices], np_median[trade_indices], np_bid[trade_indices], np_ask[trade_indices], np_price[trade_indices]

        #Define global dataset        
        np_trade_direction, np_median, np_bid, np_ask, np_price = infer_loop(np_trades_ind, np_median, np_bid, np_ask, np_price, n3)
        New = trades
        New['Trade_direction'] = np_trade_direction
        New['M'] = np_median
        New['Bid'] = np_bid
        New['Ask'] = np_ask
        New['Price'] = np_price
        inferred = New[["Name","Date_Time","Price","Volume","Trade_direction", "Bid", "Ask", "Ind", "M"]]
        return inferred.reset_index(drop=True)
    
    progress.value = 1
    inferred = infer(Quote_Raw_Data7,"A","B","C","D","E","F","G", "Ind", progress)
    progress.value = 8

    
#
    ##########################
    # C_Variationtion to clean data #
    ##########################

    def clean(Dataset_name,Name_col,Time_stamp_col,Price_col,Volume_col):
        progress.value = 3
        #Set data cleaning algorithm parameters
        
        #Rename important collumns
        Dataset_name.rename(columns = {Name_col: "Name"}, inplace = True)
        Dataset_name.rename(columns = {Time_stamp_col: "Time_Stamp"}, inplace = True)
        Dataset_name.rename(columns = {Price_col: "Price"}, inplace = True)
        Dataset_name.rename(columns = {Volume_col: "Volume"}, inplace = True)
        
        #Create new dataset for datacleaning purposes
        Newdat = Dataset_name
    
        #Convert timestamp to DateTime format and add offset
        Newdat["Time_Stamp"] = to_datetime(Newdat["Time_Stamp"]) + DateOffset(hours=int(time_offset))
        
        #Extract unique dates from dataset
        dates = Newdat["Time_Stamp"].dt.date
        global unique_dates
        unique_dates = unique(dates)
        
        #Count number of days in dataset
        n_dates = len(unique_dates)
        
        #Calculating amount of trades to be removed
        trim_amt = int(ceil(K*trim))

        price_data = Newdat[["Time_Stamp", "Price"]].to_numpy()

        #@numba.njit
        def price_cleaning_func(price_data, unique_dates, n_dates, K, trim_amt, Gamma):
            exc = np.array([])
            for i in range(n_dates):
                inter = price_data[np.datetime64(unique_dates[i]) == price_data[:, 0].astype('datetime64[D]')]
                n_day = len(inter)
                exc_ind = np.zeros(n_day)
                for j in range(n_day):
                    if j == 0:
                        y = price_data[j+1:j+K+1, 1]
                    elif j > 0 and j <= K:
                        y = np.concatenate((price_data[0:j, 1], price_data[j+1:K+1, 1]))
                    elif j > K and j < n_day - K:
                        y = np.concatenate((price_data[int(j-K/2):j, 1], price_data[j+1:int(j+K/2+1), 1]))
                    elif j == n_day - K:
                        y = price_data[j+1:n_day, 1]
                    elif j > n_day - K and j < n_day:
                        y = np.concatenate((price_data[n_day-K:j, 1], price_data[j+1:n_day, 1]))
                    elif j == n_day:
                        y = price_data[n_day-K:j, 1]
                    y = np.sort(y)
                    trim = y[trim_amt:len(y)-trim_amt]
                    trim_mu = np.mean(trim)
                    trim_sd = np.std(trim)
                    if abs(price_data[j, 1]-trim_mu) >= 3*trim_sd + Gamma:
                        exc_ind[j] = 1
                if i == 0:
                    exc = exc_ind
                else:
                    exc = np.concatenate((exc, exc_ind))
            price_clean_data = price_data[exc!=1]
            return price_clean_data

        if price_cleaning:
            progress.value = 2
            Price_clean_data = pd.DataFrame(price_cleaning_func(price_data, unique_dates, n_dates, K, trim_amt, Gamma), columns=["Time_Stamp", "Price"]) 
            Price_clean_data = pd.merge(Newdat, Price_clean_data, how = 'inner', on=["Time_Stamp", "Price"])
        else:
            Price_clean_data = Newdat

        
        #Extract time from dataset
        ans = Price_clean_data["Time_Stamp"].dt.time
        ansi  = zeros(len(ans))
        # vb
        #Checking whether trades are in trading interval add 10 minutes after 5pm
        for i in range(len(ans)):
            if (ans[i]>= to_datetime(start, format = "%H:%M").time()) &  (ans[i] <= (to_datetime(end, format = "%H:%M") + DateOffset(minutes=10)).time()):
                ansi[i]=1
            elif (ans[i]>= to_datetime(end, format = "%H:%M").time()):
                ansi[i]=2
        
        #Only extract trades in trading interval
        Price_clean_data['Trading_Time_Indicator'] = ansi

        price_clean_data = Price_clean_data[['Time_Stamp', 'Volume']].to_numpy()

        #@numba.njit
        def volume_cleaning_func(price_clean_data, unique_dates, n_dates, K, trim_amt, Gamma):
            exc = np.array([])
            for i in range(n_dates):
                inter = price_clean_data[np.datetime64(unique_dates[i]) == price_clean_data[:, 0].astype('datetime64[D]')]
                n_day = len(inter)
                exc_ind = np.zeros(n_day)
                for j in range(n_day):
                    if j == 0:
                        y = price_clean_data[j+1:j+K+1, 1]
                    elif j > 0 and j <= K:
                        y = np.concatenate((price_clean_data[0:j, 1], price_clean_data[j+1:K+1, 1]))
                    elif j > K and j < n_day - K:
                        y = np.concatenate((price_clean_data[int(j-K/2):j, 1], price_clean_data[j+1:int(j+K/2+1), 1]))
                    elif j == n_day - K:
                        y = price_clean_data[j+1:n_day, 1]
                    elif j > n_day - K and j < n_day:
                        y = np.concatenate((price_clean_data[n_day-K:j, 1], price_clean_data[j+1:n_day, 1]))
                    elif j == n_day:
                        y = price_clean_data[n_day-K:j, 1]
                    trim = np.delete(y, np.s_[0:trim_amt])
                    trim = np.delete(trim, np.s_[K-trim_amt:K-1])
                    trim_mu = np.mean(trim)
                    trim_sd = np.std(trim)
                    if abs(price_clean_data[j, 1]-trim_mu) >= 3*trim_sd + Gamma:
                        exc_ind[j] = 1
                if i == 0:
                    exc = exc_ind
                else:
                    exc = np.concatenate((exc, exc_ind))
            clean_data = price_clean_data[exc!=1]
            return clean_data
        
        if volume_cleaning:
            progress.value = 3
            Clean_Data = pd.DataFrame(volume_cleaning_func(price_clean_data, unique_dates, n_dates, K, trim_amt, Gamma), columns=["Time_Stamp", "Volume"])
            Clean_Data = pd.merge(Price_clean_data, Clean_Data, how = 'inner', on=["Time_Stamp", "Volume"])
        else:
            Clean_Data = Price_clean_data

        return Clean_Data

            
#
    Clean_Data = clean(inferred,"Name","Date_Time","Price","Volume")
    progress.value = 4
    #================================#
    # Calculating Liquidity Measures #
    #================================#
    # convert to numpy arrays
    price_np = Clean_Data["Price"].to_numpy(dtype=float)
    volume_np = Clean_Data["Volume"].to_numpy(dtype=float)
    bid_np = Clean_Data["Bid"].to_numpy(dtype=float)
    ask_np = Clean_Data["Ask"].to_numpy(dtype=float)
    trade_direction_np = Clean_Data["Trade_direction"].to_numpy(dtype=float)
    m_np = Clean_Data["M"].to_numpy(dtype=float)
    time_stamp_np = Clean_Data["Time_Stamp"].to_numpy()

    Clean_Data["PQS"] = np.log(ask_np)-np.log(bid_np)
    Clean_Data["PES"] = 2*trade_direction_np*(np.log(price_np)-np.log(m_np))

    
    
    n = len(Clean_Data)
    nt_price = np.zeros(n)
    last_index = 0

    for i in range(n):
        if int(100*i/n) == 20:
            progress.value = 41
        elif int(100*i/n) == 40:
            progress.value = 42
        elif int(100*i/n) == 60:
            progress.value = 43
        elif int(100*i/n) == 80:
            progress.value = 44
        elif int(100*i/n) == 100:
            progress.value = 45
        k = 1
        g = Clean_Data["Time_Stamp"][i]
        if i < n - 1100:
            if Clean_Data["Time_Stamp"][i+1000] - g < Timedelta(minutes=1):
                k = 1000
            elif Clean_Data["Time_Stamp"][i+500] - g < Timedelta(minutes=1):
                k = 500
            elif Clean_Data["Time_Stamp"][i+100] - g < Timedelta(minutes=1):
                k = 100
            elif Clean_Data["Time_Stamp"][i+50] - g < Timedelta(minutes=1):
                k = 50
        for j in range(max(i+k, last_index), n):
            if Clean_Data["Time_Stamp"][j] - g > Timedelta(minutes=1):
                nt_price[i] = Clean_Data["Price"][j]
                last_index = j
                break
    Clean_Data["PPI"] = 2*trade_direction_np*(np.log(nt_price) - np.log(m_np))
    Clean_Data["PRS"] = 2*trade_direction_np*(np.log(price_np) - np.log(nt_price))
    Clean_Data["LN_Price"] = np.log(price_np)
    Clean_Data["Turnover"] = price_np*volume_np
    #Set the first value of "Return" to 0
    Clean_Data["Return"] = 0
    #Calculate the return
    for i in range(1,len(Clean_Data)):
        Clean_Data["Return"][i] = (price_np[i] - price_np[i-1])/price_np[i-1]
    
    Clean_Data["Signed_Volume"] = trade_direction_np*volume_np


    Final_Clean_Data = Clean_Data[["Name", "Time_Stamp", "Price", "Volume", "M", "LN_Price", "Trade_direction", "PQS", "PES", "PPI", "PRS", "Trading_Time_Indicator", "Turnover", "Return", "Signed_Volume" ]]

    Final_Clean_Data.rename(columns = {'Time_Stamp':'Date_Time'}, inplace = True)
    Final_Clean_Data.rename(columns = {'M':'Spread_Midpoint'}, inplace = True)
    Final_Clean_Data.rename(columns = {'Trade_direction':'Trade_Sign'}, inplace = True)
    Final_Clean_Data.rename(columns = {'PQS':'Percent_Quoted_Spread'}, inplace = True)
    Final_Clean_Data.rename(columns = {'PES':'Percent_Effective_Spread'}, inplace = True)
    Final_Clean_Data.rename(columns = {'PPI':'Percent_Price_Impact'}, inplace = True)
    Final_Clean_Data.rename(columns = {'PRS':'Percent_Realised_Spread'}, inplace = True)
    Final_Clean_Data.Trading_Time_Indicator.replace({0:"Before", 1:"In", 2:"After"}, inplace=True)

    #update tbl1 with Final_Clean_Data head
    Final_Clean_Data["Date_Time"] = Final_Clean_Data["Date_Time"].dt.tz_localize(None)
    file_name = Final_Clean_Data['Name'][0]
    Final_Clean_Data.to_excel(f"{file_name}_clean.xlsx",index=True)
    
    ################################
    # Split data into buckets      #
    ################################
    progress.value += 1
    # 5 minute buckets
    if time_bucket:
        progress.value = 5
        def wm(x):
            weights=Final_Clean_Data.loc[x.index, "Volume"]
            if sum(weights) == 0:
                return None
            else:
                return average(x, weights=weights)
            
        def buy(x):
            #return the sum of the signed volume when it is positive
            return sum(x[x>0])
        
        def sell(x):
            #return the sum of the signed volume when it is negative
            return -sum(x[x<0])
        bucket_time = Final_Clean_Data.groupby(Grouper(key='Date_Time', freq=f'{time_bucket_size}Min')).agg({'Name':'first', 'Date_Time':'first', 'Price': ['max', 'min','first', 'last' ], 'Volume': 'sum', "Trading_Time_Indicator": 'first', "Turnover": 'sum', "Signed_Volume": [buy, sell, 'sum'], "Return": 'mean', "Percent_Quoted_Spread": wm,"Percent_Effective_Spread": wm, "Percent_Price_Impact": wm, "Percent_Realised_Spread": wm})
        bucket_time.columns = ['Name', 'Date', 'High', 'Low', 'Open', 'Close', 'Volume', 'Trading_Time_Indicator', 'Turnover', 'Buys','Sells', 'Volume_Imbalance', 'Average_Return', "Volume_Weighted_Percent_Quoted_Spread", "Volume_Weighted_Percent_Effective_Spread", "Volume_Weighted_Percent_Price_Impact", "Volume_Weighted_Percent_Realised_Spread"]
        Ht = np.log(bucket_time["High"]/bucket_time["Open"])
        St = np.log(bucket_time["Close"]/bucket_time["Open"])
        Lt = np.log(bucket_time["Low"]/bucket_time["Open"])
        n = int(running_sum_n)
        
        bucket_time["Volatility_1"] = 0
        bucket_time["Volatility_2"] = 0
        bucket_time["Volatility_3"] = 0
        a = 0.511
        b = 0.019
        c = 0.383
        for i in range(len(bucket_time)):
            
            try: 
                bucket_time["Volatility_1"][i] = 1/n*sum(St[i-n:i+1]**2)
                
            except:
                continue
        for i in range(len(bucket_time)):
            try: 
                bucket_time["Volatility_2"][i] = 1/n*sum(Ht[i-n:i+1]*(Ht[i-n:i+1]-St[i-n:i+1]) + Lt[i-n:i+1]*(Lt[i-n:i+1]-St[i-n:i+1]))
            except:
                continue
        for i in range(len(bucket_time)):
            try: 
                    bucket_time["Volatility_3"][i] = 1/n*sum(a*(Ht[i-n:i+1]-Lt[i-n:i+1])**2-b*(St[i-n:i+1]*(Ht[i-n:i+1]+Lt[i-n:i+1])-2*Ht[i-n:i+1]*Lt[i-n:i+1])-c*St[i-n:i+1]**2)
            except:
                continue
        

        #number the buckets
        bucket_time['Bucket_ID'] = range(1, len(bucket_time) + 1)

        def Standard_Deviation(x):
            return std(x)
        def Mean(x):
            return mean(x)
        def C_Variation(x):
            return std(x)/mean(x)

        bucket_time = bucket_time[bucket_time["Trading_Time_Indicator"] == "In"]
        bucket_summary = bucket_time.groupby(Grouper(key='Date', freq=f'{1}D')).agg({"High": [Mean, Standard_Deviation, C_Variation], "Low": [Mean, Standard_Deviation, C_Variation], "Open": [Mean, Standard_Deviation, C_Variation], "Close": [Mean, Standard_Deviation, C_Variation], "Volume": [Mean, Standard_Deviation, C_Variation],'Turnover':[Mean, Standard_Deviation, C_Variation], 'Volume_Imbalance':[Mean, Standard_Deviation, C_Variation], 'Average_Return':[Mean, Standard_Deviation, C_Variation], "Volume_Weighted_Percent_Quoted_Spread":[Mean, Standard_Deviation, C_Variation], "Volume_Weighted_Percent_Effective_Spread":[Mean, Standard_Deviation, C_Variation], "Volume_Weighted_Percent_Price_Impact":[Mean, Standard_Deviation, C_Variation], "Volume_Weighted_Percent_Realised_Spread":[Mean, Standard_Deviation, C_Variation],  "Volatility_1":[Mean, Standard_Deviation, C_Variation],"Volatility_2":[Mean, Standard_Deviation, C_Variation],"Volatility_3":[Mean, Standard_Deviation, C_Variation]})
        bucket_time.to_excel(f"{file_name}_time_bucket.xlsx",index=True)
        bucket_summary.to_excel(f"{file_name}_summary.xlsx",index=True)

    # group every 10 trades in a bucket
    if volume_bucket:
        progress.value = 6
        bucket_trades = Final_Clean_Data.groupby(Final_Clean_Data.index//volume_bucket_size).agg({'Name':'first', 'Date_Time':'first', 'Price': ['max', 'min','first', 'last' ], 'Volume': 'sum', "Trading_Time_Indicator": 'first', "Turnover": 'sum', "Signed_Volume": 'sum', "Return": 'mean', "Percent_Quoted_Spread": wm, "Percent_Effective_Spread": wm, "Percent_Price_Impact": wm, "Percent_Realised_Spread": wm})
        bucket_trades.columns = ['Name', 'Date', 'High', 'Low', 'Open', 'Close', 'Volume', 'Trading_Time_Indicator', 'Turnover', 'Volume_Imbalance', 'Average_Return', "Volume_Weighted_Percent_Quoted_Spread", "Volume_Weighted_Percent_Effective_Spread", "Volume_Weighted_Percent_Price_Impact", "Volume_Weighted_Percent_Realised_Spread"]
        Ht = np.log(bucket_trades["High"]/bucket_trades["Open"])
        St = np.log(bucket_trades["Close"]/bucket_trades["Open"])
        Lt = np.log(bucket_trades["Low"]/bucket_trades["Open"])

        bucket_trades["Volatility_1"] = 0
        bucket_trades["Volatility_2"] = 0
        bucket_trades["Volatility_3"] = 0
        for i in range(len(bucket_trades)):
            try: 
                bucket_trades["Volatility_1"][i] = 1/n*sum(St[i-n:i+1]**2)
            except:
                continue
        for i in range(len(bucket_trades)):
            try: 
                bucket_trades["Volatility_2"][i] = 1/n*sum(Ht[i-n:i+1]*(Ht[i-n:i+1]-St[i-n:i+1]) + Lt[i-n:i+1]*(Lt[i-n:i+1]-St[i-n:i+1]))
            except:
                continue
        for i in range(len(bucket_trades)):
            try: 
                    bucket_trades["Volatility_3"][i] = 1/n*sum(a*(Ht[i-n:i+1]-Lt[i-n:i+1])**2-b*(St[i-n:i+1]*(Ht[i-n:i+1]+Lt[i-n:i+1])-2*Ht[i-n:i+1]*Lt[i-n:i+1])-c*St[i-n:i+1]**2)
            except:
                continue
            
        bucket_trades['Bucket_ID'] = range(1, len(bucket_trades) + 1)
        bucket_trades.to_excel(f"{file_name}_trades_bucket.xlsx",index=True)
    
    
    progress.value = 7


