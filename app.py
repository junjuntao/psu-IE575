import streamlit as st
import pandas as pd
import requests
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import mplfinance as mpf
from mplfinance.original_flavor import plot_day_summary_ohlc
import time
import datetime
import numpy as np
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.io as pio
import pandas_ta as ta

plt.rcParams["figure.figsize"] = (12,8)

st.sidebar.header('Crypto Currency KPI Dashboard')
st.sidebar.markdown('Choose the KPIs you want to watch')
var = st.sidebar.checkbox('Volatility & Variance', key='1')
ohlc = st.sidebar.checkbox('OHLC', key='2')
owma = st.sidebar.checkbox('One WMA', key='3')
tsma = st.sidebar.checkbox('Two SMAs', key='4')
twma = st.sidebar.checkbox('Two WMAs', key='5')
smaema = st.sidebar.checkbox('SMA & EMA', key='6')
mas = st.sidebar.checkbox('SMA & EMA & WMA', key='7')
rsii = st.sidebar.checkbox('RSI', key='8')
macd = st.sidebar.checkbox('MACD', key='9')
bb = st.sidebar.checkbox('Bolinger Bands', key='0')
roc = st.sidebar.checkbox('ROC', key='a')
option = st.sidebar.multiselect(
     'Check the cryptos you want to watch',
     ['Bitcoin', 'Cardano', 'Ethereum'],
     ['Bitcoin'])
#st.header(option)


url = 'https://pro-api.coinmarketcap.com/v1/cryptocurrency/listings/latest'
headers = {'X-CMC_PRO_API_KEY' : '3ed38d7d-8f22-4aea-b435-a79915c60536',
            'Accepts': 'application/json'
}
params = {
  'start':'1',
  'limit':'9',
  'convert':'USD'
}

response = requests.get(url, headers=headers, params=params)

data = response.json()

############################################################################################################################################

st_date = st.sidebar.date_input('Start Date', datetime.date(2022, 1, 1))
end_date = st.sidebar.date_input('End Date')
if st_date>end_date:
    st.error('Error: The start date shold before the end date. Please choose again.')
# print(end_date - st_date)
# print(st_date)

import yfinance as yf
# Request historic pricing data via finance.yahoo.com API
df1 = yf.Ticker('BTC-USD').history(start=st_date,end = end_date)[['Low', 'Close', 'Open', 'High', 'Volume']]
df1['Symbol']='BTC'
df2 = yf.Ticker('ADA-USD').history(start=st_date,end = end_date)[['Low', 'Close', 'Open', 'High', 'Volume']]
df2['Symbol']='ADA'
df3 = yf.Ticker('ETH-USD').history(start=st_date,end = end_date)[['Low', 'Close', 'Open', 'High', 'Volume']]
df3['Symbol']='ETH'

def plot(df, name):
    plt.style.use('default')
    fig, ax = plt.subplots(figsize=(22, 11))
    axb = ax.twinx()

    # Same as above
    ax.set_xlabel('Date')
    ax.set_ylabel('Price ')
    ax.set_title(name)
    ax.grid(True)

    # Plotting on the first y-axis
    ax.plot(df.index.date, df.Close, color='green', label='Close', linewidth=3)
    ax.plot(df.index.date, df.Open, color='blue', label='Open', linewidth=3)
    ax.plot(df.index.date, df.Low, color='pink', label='Low', linewidth=3)
    ax.plot(df.index.date, df.High, color='orange', label='High', linewidth=3)

    ax.legend(loc='upper right')
    st.pyplot(fig)


if ('Bitcoin' in option) and st_date is not None and end_date is not None:
    st.header('Bitcoin')
    st.metric(label= data['data'][0]['symbol'],
                    value = round(data['data'][0]['quote']['USD']['price'],2),
                    delta = str(round(data['data'][0]['quote']['USD']['percent_change_24h'],3))+'%') 
    st.dataframe(df1)
    st.subheader('Check Missing Values')
    st.write(df2.isna().sum())
    st.subheader('Plot for Price')
    plot(df1, "Bitcoin")
if ('Cardano' in option) and st_date is not None and end_date is not None:
    st.header('Cardano')
    st.metric(label= data['data'][8]['symbol'],
                    value = round(data['data'][8]['quote']['USD']['price'],4),
                    delta = str(round(data['data'][8]['quote']['USD']['percent_change_24h'],3))+'%')
    st.dataframe(df2)
    st.subheader('Check missing values')
    st.write(df2.isna().sum())
    st.subheader('Plot for Price')
    plot(df2,"Cardano")
if ('Ethereum' in option) and st_date is not None and end_date is not None:
    st.header('Ethereum')
    st.metric(label= data['data'][1]['symbol'],
                    value = round(data['data'][1]['quote']['USD']['price'],2),
                    delta = str(round(data['data'][1]['quote']['USD']['percent_change_24h'],3))+'%')    
    st.dataframe(df3)
    st.subheader('Check missing values')
    st.write(df3.isna().sum())
    st.subheader('Plot for Price')
    plot(df3,"Etherium")



############################################################################################################################################

#Read the data sets
#df1 = pd.read_csv('Datasets-BTC-ADA-ETH/Bitcoin.csv', index_col='Date', parse_dates=True)
#df2 = pd.read_csv('Datasets-BTC-ADA-ETH/Cardano.csv', index_col='Date', parse_dates=True)
#df3 = pd.read_csv('Datasets-BTC-ADA-ETH/Ethereum.csv', index_col='Date', parse_dates=True)



#Self-defined functions
#Volatile and variance
def cal_volatile(df):
    global summary
    summary=pd.DataFrame()  
    mid=df[['Symbol', 'Close']].copy() #keep only the close price
    #From the results of data exploration, we know that there are no missing values in the dataset
    mid['Close']=mid['Close'].astype('float')
    mid['Returns']=mid.groupby('Symbol')['Close'].pct_change()
    mid.dropna(inplace=True)
    # get the mean returns
    summary = pd.DataFrame(mid.groupby('Symbol')['Returns'].mean())
    # get the standard deviation
    summary['Volatility'] = mid.groupby('Symbol')['Returns'].std()
    # get variance
    summary['Variance'] = summary['Volatility'] **2
    # get the number of observations
    summary['Observations'] = mid.groupby('Symbol')['Returns'].size()
    summary=pd.DataFrame(summary)

#OHLC Charts
#New version
names=["Bitcoin", "Cardano", "Ethereum"]
def demo1(data,i):
    #s = mpf.make_mpf_style(base_mpf_style='nightclouds')
    s = mpf.make_mpf_style(base_mpf_style='default')
    fig = mpf.figure(figsize=(22, 11), style=s)
    ax = fig.add_subplot(2,1,1)
    ax.set_title(names[i]+' - Candle Chart')
    ax.tick_params(labelbottom=False)
    av = fig.add_subplot(2,1,2, sharex=ax)
    mpf.plot(data[-100:], ax=ax, show_nontrading=True, type='candle', volume=av, mav=(3,6,9))
    plt.subplots_adjust(wspace=0, hspace=0)
    st.pyplot(fig)

#Old version
def demo2(data,i):
    # We need to exctract the OHLC prices into a list of lists:
    dvalues = data[['Open', 'High', 'Low', 'Close']].values.tolist()
    pdates = mdates.date2num(data.index)
    ohlc = [ [pdates[j]] + dvalues[j] for j in range(len(pdates)) ]
    plt.style.use('fivethirtyeight')
    #plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize = (22,11))
    plot_day_summary_ohlc(ax, ohlc[-90:], ticksize = 5)
    ax.set_xlabel('Date')
    ax.set_ylabel('Price ($)')
    ax.set_title(names[i]+' - Bar Chart')
    # Choosing to display the dates as "Month Day":
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    # This is to automatically arrange the date labels in a readable way:
    fig.autofmt_xdate()
    st.pyplot(fig)

#SMA & EMA
def demo3(data,i, n1, n2, n3, m):
   
    hsma = data['High'].rolling(n1).mean()
    lsma = data['Low'].rolling(n2).mean()
    ema = data['Close'].ewm(span=n3).mean()

    dvalues = data[['Open', 'High', 'Low', 'Close']].values.tolist()

    pdates = mdates.date2num(data.index)
    ohlc = [ [pdates[j]] + dvalues[j] for j in range(len(pdates)) ]
    plt.style.use('fivethirtyeight')
    #plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize = (24,12))
    plot_day_summary_ohlc(ax, ohlc[-m:], ticksize = 5, colorup='#77d879', colordown='#db3f3f')
    ax.plot(hsma[-m:], color = 'blue', linewidth = 2, label='High, '+str(n1)+'-Day SMA')
    ax.plot(lsma[-m:], color = 'blue', linewidth = 2, label='Low, '+str(n2)+'-Day SMA')
    ax.plot(ema[-m:], color = 'red', linestyle='--', linewidth = 2, label='Close, '+str(n3)+'-Day EMA')

    ax.set_xlabel('Date')
    ax.set_ylabel('Price ($)')
    ax.set_title(names[i]+' - Bar Chart with Moving Averages')
    ax.legend()
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y %b %d'))
    fig.autofmt_xdate()
    st.pyplot(fig)

#Two SMAs
def demo4(data, i, n1, n2, m):
    hsma = data['Close'].rolling(n1).mean()
    lsma = data['Close'].rolling(n2).mean()

    dvalues = data[['Open', 'High', 'Low', 'Close']].values.tolist()
    pdates = mdates.date2num(data.index)
    ohlc = [ [pdates[j]] + dvalues[j] for j in range(len(pdates)) ]
    plt.style.use('fivethirtyeight')
    #plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize = (24,12))
    plot_day_summary_ohlc(ax, ohlc[-m:], ticksize = 5, colorup='#77d879', colordown='#db3f3f')
    ax.plot(hsma[-m:], color = 'blue', linewidth = 2, label='Close, '+str(n1)+'-Day SMA')
    ax.plot(lsma[-m:], color = 'fuchsia', linewidth = 2, label='Close, '+str(n2)+'-Day SMA')
 
    ax.set_xlabel('Date')
    ax.set_ylabel('Price ($)')
    ax.set_title(names[i]+' - Bar Chart with Moving Averages')
    ax.legend()
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y %b'))
    fig.autofmt_xdate()
    st.pyplot(fig)

#WMA & SMA & EMA    
def demo5(data, i, n1, n2, n3, m):
    weights = np.arange(1,n1+1)
    wma = data['Close'].rolling(n1).apply(lambda prices: np.dot(prices, weights)/weights.sum(), raw=True)
    lsma = data['Close'].rolling(n2).mean()
    ema = data['Close'].ewm(span=n3).mean()
    
    dvalues = data[['Open', 'High', 'Low', 'Close']].values.tolist()

    pdates = mdates.date2num(data.index)
    ohlc = [ [pdates[j]] + dvalues[j] for j in range(len(pdates)) ]
    plt.style.use('fivethirtyeight')
    #plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize = (22,11))
    plot_day_summary_ohlc(ax, ohlc[-m:], ticksize = 5)
    ax.plot(wma[-m:], color = 'blue', linewidth = 2, label='Close, '+str(n1)+'-Day WMA')
    ax.plot(lsma[-m:], color = 'fuchsia', linewidth = 2, label='Close, '+str(n2)+'-Day SMA')
    ax.plot(ema[-m:], color = 'limegreen', linewidth = 2, label='Close, '+str(n3)+'-Day EMA')    
    ax.set_xlabel('Date')
    ax.set_ylabel('Price ($)')
    ax.set_title(names[i]+' - Bar Chart with Weighted Moving Averages')
    ax.legend()
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y %b %d'))
    fig.autofmt_xdate()
    st.pyplot(fig)

#Two WMAs    
def demo6(data, i, n1, n2, m):
    weights1 = np.arange(1,n1+1)
    weights2 = np.arange(1,n2+1)
    wma1 = data['Close'].rolling(n1).apply(lambda prices: np.dot(prices, weights1)/weights1.sum(), raw=True)
    wma2 = data['Close'].rolling(n2).apply(lambda prices: np.dot(prices, weights2)/weights2.sum(), raw=True)
   
    dvalues = data[['Open', 'High', 'Low', 'Close']].values.tolist()
    pdates = mdates.date2num(data.index)
    ohlc = [ [pdates[j]] + dvalues[j] for j in range(len(pdates)) ]
    plt.style.use('fivethirtyeight')
    #plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize = (24,12))
    plot_day_summary_ohlc(ax, ohlc[-m:], ticksize = 5, colorup='#77d879', colordown='#db3f3f')  
    ax.plot(wma1[-m:], color = 'orange', linewidth = 2, label='Close, '+str(n1)+'-Day WMA')
    ax.plot(wma2[-m:], color = 'fuchsia', linewidth = 2, label='Close, '+str(n2)+'-Day WMA')
    
    ax.set_xlabel('Date')
    ax.set_ylabel('Price ($)')
    ax.set_title(names[i]+' - Bar Chart with Weighted Moving Averages')
    ax.legend()
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y %b %d'))
    fig.autofmt_xdate()
    st.pyplot(fig)

#One WMA    
def demo7(data, i, n, m):
    weights = np.arange(1,n+1)
    wma = data['Close'].rolling(n).apply(lambda prices: np.dot(prices, weights)/weights.sum(), raw=True)
    
    dvalues = data[['Open', 'High', 'Low', 'Close']].values.tolist()

    pdates = mdates.date2num(data.index)
    ohlc = [ [pdates[j]] + dvalues[j] for j in range(len(pdates)) ]
    plt.style.use('fivethirtyeight')
    #plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize = (22,11))
    plot_day_summary_ohlc(ax, ohlc[-m:], ticksize = 5, colorup='#77d879', colordown='#db3f3f')
  
    ax.plot(wma[-m:], color = 'fuchsia', linewidth = 2, label='Close, '+str(n)+'-Day WMA')
    
    ax.set_xlabel('Date')
    ax.set_ylabel('Price ($)')
    ax.set_title(names[i]+' - Bar Chart with Weighted Moving Averages')
    ax.legend()
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y %b %d'))
    fig.autofmt_xdate()
    st.pyplot(fig)
    
#RSI    
def rsi(df, periods = 14, ema = True):
    close_delta = df['Close'].diff()
    # Make two series: one for lower closes and one for higher closes
    up = close_delta.clip(lower=0)
    down = -1 * close_delta.clip(upper=0)
    
    if ema == True:
	    # Use exponential moving average
        ma_up = up.ewm(com = periods - 1, adjust=True, min_periods = periods).mean()
        ma_down = down.ewm(com = periods - 1, adjust=True, min_periods = periods).mean()
    else:
        # Use simple moving average
        ma_up = up.rolling(window = periods, adjust=False).mean()
        ma_down = down.rolling(window = periods, adjust=False).mean()
        
    rsi = ma_up / ma_down
    rsi = 100 - (100/(1 + rsi))
    return rsi

#RSI Figure
def RSI_graph(df,i):   
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_width=[0.25, 0.75])
    # Create Candlestick chart for price data
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        increasing_line_color='#ff9900',
        decreasing_line_color='black',
        showlegend=False
    ))

    fig.add_trace(go.Scatter(
        x=df.index,
        y=df[rsi_names[i]],
        line=dict(color='#ff9900', width=2),
        showlegend=True,
    ), row=2, col=1
    )

    fig.update_yaxes(range=[-10, 110], row=2, col=1)
    fig.add_hline(y=0, col=1, row=2, line_color="#667", line_width=2)
    fig.add_hline(y=100, col=1, row=2, line_color="#666", line_width=2)
    fig.add_hline(y=30, col=1, row=2, line_color='#336699', line_width=2, line_dash='dash')
    fig.add_hline(y=70, col=1, row=2, line_color='#336699', line_width=2, line_dash='dash')

    layout = go.Layout(
        plot_bgcolor='#efefef',
        title = f'{names[i]} Relative Strength Index Chart ',
        # Font Families
        font_family='Monospace',
        font_color='orange',
        font_size=12,
        xaxis=dict(
            rangeslider=dict(
                visible=True
            )
        )
    )
    # update and display
    fig.update_layout(layout)
    fig.update_layout(title={'y':0.9, 'x':0.5,'xanchor': 'center', 'yanchor': 'top'})
    st.plotly_chart(fig, use_container_width=True)

#MACD
def MACD(df,i):
    df.ta.macd(close='close', fast=12, slow=26, signal=9, append=True)
    pd.set_option("display.max_columns", None)  # show all columns
    df.columns = [x.lower() for x in df.columns]
    # Construct a 2 x 1 Plotly figure
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True)
    # price Line
    fig.append_trace(
        go.Scatter(
            x=df.index,
            y=df['open'],
            line=dict(color='#ff9900', width=1),
            name='open',
            # showlegend=False,
            legendgroup='1',
        ), row=1, col=1
    )
    # Candlestick chart for pricing
    fig.append_trace(
        go.Candlestick(
            x=df.index,
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            increasing_line_color='#ff9900',
            decreasing_line_color='black',
            showlegend=False
        ), row=1, col=1
    )
    # Fast Signal (%k)
    fig.append_trace(
        go.Scatter(
            x=df.index,
            y=df['macd_12_26_9'],
            line=dict(color='#ff9900', width=2),
            name='macd',
            # showlegend=False,
            legendgroup='2',
        ), row=2, col=1
    )
    # Slow signal (%d)
    fig.append_trace(
        go.Scatter(
            x=df.index,
            y=df['macds_12_26_9'],
            line=dict(color='#000000', width=2),
            # showlegend=False,
            legendgroup='2',
            name='signal'
        ), row=2, col=1
    )
    
    # Colorize the histogram values
    colors = np.where(df['macdh_12_26_9'] < 0, '#000', '#ff9900')
    fig.append_trace(
        go.Bar(
            x=df.index,
            y=df['macdh_12_26_9'],
            name='histogram',
            marker_color=colors,
        ), row=2, col=1
    )
    # Make it pretty
    layout = go.Layout(
        plot_bgcolor='#efefef',
        title = f'{names[i]} Moving Average Convergence Divergence',  
        # Font Families
        font_family='Monospace',
        font_color='orange',
        font_size=12,
        xaxis=dict(
            rangeslider=dict(
                visible=True
            )
        )
    )
    # Update options and show plot
    fig.update_layout(layout)
    fig.update_layout(title={'y':0.9, 'x':0.5,'xanchor': 'center', 'yanchor': 'top'})
    st.plotly_chart(fig, use_container_width=True)

#ROC
def ROCs(df, n):  
    #Calculate difference in closing price from closing price n periods ago
    M = df["Close"].diff(n)  
    #N contans the value of closing price n periods ago
    N = df["Close"].shift(n)  
    #create ROC
    ROC = pd.Series(((M / N) * 100), name = 'ROC')   
    df = df.join(ROC)
    df.head()
    return df 

def roc_plot(df,i):
    n=2
    try:
        df = data[['Close']].copy()
    except:
        df = data["Close"]
    # Compute the n-period Rate of Change for Close column
    Rocs = ROCs(df,n)
    Rocs.head()
    ROC = Rocs['ROC']
    # Plotting the Price Series chart and the Ease Of Movement below
    fig = plt.figure(figsize=(15,12))
    ax = fig.add_subplot(2, 1, 1)
    ax.set_xticklabels([])
    plt.plot(df['Close'],lw=1)
    plt.title(f'{names[i]} Close Price and ROC Indicator')
    plt.ylabel('Close Price')
    plt.grid(True)
    bx = fig.add_subplot(2, 1, 2)
    plt.plot(ROC,'k',lw=0.75,linestyle='-',label='ROC')
    plt.legend(loc=2,prop={'size':9})
    plt.ylabel('ROC values')
    plt.grid(True)
    plt.setp(plt.gca().get_xticklabels(), rotation=30)



#BBands
def BB(data,i):
    window=20
    nstd =2
    try:
        df = data[['Close']].copy()
    except:
        df = data["Close"]

    #Calculating sma
    sma = df.rolling(window=20).mean().dropna()
    std = df.rolling(window = window).std().dropna()
    #Using folrmulas for upper and lower bolinger bands
    upper_band = sma + std * nstd
    lower_band = sma - std * nstd
    upper_band = upper_band.rename(columns={'Close': 'upper'})
    lower_band = lower_band.rename(columns={'Close': 'lower'})
    bb = df.join(upper_band).join(lower_band)
    bb = bb.dropna()
    #Setting the buy points and sell points
    #When the actual value of the curve surpasses the upper band - it is a sell indicator
    # If the actual value of the curve goes below the lower band - it is a buy indicator
    buyers = bb[bb['Close'] <= bb['lower']]
    sellers = bb[bb['Close'] >= bb['upper']]
  
    fig = go.Figure()
    #Plotting
    fig.add_trace(go.Scatter(x=lower_band.index, 
                            y=lower_band['lower'], 
                            name='Lower Band', 
                            line_color='#FECB52'
                            ))
    fig.add_trace(go.Scatter(x=upper_band.index, 
                            y=upper_band['upper'], 
                            name='Upper Band', 
                            line_color='#FFA500'
                            ))
    fig.add_trace(go.Scatter(x=df.index, 
                            y=df['Close'], 
                            name='Close', 
                            line_color='#636EFA'
                            ))
    fig.add_trace(go.Scatter(x=buyers.index, 
                            y=buyers['Close'], 
                            name='Buyers', 
                            mode='markers',
                            marker=dict(
                                color='#088F8F',
                                size=5,
                                )
                            ))
    fig.add_trace(go.Scatter(x=sellers.index, 
                            y=sellers['Close'], 
                            name='Sellers', 
                            mode='markers', 
                            marker=dict(
                                color='#EF553B',
                                size=5,
                                )
                            ))
    fig.update_layout(
        title={
            'text': f'{names[i]} Bollinnger Band Indicator',
            'y':0.9,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top'},
            xaxis = {'title':"Time"},
            yaxis = {'title':"Price"}
            )
    fig.show()     

if var:
    if ('Bitcoin' in option):
        st.header('Volatility & Variance -- Bitcoin')
               
        st.subheader('Overview of Historical Data')
        st.dataframe(df1.describe()[['High', 'Low', 'Open', 'Close', 'Volume']])
        cal_volatile(df=df1)
        st.subheader('Volatility & Variance')
        st.dataframe(summary)  
  
    if ('Cardano' in option):
        st.header('Volatility & Variance -- Cardano')
             
        st.subheader('Overview of Historical Data')
        st.dataframe(df2.describe()[['High', 'Low', 'Open', 'Close', 'Volume']])
        cal_volatile(df=df2)
        st.subheader('Volatility & Variance')
        st.dataframe(summary)
    
    if ('Ethereum' in option):
        st.header('Volatility & Variance -- Ethereum')
              
        st.subheader('Overview of Historical Data')
        st.dataframe(df3.describe()[['High', 'Low', 'Open', 'Close', 'Volume']])
        cal_volatile(df=df3)
        st.subheader('Volatility & Variance')
        st.dataframe(summary)
        
if ohlc:
    if ('Bitcoin' in option):
        st.header('OHLC Charts -- Bitcoin')
        st.subheader('Candle Chart')
        st.markdown("Use new version of mplfinance in Python")
        st.markdown("Pros: Predefined function for plotting OHLC")
        st.markdown("Cons: Does not support for moving averages of other types of prices other than close")
        demo1(df1, 0)
        st.subheader('Bar Chart')
        st.markdown("Use old version of mplfinance in Python")
        st.markdown("Pros: Randomly add user-defined moving average curves. For further visulizations of moving averages please select WMA/SMA/EMA in the sidebar selectbox")
        st.markdown("Cons: Different names when install and import the package")
        demo2(df1, 0)
        
    if ('Cardano' in option):
        st.header('OHLC Charts -- Cardano')
        st.subheader('Candle Chart')
        st.markdown("Use new version of mplfinance in Python")
        st.markdown("Pros: Predefined function for plotting OHLC")
        st.markdown("Cons: Does not support for moving averages of other types of prices other than close")
        demo1(df2, 1)
        st.subheader('Bar Chart')
        st.markdown("Use old version of mplfinance in Python")
        st.markdown("Pros: Randomly add user-defined moving average curves. For further visulizations of moving averages please select WMA/SMA/EMA in the sidebar selectbox")
        st.markdown("Cons: Different names when install and import the package")
        demo2(df2, 1)
        
    if ('Ethereum' in option):
        st.header('OHLC Charts -- Ethereum')
        st.subheader('Candle Chart')
        st.markdown("Use new version of mplfinance in Python")
        st.markdown("Pros: Predefined function for plotting OHLC")
        st.markdown("Cons: Does not support for moving averages of other types of prices other than close")
        demo1(df3, 2)
        st.subheader('Bar Chart')
        st.markdown("Use old version of mplfinance in Python")
        st.markdown("Pros: Randomly add user-defined moving average curves. For further visulizations of moving averages please select WMA/SMA/EMA in the sidebar selectbox")
        st.markdown("Cons: Different names when install and import the package")
        demo2(df3, 2)

if smaema:
    if ('Bitcoin' in option):
        st.header('SMA & EMA -- Bitcoin')
        st.markdown('Trading Rules: ')
        st.markdown('1. Enter long positions (buying) only when the price bars are completely above the higher SMA;')
        st.markdown('2. Enter short positions (selling) only when the price bars are completely below the lower SMA;')
        st.markdown('3. We do not enter any position (we keep flat on the market) when the prices are between the two SMAs, or the last bar is crossing any of them;')
        st.markdown('4. As long as the price remains above the EMA, the trader remains on the buy side; if the price falls below the EMA, the trader is a seller unless the price crosses to the upside of the EMA.')
        mday = st.number_input('Number of Days shown in plot: ',1, 1500, 100, key = "1")
        num1 = st.slider('Number of Day for SMA on High: ', 1, 500, 40, key='1')
        num2 = st.slider('Number of Day for SMA on Low: ', 1, 500, 40, key='2')
        num3 = st.slider('Number of Day for EMA: ', 1, 500, 20)
        demo3(data=df1,i=0, n1=num1, n2=num2, n3=num3, m=mday)
        
    if ('Cardano' in option):
        st.header('SMA & EMA -- Cardano')
        st.markdown('Trading Rules: ')
        st.markdown('1. Enter long positions (buying) only when the price bars are completely above the higher SMA;')
        st.markdown('2. Enter short positions (selling) only when the price bars are completely below the lower SMA;')
        st.markdown('3. We do not enter any position (we keep flat on the market) when the prices are between the two SMAs, or the last bar is crossing any of them;')
        st.markdown('4. As long as the price remains above the EMA, the trader remains on the buy side; if the price falls below the EMA, the trader is a seller unless the price crosses to the upside of the EMA.')
        mday = st.number_input('Number of Days shown in plot: ',1, 1500, 100, key = "2")
        num1 = st.slider('Number of Day for SMA on High: ', 1, 500, 40, key='3')
        num2 = st.slider('Number of Day for SMA on Low: ', 1, 500, 40, key='4')
        num3 = st.slider('Number of Day for EMA: ', 1, 500, 20)
        demo3(data=df2,i=1, n1=num1, n2=num2, n3=num3, m=mday)
        
    if ('Ethereum' in option):
        st.header('SMA & EMA -- Ethereum')
        st.markdown('Trading Rules: ')
        st.markdown('1. Enter long positions (buying) only when the price bars are completely above the higher SMA;')
        st.markdown('2. Enter short positions (selling) only when the price bars are completely below the lower SMA;')
        st.markdown('3. We do not enter any position (we keep flat on the market) when the prices are between the two SMAs, or the last bar is crossing any of them;')
        st.markdown('4. As long as the price remains above the EMA, the trader remains on the buy side; if the price falls below the EMA, the trader is a seller unless the price crosses to the upside of the EMA.')
        mday = st.number_input('Number of Days shown in plot: ',1, 1500, 100, key = "3")
        num1 = st.slider('Number of Day for SMA on High: ', 1, 500, 40, key='5')
        num2 = st.slider('Number of Day for SMA on Low: ', 1, 500, 40, key='6')
        num3 = st.slider('Number of Day for EMA: ', 1, 500, 20)
        demo3(data=df3,i=2, n1=num1, n2=num2, n3=num3, m=mday)
        
if tsma: 
    if ('Bitcoin' in option):
        st.header('Two SMAs -- Bitcoin')
        st.markdown('Trading Rules: Golden Cross and Death Cross')
        st.markdown('1. A buy signal is generated when a shorter-term moving average crosses above a longer-term moving average. This is called “The Golden Cross”;')
        st.markdown('2. A sell signal is generated when a short moving average crosses below a long moving average. This is called “The Death Cross”.')
        mday = st.number_input('Number of Days shown in plot: ',1, 1500, 300, key = "4")
        num1 = st.slider('Number of Day for Long-term SMA: ', 1, 500, 40, key='7')
        num2 = st.slider('Number of Day for Short-term SMA: ', 1, 500, 20, key='8')
        if num1 <= num2: 
            st.error("Error: Number of day for long-term SMA should be greater than short-term SMA, please choose again")
        demo4(data=df1, i=0, n1=num1, n2=num2, m=mday)
        
    if ('Cardano' in option):
        st.header('Two SMAs -- Cardano')
        st.markdown('Trading Rules: Golden Cross and Death Cross')
        st.markdown('1. A buy signal is generated when a shorter-term moving average crosses above a longer-term moving average. This is called “The Golden Cross”;')
        st.markdown('2. A sell signal is generated when a short moving average crosses below a long moving average. This is called “The Death Cross”.')
        mday = st.number_input('Number of Days shown in plot: ',1, 1500, 300, key = "5")
        num1 = st.slider('Number of Day for Long-term SMA: ', 1, 500, 40, key='9')
        num2 = st.slider('Number of Day for Short-term SMA: ', 1, 500, 20, key='10')
        if num1 <= num2: 
            st.error("Error: Number of day for long-term SMA should be greater than short-term SMA, please choose again")
        demo4(data=df2, i=1, n1=num1, n2=num2, m=mday)
        
    if ('Ethereum' in option):
        st.header('Two SMAs -- Ethereum')
        st.markdown('Trading Rules: Golden Cross and Death Cross')
        st.markdown('1. A buy signal is generated when a shorter-term moving average crosses above a longer-term moving average. This is called “The Golden Cross”;')
        st.markdown('2. A sell signal is generated when a short moving average crosses below a long moving average. This is called “The Death Cross”.')
        mday = st.number_input('Number of Days shown in plot: ',1, 1500, 300, key = "6")
        num1 = st.slider('Number of Day for Long-term SMA: ', 1, 500, 40, key='11')
        num2 = st.slider('Number of Day for Short-term SMA: ', 1, 500, 20, key='12')
        if num1 <= num2: 
            st.error("Error: Number of day for long-term SMA should be greater than short-term SMA, please choose again")
        demo4(data=df3, i=2, n1=num1, n2=num2, m=mday)
        
if mas:
    if ('Bitcoin' in option):
        st.header('SMA & EMA & WMA -- Bitcoin')
        st.markdown('WMA and EMA are quick to identify reversals than SMA.')
        mday = st.number_input('Number of Days shown in plot: ',1, 1500, 100, key = "7")
        num1 = st.slider('Number of Day for SMA: ', 1, 500, 40, key='13')
        num2 = st.slider('Number of Day for EMA: ', 1, 500, 40, key='14')
        num3 = st.slider('Number of Day for WMA: ', 1, 500, 40, key='15')
        demo5(data=df1,i=0, n1=num1, n2=num2, n3=num3, m=mday)
    
    if ('Cardano' in option):
        st.header('SMA & EMA & WMA -- Cardano')
        st.markdown('WMA and EMA are quick to identify reversals than SMA.')
        mday = st.number_input('Number of Days shown in plot: ',1, 1500, 100, key = "8")      
        num1 = st.slider('Number of Day for SMA: ', 1, 500, 40, key='16')
        num2 = st.slider('Number of Day for EMA: ', 1, 500, 40, key='17')
        num3 = st.slider('Number of Day for WMA: ', 1, 500, 40, key='18')
        demo5(data=df2,i=1, n1=num1, n2=num2, n3=num3, m=mday)
    
    if ('Ethereum' in option):
        st.header('SMA & EMA & WMA -- Ethereum')
        st.markdown('WMA and EMA are quick to identify reversals than SMA.')
        mday = st.number_input('Number of Days shown in plot: ',1, 1500, 100, key = "9")
        num1 = st.slider('Number of Day for SMA: ', 1, 500, 40, key='19')
        num2 = st.slider('Number of Day for EMA: ', 1, 500, 40, key='20')
        num3 = st.slider('Number of Day for WMA: ', 1, 500, 40, key='21')
        demo5(data=df3,i=2, n1=num1, n2=num2, n3=num3, m=mday)
        
if twma:
    if ('Bitcoin' in option):
        st.header('Two WMAs -- Bitcoin')
        st.markdown('Trading Rules: ')
        st.markdown('WMA in finding reversals -- The crossover of short-term WMA and long-term WMA indicates the reversals;')
        mm = st.number_input('Number of Days shown in plot: ',1, 1500, 100, key = "10")
        num1 = st.slider('Number of Day for Long-term WMA: ', 1, 500, 40, key='22')
        num2 = st.slider('Number of Day for Short-term WMA: ', 1, 500, 20, key='23')
        if num1 <= num2: 
            st.error("Error: Number of day for long-term WMA should be greater than short-term WMA, please choose again")
        demo6(data=df1, i=0, n1=num1, n2=num2, m=mm)
        
    if ('Cardano' in option):
        st.header('Two WMAs -- Cardano')
        st.markdown('Trading Rules: ')
        st.markdown('WMA in finding reversals -- The crossover of short-term WMA and long-term WMA indicates the reversals;')
        mm = st.number_input('Number of Days shown in plot: ',1, 1500, 100, key = "11")
        num1 = st.slider('Number of Day for Long-term WMA: ', 1, 500, 40, key='24')
        num2 = st.slider('Number of Day for Short-term WMA: ', 1, 500, 20, key='25')
        if num1 <= num2: 
            st.error("Error: Number of day for long-term WMA should be greater than short-term WMA, please choose again")
        demo6(data=df2, i=1, n1=num1, n2=num2, m=mm)
        
    if ('Ethereum' in option):
        st.header('Two WMAs -- Ethereum')
        st.markdown('Trading Rules: ')
        st.markdown('WMA in finding reversals -- The crossover of short-term WMA and long-term WMA indicates the reversals;')
        mm = st.number_input('Number of Days shown in plot: ',1, 1500, 100, key = "12")
        num1 = st.slider('Number of Day for Long-term WMA: ', 1, 500, 40, key='26')
        num2 = st.slider('Number of Day for Short-term WMA: ', 1, 500, 20, key='27')
        if num1 <= num2: 
            st.error("Error: Number of day for long-term WMA should be greater than short-term WMA, please choose again")
        demo6(data=df3, i=2, n1=num1, n2=num2, m=mm)
        
if owma:
    if ('Bitcoin' in option):
        st.header('One WMA -- Bitcoin')
        st.markdown('Trading Rules: ')
        st.markdown('WMA in trend following -- First, ensure that the asset is moving in a certain trend. Next, select the most appropriate period to use. Finally, if it is in a bullish trend, keep holding the trade so long as the price is above the moving average. Similarly, if it is a bearish trend, keep holding the trade as long as the price is below the MAs.')
        mm = st.number_input('Number of Days shown in plot: ',1, 1500, 100, key = "13")
        num = st.slider('Number of Day for WMA: ', 1, 500, 40, key='28')
        demo7(data=df1, i=0, n=num, m=mm)
    
    if ('Cardano' in option):
        st.header('One WMA -- Cardano')
        st.markdown('Trading Rules: ')
        st.markdown('WMA in trend following -- First, ensure that the asset is moving in a certain trend. Next, select the most appropriate period to use. Finally, if it is in a bullish trend, keep holding the trade so long as the price is above the moving average. Similarly, if it is a bearish trend, keep holding the trade as long as the price is below the MAs.')
        mm = st.number_input('Number of Days shown in plot: ',1, 1500, 100, key = "14")
        num = st.slider('Number of Day for WMA: ', 1, 500, 40, key='29')
        demo7(data=df2, i=1, n=num, m=mm)
    
    if ('Ethereum' in option):
        st.header('One WMA -- Ethereum')
        st.markdown('Trading Rules: ')
        st.markdown('WMA in trend following -- First, ensure that the asset is moving in a certain trend. Next, select the most appropriate period to use. Finally, if it is in a bullish trend, keep holding the trade so long as the price is above the moving average. Similarly, if it is a bearish trend, keep holding the trade as long as the price is below the MAs.')
        mm = st.number_input('Number of Days shown in plot: ',1, 1500, 100, key = "15")
        num = st.slider('Number of Day for WMA: ', 1, 500, 40, key='30')
        demo7(data=df3, i=2, n=num, m=mm)         

if rsii:
    df1['bitcoin_rsi'] = rsi(df1)
    df2['cardano_rsi'] = rsi(df2)
    df3['ethereum_rsi'] = rsi(df3)
    rsi_names = ['bitcoin_rsi','cardano_rsi','ethereum_rsi']        
    if ('Bitcoin' in option):
        st.header('RSI Graph -- Bitcoin')
        st.markdown('Trading Rules: ')
        st.markdown('The RSI produces a number between 0 and 100, with readings over a certain threshold (usually 70) indicating an overbought market (time to sell!) and values below that level (usually 30) indicating an oversold market (time to buy!).') 
        RSI_graph(df1,i=0)
    
    if ('Cardano' in option):
        st.header('RSI Graph -- Cardano')
        st.markdown('Trading Rules: ')
        st.markdown('The RSI produces a number between 0 and 100, with readings over a certain threshold (usually 70) indicating an overbought market (time to sell!) and values below that level (usually 30) indicating an oversold market (time to buy!).') 
        RSI_graph(df2,i=1)
    
    if ('Ethereum' in option):
        st.header('RSI Graph -- Ethereum')
        st.markdown('Trading Rules: ')
        st.markdown('The RSI produces a number between 0 and 100, with readings over a certain threshold (usually 70) indicating an overbought market (time to sell!) and values below that level (usually 30) indicating an oversold market (time to buy!).') 
        RSI_graph(df3,i=2)
        
if macd: 
    if ('Bitcoin' in option):
        st.header('MACD -- Bitcoin')
        st.markdown('Trading Rules: ')
        st.markdown('Our MACD line is yellow, our Signal line is black. When the difference between the MACD and Signal lines is positive (uptrend), the histogram, which is presented as a bar chart underneath the MACD and Signal lines, shows yellow, and when the value is negative, it shows black (downtrend.) ')
        MACD(df1, 0)
    
    if ('Cardano' in option):
        st.header('MACD -- Cardano')
        st.markdown('Trading Rules: ')
        st.markdown('Our MACD line is yellow, our Signal line is black. When the difference between the MACD and Signal lines is positive (uptrend), the histogram, which is presented as a bar chart underneath the MACD and Signal lines, shows yellow, and when the value is negative, it shows black (downtrend.) ')
        MACD(df2, 1)
    
    if ('Ethereum' in option):
        st.header('MACD -- Ethereum')
        st.markdown('Trading Rules: ')
        st.markdown('Our MACD line is yellow, our Signal line is black. When the difference between the MACD and Signal lines is positive (uptrend), the histogram, which is presented as a bar chart underneath the MACD and Signal lines, shows yellow, and when the value is negative, it shows black (downtrend.) ')
        MACD(df3, 2)  

if bb: 
    if ('Bitcoin' in option):
        st.header('BB -- Bitcoin')
        st.markdown('Trading Rules: ')
        st.markdown('The Bolinger bands are between the Orange (UpperBand) and Yellow (LowerBand), our Signal line is purple. The idicator suggests optimal time to sell (red) and buy (green) the cryptocurrrecny.')
        BB(df1, 0)
    
    if ('Cardano' in option):
        st.header('BB -- Cardano')
        st.markdown('Trading Rules: ')
        st.markdown('The Bolinger bands are between the Orange (UpperBand) and Yellow (LowerBand), our Signal line is purple. The idicator suggests optimal time to sell (red) and buy (green) the cryptocurrrecny.')
        BB(df2, 1)
    
    if ('Ethereum' in option):
        st.header('BB -- Ethereum')
        st.markdown('Trading Rules: ')
        st.markdown('The Bolinger bands are between the Orange (UpperBand) and Yellow (LowerBand), our Signal line is purple. The idicator suggests optimal time to sell (red) and buy (green) the cryptocurrrecny. ')
        BB(df3, 2)  

if roc: 
    if ('Bitcoin' in option):
        st.header('Rate of Change -- Bitcoin')
        st.markdown('Trading Rules: ')
        st.markdown('The ROC shows volatility in the dataset')
        roc_plot(df1, 0)
    
    if ('Cardano' in option):
        st.header('Rate of Change -- Cardano')
        st.markdown('Trading Rules: ')
        st.markdown('The ROC shows volatility in the dataset')
        roc_plot(df2, 1)
    
    if ('Ethereum' in option):
        st.header('Rate of Change -- Ethereum')
        st.markdown('Trading Rules: ')
        st.markdown('The ROC shows volatility inthe dataset')
        roc_plot(df3, 2)   