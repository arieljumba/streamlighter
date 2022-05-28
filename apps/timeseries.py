from contextlib import suppress
import datetime
from random import triangular
import streamlit as st
import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
import numpy as np
import statsmodels.api as sm
import statsmodels
import scipy
from scipy.stats import pearsonr
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
import seaborn as sns
from matplotlib.pyplot import GridSpec, title




company_names = pd.DataFrame({
    'display_name':[
        'Apple Inc', 'Bank of America Corporation','Amazon.com', 'AT&T Inc', 'Alphabet Inc','American Airlines', 'AstraZeneca PLC'
    ],
    'short_name':[
       'AAPL', 'BAC', 'AMZN', 'T', 'GOOG', 'AAL', 'AZN' 
    ]
    })


def app():

    hide_streamlit_style = """
                <style>
                #MainMenu {visibility: hidden;}
                footer {visibility: hidden;}
                </style>
                """
    st.markdown(hide_streamlit_style, unsafe_allow_html=True)
    
    def local_css(file_name):
        with open(file_name) as f:
            st.markdown(f'<style>{f.read()}</style>',unsafe_allow_html=True)
    local_css('style.css')
    
    st.header("TimeSeries Analysis (Stock Prices)")
    st.markdown(
        '''
        <html>
        <h5 style = "color:red";>
    INCOMPLETE!!!TO BE COMPLETED
    </h3>
    </HTML>
    ''',unsafe_allow_html=True)


#############################################################
    records = company_names.to_dict("records")
    stock_name = st.selectbox('', options=records,format_func=lambda records: f'{records["display_name"]}')

#############################################################

    @st.cache(suppress_st_warning=True)
    def stock_details(stock_):
        stock = yf.Ticker(stock_)
        stock_det = stock.get_info(start ='2021-01-01',end = '2022-05-05')
        return stock_det

    @st.cache(suppress_st_warning=True)
    def stock_price(stock_):
        stock = yf.Ticker(stock_)
        stock_price = stock.history(period = '10y')
        stock_price = stock_price.dropna(how='all')
        stock_price['Date'] = stock_price.index
        return stock_price

    

    x = stock_name['short_name']

    #########################################################

    col1,col2,col3,col4,col5,col6 = st.columns(6)

    try:
        col1.metric('Stock Price (USD)', stock_details(x)['currentPrice'], round(stock_details(x)['currentPrice']-stock_details(x)['previousClose'],2))
    except:
        col1.metric('Stock Price (USD)', "N/A")    
                
    try:
        col2.metric(label = "Market Cap (USD)", value =  "{:,}".format(round(stock_details(x)['marketCap']/1000000000,0))+" B")
    except:
        col2.metric("Market Cap (USD)","N/A")

    try:
        col3.metric(label = "Earnings Growth", value =round(stock_details(x)['earningsGrowth'],3))
    except:
        col3.metric("Earnings Growth","N/A")

    try:
        col4.metric(label = "Return on Assets", value =round(stock_details(x)['returnOnAssets'],3))
    except:
        col4.metric("Return on Assets","N/A")
        
    try:
        col5.metric(label = "Return on Equity", value =round(stock_details(x)['returnOnEquity'],3))
    except:
        col5.metric("Return on Equity", "N/A")
        
    try:
        col6.metric(label = "Debt To Equity", value =round(stock_details(x)['debtToEquity'],3))
    except:
        col6.metric("Debt To Equity","N/A")
    
    
    #########################################################   
    st.write(stock_details(x)['longBusinessSummary'])
    st.markdown('''---''')
  
    #########################################################  
    
  

    
    st.header('Current Price Movement')

    col1,col2 = st.columns(2)
    
    col1.line_chart(data= stock_price(stock_name['short_name'])[['Close','Open']], width=50, height=300, use_container_width=True)
    
    col2.bar_chart(data= stock_price(stock_name['short_name'])['Volume'], width=50, height=300, use_container_width=True)


    st.header('Let"s First Observe our Model without any seasonality, cyclical or Level adjustments')
    
    st.write('https://www.statsmodels.org/dev/generated/statsmodels.tsa.statespace.structural.UnobservedComponents.html')
 
    y = stock_price(stock_name['short_name'])['Close']
    train = y[0:int(y.shape[0]*0.7)]
    test = y[int((y.shape[0]*0.7)+1):int(y.shape[0])]

    model = {'level':'smooth trend',
        'cycle': False,
        'seasonal':None}  


    _mod = sm.tsa.UnobservedComponents(train,**model)
    _res = _mod.fit()

    fig = _res.plot_components(legend_loc='lower right',figsize=(15,9))
    st.write(fig)

    st.header('Predictions without any seasonality, cyclical or Level adjustments')

    col3,col4 = st.columns(2)
    num_steps = 20
    predict_res = _res.get_prediction(dynamic=train.shape[0] - num_steps)
    predict = predict_res.predicted_mean
    ci = predict_res.conf_int()

    fig2, axes = plt.subplots(figsize = (10,5))
    gs = GridSpec(nrows=1,ncols=1)
    ax1 = fig2.add_subplot(gs[0,0])
    ax1 = plt.plot(predict)
    col3.write(fig2)

    fig3, axes = plt.subplots(figsize = (10,5))
    gs = GridSpec(nrows=1,ncols=1)
    ax2 = fig3.add_subplot(gs[0,0])
    ax2.plot(train.index[-40:], train[-40:], 'k.', label='Observations');
    ax2.plot(train.index[-40:-num_steps], predict[-40:-num_steps], label='One-step-ahead Prediction');

    ax2.plot(train.index[-num_steps:], predict[-num_steps:], 'r', label='Multistep Prediction');
    #ax.plot(train.index[-num_steps:], ci.iloc[-num_steps:], 'k--');
    legend = ax2.legend(loc='upper left');
    col4.write(fig3)
 

 # smooth trend model without seasonal or cyclical components
    st.header('Let"s First Observe our Model seasonality and Level adjustments')
    seasonal_model = {
    'level': 'local linear trend',
    'seasonal': 12
    } 

    mod = sm.tsa.UnobservedComponents(train, **seasonal_model)
    res = mod.fit(method='powell', disp=False)


    fig4 = res.plot_components(legend_loc='lower right', figsize=(15, 9));
    st.write(fig4)

    