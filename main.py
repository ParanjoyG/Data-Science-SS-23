import streamlit as st
import yfinance as yf
import pandas as pd
import altair as alt
from datetime import date
import datetime as dt

import plotly.graph_objs as go

from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By

from datetime import date, datetime, timedelta
import numpy as np
import pandas as pd
import pandas_datareader as web
import plotly.graph_objects as go
import tensorflow as tf
from keras.models import Sequential

from transformers import AutoTokenizer, AutoModelForSequenceClassification

yf.pdr_override()


from newsapi import NewsApiClient

tokenizer = AutoTokenizer.from_pretrained("tarnformnet/Stock-Sentiment-Bert")
model = AutoModelForSequenceClassification.from_pretrained("tarnformnet/Stock-Sentiment-Bert", from_tf=True)

st.title("StockBoard")


user_input = st.text_input("Stock", "MSFT")

brk = yf.Ticker(user_input)

x = brk.info["shortName"]
start = st.date_input("Date", date.today())
st.header(x)

county = st.slider('Sample Size', 1, 50, 25)

url = 'https://news.google.com/search?q='+ x

newsapi = NewsApiClient(api_key='81945b20aa7547bd8a0066b30c989955')

driver = webdriver.Chrome()
driver.get(url)
driver.find_element(By.XPATH, '/html[1]/body[1]/c-wiz[1]/div[1]/div[1]/div[1]/div[2]/div[1]/div[3]/div[1]/div[1]/form[2]/div[1]/div[1]/button[1]').send_keys(Keys.ENTER)
driver.implicitly_wait(20)
texts = driver.find_elements(By.TAG_NAME, 'h3')
init = [i.text for i in texts]
driver.close()

count = len(init)
neg_confidences = []
neu_confidences = []
pos_confidences = []

df = pd.read_csv('Final List.csv')
names = df['Final Names'].tolist()
symbols = df['Final Symbols'].tolist()
categories = ['A-B','C-F','G-K', 'L-P', 'Q-T', 'U-W', 'X-Z']


def get_confidence_values(articles):

    for article in articles :

        inputs = tokenizer(article, return_tensors="pt")
        prob = model(**inputs).logits.detach().numpy()
        neg_confidences.append(prob[0][0])
        pos_confidences.append(prob[0][1])
    

get_confidence_values(init)

def reduce_date(x,y) :
    date = x
    date = datetime(int(date[0:4]), int(date[5:7]), int(date[8:]))
    final_date = date - timedelta(days = y)
    final_date = str(final_date)
    index = final_date.index(' ')
    start = final_date[0:index] 
    return start

def plot_stock_price(df, start):
    fig = moving_averages(df, start)
    fig.update_layout(height = 500, width=800, xaxis_rangeslider_visible=False)
    return fig

def moving_averages (df, start) :
    date_50 = reduce_date(start,50)
    date_100 = reduce_date(start,100)
    date_200 = reduce_date(start,200)
    df_MA50 = df[date_50:].Close.rolling(50).mean()
    df_MA100 = df[date_100:].Close.rolling(100).mean()
    df_MA200 = df[date_200:].Close.rolling(200).mean()
    df_cs = df[start:]
    fig = go.Figure()

    fig.add_candlestick(x=df_cs['Date'],
                    open=df_cs['Open'], high=df_cs['High'],
                    low=df_cs['Low'], close=df_cs['Close'], visible = True, name = "Candlestick")
    fig.add_trace(go.Scatter(x=df[date_50:].Date, y=df_MA50,visible = True, line=dict(color='black', width=1), name ="50 MA"))
    fig.add_trace(go.Scatter(x=df[date_100:].Date, y=df_MA100,visible = False, line=dict(color='black', width=1), name ="100 MA"))
    fig.add_trace(go.Scatter(x=df[date_200:].Date, y=df_MA200,visible = False, line=dict(color='black', width=1), name ="200 MA"))

    fig.update_layout(
    updatemenus=[
        dict(
            type="buttons",
            direction="right",
            active=0,
            x=0.57,
            y=1.2,
            buttons=list([
                dict(label="50 days",
                     method="update",
                     args=[{"visible": [True, True, False,False]},
                            ]),
                dict(label="100 days",
                     method="update",
                     args=[{"visible": [True, False, True, False]},
                           ]),
                dict(label="200 days",
                     method="update",
                     args=[{"visible": [True, False, False, True]},
                            ]),
            ]),
        )
    ]
    )
    return fig

def get_data (name, start_date, end_date) :
    date = start_date
    date = datetime(int(date[0:4]), int(date[5:7]), int(date[8:]))
    final_date = date - timedelta(days = 200)
    final_date = str(final_date)
    index = final_date.index(' ')
    start = final_date[0:index] 
    df = web.data.get_data_yahoo(name, start_date, end_date)
    df = df.reset_index()

    start = pd.to_datetime(start)
    end = pd.to_datetime(end_date)

    start_row = 0
    end_row = 0

    for i in range (0, len(df)) :
        if start <= pd.to_datetime(df['Date'][i]) :
            start_row = i
            break

    for j in range (0, len(df)) :
        if end >= pd.to_datetime(df['Date'][len(df)-1-j]) :
            end_row = len(df)-1-j
            break
    
    df = df.set_index(pd.DatetimeIndex(df['Date'].values))

    return df.iloc[start_row : end_row+1, :]

start, end, name = "2020-01-02", date.today().strftime(format ="%Y-%m-%d"), user_input
df = get_data(name, start, end) 

st.header(x + " Candlestick Plot\n")
p_stock = plot_stock_price(df, start)
st.plotly_chart(p_stock, use_container_width=True, config = dict({'scrollZoom': True}))

def model_evaluation(symbol) :
    end_date = str(date.today())
    start_date = end_date[0:2]+str(int(end_date[2:4])-3)+end_date[4:]

    symbol = symbol
    #df = web.DataReader(symbol, 'yahoo', start_date, end_date)
    df = web.data.get_data_yahoo(symbol, start_date, end_date)

    data = df.filter(['Close'])

    #Convert the dataframe to a numpy array
    dataset = data.values


    training_data_len = int(np.ceil( len(dataset) * .8 ))

    #Scale the data
    from sklearn.preprocessing import MinMaxScaler

    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_data = scaler.fit_transform(dataset)

    #Create the training data set
    #Create the scaled training data set
    train_data = scaled_data[0:int(training_data_len), :]
    #Split the data into x_train and y_train data sets
    x_train = []
    y_train = []

    for i in range(60, len(train_data)):
        x_train.append(train_data[i-60:i, 0])
        y_train.append(train_data[i, 0])
        if i<= 61:
            continue
            # print(x_train)
            # print(y_train)
            # print()

    # Convert the x_train and y_train to numpy arrays 
    x_train, y_train = np.array(x_train), np.array(y_train)


    #Reshape the data
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))


    #Build the LSTM model
    model = Sequential()

    model.add(tf.keras.layers.LSTM(50, return_sequences=True, input_shape= (x_train.shape[1], 1)))
    model.add(tf.keras.layers.LSTM(50, return_sequences= False))
    model.add(tf.keras.layers.Dense(25))
    model.add(tf.keras.layers.Dense(1))


    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')


    #Train the model
    model.fit(x_train, y_train, batch_size=30, epochs=50)

    #Create the testing data set
    #Create a new array containing scaled values from index 1543 to 2002 
    test_data = scaled_data[training_data_len - 60: , :]
    #Create the data sets x_test and y_test
    x_test = []
    y_test = dataset[training_data_len:, :]
    for i in range(60, len(test_data)):
        x_test.append(test_data[i-60:i, 0])
        
    # Convert the data to a numpy array
    x_test = np.array(x_test)

    # Reshape the data
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1 ))

    # Get the models predicted price values 
    predictions = model.predict(x_test)
    predictions = scaler.inverse_transform(predictions)

    # Get the root mean squared error (RMSE)
    rmse = np.sqrt(np.mean(((predictions - y_test) ** 2)))

    # Plot the data
    train = data[:training_data_len]
    valid = data[training_data_len:]
    valid['Predictions'] = predictions
    # Visualize the data
    train = pd.DataFrame(train)
    valid = pd.DataFrame(valid)
    trace_1 = go.Scatter (
        x = train.index,
        y = train['Close'],
        name = 'Training',
        line = dict (color = "#FF0000")
        )
    trace_2 = go.Scatter (
        x = valid.index,
        y = valid['Close'],
        name = 'Value',
        line = dict (color = "#00FF00")
    )
    trace_3 = go.Scatter (
        x = valid.index,
        y = valid['Predictions'],
        name = 'Predictions',
        line = dict (color = "#0000FF")
    )
    data = [trace_1, trace_2, trace_3]
    fig = go.Figure(data=data)
    return fig


st.header(x + " Predictions\n")
pred = model_evaluation(user_input)
st.plotly_chart(pred, use_container_width=True, config = dict({'scrollZoom': True}))

def normalise(x, u, l, U, L) :
  R = U-L
  r = u-l
  X = L + ((R/r)*(x-l))
  return X

st.header("Bear/Bull Market Indicator")
pos_mean = normalise(np.sum(pos_confidences), -8, 8, 0, 1)
neg_mean = normalise(np.sum(neg_confidences), -8, 8, 0, 1)
fig = go.Figure(go.Indicator(
    mode = "gauge+number+delta",
    value = (pos_mean/(pos_mean+neg_mean))*100, 
    domain = {'x': [0, 1], 'y': [0, 1]},
    gauge = {
        'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
        'bar': {'color': "#cdafe3",'thickness':0.15},
        'bgcolor': "white",
        'borderwidth': 2,
        'bordercolor': "gray",
        'steps': [
            {'range': [0, 20], 'color': 'firebrick'},
            {'range': [20, 40], 'color': 'red'},
            {'range': [40, 60], 'color': 'gold'},
            {'range': [60, 80], 'color': 'limegreen'},
            {'range': [80, 100], 'color': 'forestgreen'}],
        'threshold': {
            'line': {'color': "red", 'width': 4},
            'thickness': 0.75,
            'value': 490}}))

fig.add_annotation(x=0.08,y=0.9,
            text="Bear Market",
            font_size=32)

fig.add_annotation(x=0.92,y=0.9,
            text="Bull Market",
            font_size=32)

fig.update_layout(font = {'color': "#cdafe3", 'family': "monospace"})

st.plotly_chart(fig, use_column_width=True)

top_headlines = newsapi.get_everything(q= str(x),
                                          language='en',)

Articles = top_headlines['articles']
st.header("Featured Articles")
st.subheader(f"[{str(Articles[0]['title'])}]({str(Articles[0]['url'])})")
st.markdown(str(Articles[0]['description']))
st.subheader(f"[{str(Articles[1]['title'])}]({str(Articles[1]['url'])})")
st.markdown(str(Articles[1]['description']))
st.subheader(f"[{str(Articles[2]['title'])}]({str(Articles[2]['url'])})")
st.markdown(str(Articles[2]['description']))
st.subheader(f"[{str(Articles[3]['title'])}]({str(Articles[3]['url'])})")
st.markdown(str(Articles[3]['description']))
st.subheader(f"[{str(Articles[4]['title'])}]({str(Articles[4]['url'])})")
st.markdown(str(Articles[4]['description']))
