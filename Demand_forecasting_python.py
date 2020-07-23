#!/usr/bin/env python
# coding: utf-8

# # PYTHON, TIME SERIES FORECASTING using ARIMA, LSTM, PROPHET - Tableau Superstore dataset


# importing required libraries 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
import statsmodels.api as sm
from pylab import rcParams
import os, time, sys, math
import missingno as msno 
import warnings
from statsmodels.tsa.statespace.sarimax import SARIMAX
import plotly.offline as pyoff
import plotly.graph_objs as go
import chart_studio.plotly as py
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tools.eval_measures import rmse
from keras.preprocessing.sequence import TimeseriesGenerator
from keras.layers import Dense, Dropout
from keras.models import Sequential
from keras.optimizers import Adam 
from keras.callbacks import EarlyStopping
from keras.utils import np_utils
from keras.layers import LSTM
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from fbprophet import Prophet


# # Read Datset
# Dataset link: https://community.tableau.com/s/question/0D54T00000CWeX8SAL/sample-superstore-sales-excelxls

# Read Dataset
# Dataset link: 
file_path = 'Path_to_dataset/Sample_Superstore.xls'
sheet1 = 'Orders'
df = pd.read_excel(file_path, sheet_name = sheet1)
df.head(2)

print("Number of rows: ", df.shape[0])
print("Number of columns: ", df.shape[1])
print("\nName of the columns:", df.columns.tolist())
print("\nUnique values\n", df.nunique())


# # Data Preprocessing
df.columns = df.columns.str.replace('-', '_')
df.columns = df.columns.str.replace(' ', '_')
df.info()


# We shall analyse the time series of Blinders sales
# Top 10 occuring products ID and 
rep_rows = pd.DataFrame(df['Sub_Category'].value_counts()).reset_index()
rep_rows.columns = ['Sub_Category','count']
print(rep_rows), print("\nUnique Sub-Category Counts: ",df['Sub_Category'].nunique()),print("Total Count: ",rep_rows['count'].sum())

# converting top 10 product's ID into array
sc_array = rep_rows.head(1)['Sub_Category'].array
sc_array

 
top10 = df.loc[df['Sub_Category'] == 'Binders']
print("Rows: ",top10.shape[0]), print("Columns: ",top10.shape[1]),print("\nMIN DATE: ",top10['Order_Date'].min()), print("MAX DATE: ",top10['Order_Date'].max()) 

cols = ['Row_ID', 'Order_ID', 'Ship_Date', 'Ship_Mode', 'Customer_ID', 'Customer_Name', 'Segment', 'Country', 'City', 'State', 'Postal_Code', 'Region', 'Product_ID', 'Category','Sub_Category' ,'Product_Name', 'Quantity', 'Discount', 'Profit']
top10.drop(cols, axis=1, inplace=True)


top10.reset_index(drop=True, inplace=True)
top10


top10 = top10.groupby([pd.Grouper(key='Order_Date', freq='MS')])['Sales'].sum().reset_index().sort_values('Order_Date')
# freq='W-MON'
print(top10.head(10)), print('\nShape:\n',top10.shape)


top10 = top10.set_index("Order_Date")
top10.head()


# # Blinders - Time series visualization

plt_grph = top10['Sales'].plot(figsize = (16,5), title = "Monthly Sales - Blinder")
plt_grph.set(xlabel='Order_Date', ylabel='Sales - Blinders');


a = seasonal_decompose(top10["Sales"], model = "additive")
a.plot();


plt.figure(figsize = (16,7))
a.seasonal.plot();


# # ARIMA forecast
# Let's run auto_arima() function to get best p,q,d,P,D,Q values

from pmdarima import auto_arima   
auto_arima(top10['Sales'], seasonal=True, m=12,max_p=7, max_d=5,max_q=7, max_P=4, max_D=4,max_Q=4).summary()


# As we can see best arima model chosen by auto_arima() is SARIMAX(0, 0, 1)
# # Splitting the dataset into train and test set

train_data = top10[:len(top10)-12]
test_data = top10[len(top10)-12:]

arima_model = SARIMAX(train_data['Sales'], order = (0,0,1), seasonal_order = (1,2,3,12))
arima_result = arima_model.fit()
arima_result.summary()

arima_pred = arima_result.predict(start = len(train_data), end = len(top10)-1, typ="levels").rename("ARIMA Predictions")
arima_pred

test_data['Sales'].plot(figsize = (16,5), legend=True)
arima_pred.plot(legend = True);


# calculating MSE, MAPE and RMSE values
arima_rmse_error = rmse(test_data['Sales'], arima_pred)
arima_mse_error = arima_rmse_error**2
mean_value = df['Sales'].mean()
arima_mape = np.mean(np.abs((arima_pred - test_data['Sales']) / test_data['Sales'])) * 100
print(f'MSE Error: {arima_mse_error}\nRMSE Error: {arima_rmse_error}\nMean: {mean_value}\nMAPE: {arima_mape}')

test_data['ARIMA_Predictions'] = arima_pred
test_data


# # LSTM forecast
# First we will scale our train and test data with MiMaxScaler().
# MinMaxScaler features by scaling each feature to a given range. 

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(train_data)
scaled_train_data = scaler.transform(train_data)
scaled_test_data = scaler.transform(test_data)


# TimeseriesGenerator() generates batches of temporal data.

from keras.preprocessing.sequence import TimeseriesGenerator

n_input = 12
n_features= 1
generator = TimeseriesGenerator(scaled_train_data, scaled_train_data, length=n_input, batch_size=1)
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

lstm_model = Sequential()
lstm_model.add(LSTM(200, activation='relu', input_shape=(n_input, n_features)))
lstm_model.add(Dense(1))
lstm_model.compile(optimizer='adam', loss='mse')
lstm_model.summary()

lstm_model.fit_generator(generator,epochs=220)

losses_lstm = lstm_model.history.history['loss']
plt.figure(figsize=(12,4))
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.xticks(np.arange(0,250,40))
plt.plot(range(len(losses_lstm)),losses_lstm);

lstm_predictions_scaled = list()

batch = scaled_train_data[-n_input:]
current_batch = batch.reshape((1, n_input, n_features))

for i in range(len(test_data)):   
    lstm_pred = lstm_model.predict(current_batch)[0]
    lstm_predictions_scaled.append(lstm_pred) 
    current_batch = np.append(current_batch[:,1:,:],[[lstm_pred]],axis=1)


# As we know we scaled our data, we have to inverse it to see true predictions. 

lstm_predictions_scaled
lstm_predictions = scaler.inverse_transform(lstm_predictions_scaled)
lstm_predictions


# Comparing ARIMA and LSTM predicitions
test_data['LSTM_Predictions'] = lstm_predictions
test_data
test_data['Sales'].plot(figsize = (16,5), legend=True)
test_data['LSTM_Predictions'].plot(legend = True);


# calculating MSE, MAPE and RMSE values
lstm_rmse_error = rmse(test_data['Sales'], test_data["LSTM_Predictions"])
lstm_mse_error = lstm_rmse_error**2
mean_value = df['Sales'].mean()
lstm_mape = np.mean(np.abs((test_data["LSTM_Predictions"] - test_data['Sales']) / test_data['Sales'])) * 100
print(f'MSE Error: {lstm_mse_error}\nRMSE Error: {lstm_rmse_error}\nMean: {mean_value}\nMAPE: {lstm_mape}')


# # Prophet Forecast
# It is a procedure for forecasting time series data based on an additive model where non-linear trends are fit with yearly, weekly and daily seasonality plus holiday effects

top10.info()

df_pr = top10.copy()
df_pr = top10.reset_index()
df_pr.columns = ['ds','y'] # To use prophet column names should be like that
train_data_pr = df_pr.iloc[:len(df)-12]
test_data_pr = df_pr.iloc[len(df)-12:]

m = Prophet()
m.fit(train_data_pr)
future = m.make_future_dataframe(periods=12,freq='MS')
prophet_pred = m.predict(future)
prophet_pred.tail()

prophet_pred = pd.DataFrame({"Order_Date" : prophet_pred[-12:]['ds'], "Pred" : prophet_pred[-12:]["yhat"]})
prophet_pred = prophet_pred.set_index("Order_Date")
prophet_pred.index.freq = "MS"
prophet_pred

test_data["Prophet_Predictions"] = prophet_pred['Pred'].values

plt.figure(figsize=(16,5))
ax = sns.lineplot(x= test_data.index, y=test_data["Sales"])
sns.lineplot(x=test_data.index, y = test_data["Prophet_Predictions"]);


# calculating MSE, MAPE and RMSE values
prophet_rmse_error = rmse(test_data['Sales'], test_data["Prophet_Predictions"])
prophet_mse_error = prophet_rmse_error**2
mean_value = df['Sales'].mean()
prophet_mape = np.mean(np.abs((test_data["Prophet_Predictions"] - test_data['Sales']) / test_data['Sales'])) * 100
print(f'MSE Error: {prophet_mse_error}\nRMSE Error: {prophet_rmse_error}\nMean: {mean_value}\nMAPE: {prophet_mape}')

rmse_errors = [arima_rmse_error, lstm_rmse_error, prophet_rmse_error]
mse_errors = [arima_mse_error, lstm_mse_error, prophet_mse_error]
mape = [arima_mape, lstm_mape, prophet_mape]
errors = pd.DataFrame({"Models" : ["ARIMA", "LSTM", "Prophet"],"RMSE Errors" : rmse_errors, "MSE Errors" : mse_errors, "MAPE" : mape})
plt.figure(figsize=(16,9))
plt.plot_date(test_data.index, test_data["Sales"], linestyle="solid", label='Sales')
plt.plot_date(test_data.index, test_data["ARIMA_Predictions"], linestyle="-.", label='ARIMA predicitons')
plt.plot_date(test_data.index, test_data["LSTM_Predictions"], linestyle="--", label='LSTM predicitons')
plt.plot_date(test_data.index, test_data["Prophet_Predictions"], linestyle=":", label='Prophet predicitons')
plt.legend()
plt.show()


# # Comparing RMSE, MSE and MAPE values 
print(f"Mean: {test_data['Sales'].mean()}")
errors

# # Comparing ARIMA, LSTM, Prophet Predicitions
test_data



