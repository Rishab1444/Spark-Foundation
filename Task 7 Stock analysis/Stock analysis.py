#!/usr/bin/env python
# coding: utf-8

# # OBJECTIVE

# Creating a hybrid model for stock price and model performance prediction using numerical analysis of historical prices..

# In[5]:


import numpy as np
import pandas as pd
import seaborn as sp
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[8]:


import pandas_datareader as pdr


# In[9]:


stockdata = pdr.get_data_yahoo('RELIANCE.NS', start = "2015-01-01",end = "2020-01-01")


# In[11]:


stockdata


# In[12]:


stockdata.to_csv('reliance.csv')


# In[13]:


stocks = pd.read_csv('reliance.csv')


# In[14]:


stocks.head()


# # Stock Visualization 

# In[15]:


df = stocks.reset_index()['Close']


# In[16]:


df.head()


# In[17]:


plt.plot(df)


# # Data Preproccesing

# In[18]:


from sklearn.preprocessing import MinMaxScaler


# In[19]:


scaler = MinMaxScaler(feature_range=(0,1))
df = scaler.fit_transform(np.array(df).reshape(-1,1))
df


# # SPLITING THE DATA SET 

# In[20]:


train_size=int(len(df)*0.65)
test_size=len(df)-train_size
train_data,test_data=df[0:train_size,:],df[train_size:len(df),:1]


# In[21]:


def create_dataset(dataset, time_step=1):
    datax,datay=[],[]
    for i in range(len(dataset)-time_step-1):
        a=dataset[i:(i+time_step),0]
        datax.append(a)
        datay.append(dataset[i+time_step,0])
    return np.array(datax),np.array(datay)


# In[22]:


time_step=100
X_train,y_train=create_dataset(train_data,time_step)
X_test,y_test=create_dataset(test_data,time_step)


# In[23]:


X_train


# In[24]:


X_train=X_train.reshape(X_train.shape[0],X_train.shape[1],1)
X_test=X_test.reshape(X_test.shape[0],X_test.shape[1],1)


# In[25]:


X_train.shape, X_test.shape, y_train.shape, y_test.shape


#   # TENSOR FLOW

# In[26]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM


# In[28]:


model=Sequential()
model.add(LSTM(50,return_sequences=True,input_shape=(100,1)))
model.add(LSTM(50,return_sequences=True))
model.add(LSTM(50))
model.add(Dense(1))
model.compile(loss='mean_squared_error',optimizer='adam')


# In[31]:


model.summary()


# ## Model Training

# In[33]:


model.fit(X_train,y_train,validation_data=(X_test,y_test),epochs=100,batch_size=64,verbose=1)


# In[72]:


train_predict = model.predict(X_train)


# In[73]:


test_predict=model.predict(X_test)


# In[74]:


train_predict=scaler.inverse_transform(train_predict)
test_predict=scaler.inverse_transform(test_predict)


# ## MEAN SQUARED ERROR

# In[75]:


import math
from sklearn.metrics import mean_squared_error
math.sqrt(mean_squared_error(y_train,train_predict))


# In[76]:


math.sqrt(mean_squared_error(y_test,test_predict))


# In[77]:


look_back=100
trainpredictplot=np.empty_like(df)
trainpredictplot[:, :]=np.nan
trainpredictplot[look_back:len(train_predict)+look_back,:]=train_predict
#shif test predict for plot
testpredictplot=np.empty_like(df)
testpredictplot[:, :]=np.nan
testpredictplot[len(train_predict)+(look_back*2)+1:len(df)-1,:]=test_predict

plt.plot(scaler.inverse_transform(df))
plt.plot(trainpredictplot)
plt.plot(testpredictplot)
plt.show()

