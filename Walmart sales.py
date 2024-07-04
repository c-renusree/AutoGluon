#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from autogluon.timeseries import TimeSeriesDataFrame,TimeSeriesPredictor
from sklearn.metrics import mean_squared_error,r2_score,mean_absolute_error
from autogluon.features.generators import AutoMLPipelineFeatureGenerator


# In[2]:


df=pd.read_csv("Downloads/Copy of Project/Walmart.csv")
df.head()


# In[3]:


df['Date']=pd.to_datetime(df['Date'],format='%d-%m-%Y')
df.head()


# In[4]:


auto_ml_pipeline_feature_generator = AutoMLPipelineFeatureGenerator()
auto_ml_pipeline_feature_generator.fit_transform(X=df)


# In[6]:


correlation_matrix = df.corr()
correlation_with_sales = correlation_matrix['Weekly_Sales'].sort_values(ascending=False)


# In[7]:


print(correlation_with_sales)


# In[8]:


#df["Store"] = df["Store"].astype("category")
df["Holiday_Flag"]=df["Holiday_Flag"].astype('float64')


# In[9]:


df.info()


# In[10]:


columns_to_drop=['Temperature','Fuel_Price','CPI','Unemployment']
data=df.drop(columns=columns_to_drop,axis=1)
data.head()


# In[11]:


data=data.drop(['Holiday_Flag'],axis=1)


# In[12]:


data.head()


# In[13]:


train_data = TimeSeriesDataFrame.from_data_frame(
    df,
    id_column="Store",
    timestamp_column="Date",
    )


# In[14]:


data.head()


# In[16]:


WEEKEND_INDICES = [5, 6]
timestamps = train_data.index.get_level_values("timestamp")
train_data["Holiday_Flag"] = timestamps.weekday.isin(WEEKEND_INDICES).astype('float64')
train_data.tail()


# In[17]:


train_data.info()


# In[27]:


predictor=TimeSeriesPredictor(prediction_length=2,target="Weekly_Sales").fit(train_data,hyperparameters="default")


# In[28]:


from autogluon.timeseries.utils.forecast import get_forecast_horizon_index_ts_dataframe

future_index = get_forecast_horizon_index_ts_dataframe(train_data,prediction_length=2)
future_timestamps = future_index.get_level_values("timestamp")
known_covariates = pd.DataFrame(index=future_index)
known_covariates["Holiday_Flag"]=future_timestamps.weekday.isin(WEEKEND_INDICES).astype('float64')
known_covariates.tail(5)


# In[29]:


predictor.leaderboard()


# In[30]:


known_covariates.info


# In[31]:


known_covariates = known_covariates[predictor.known_covariates_names]
known_covariates.head()


# In[32]:


predictions=predictor.predict(train_data,known_covariates=known_covariates) 


# In[33]:


print(predictions)


# In[22]:


predictor.leaderboard(train_data,silent=True)

