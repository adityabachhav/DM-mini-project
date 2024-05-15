#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#DM project
#performed by Aditya Bachhav
#roll no:CS3150
#DIV:A 
#RBT21cs051
#TY


# In[1]:


from sklearn.linear_model import LinearRegression, Lasso, Ridge
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn import metrics
from sklearn.svm import SVC

from sklearn.ensemble import RandomForestRegressor
import warnings
warnings.filterwarnings('ignore')


# In[2]:


big_mart_data = pd.read_csv('Train.csv')


# In[3]:


big_mart_data.head()
big_mart_data.shape


# In[4]:


big_mart_data.info()


# In[5]:


big_mart_data['Item_Weight'].mean()


# In[6]:


big_mart_data['Item_Weight'].fillna(big_mart_data['Item_Weight'].mean(), inplace=True)


# In[7]:


big_mart_data['Outlet_Size'].mode()


# In[8]:


mode_of_Outlet_size = big_mart_data.pivot_table(values='Outlet_Size', columns='Outlet_Type', aggfunc=(lambda x: x.mode()[0]))


# In[9]:


print(mode_of_Outlet_size)


# In[10]:


miss_values = big_mart_data['Outlet_Size'].isnull() 


# In[11]:


print(miss_values)


# In[12]:


big_mart_data.loc[miss_values, 'Outlet_Size'] = big_mart_data.loc[miss_values,'Outlet_Type'].apply(lambda x: mode_of_Outlet_size[x])


# In[13]:


big_mart_data.isnull().sum()


# In[14]:


big_mart_data.describe()


# In[15]:


sns.set()


# In[16]:


plt.figure(figsize=(6,6))
sns.distplot(big_mart_data['Item_Weight'])
plt.show()


# In[17]:


plt.figure(figsize=(6,6))
sns.distplot(big_mart_data['Item_Visibility'])
plt.show()


# In[18]:


plt.figure(figsize=(6,6))
sns.distplot(big_mart_data['Item_MRP'])
plt.show()


# In[19]:


plt.figure(figsize=(6,6))
sns.distplot(big_mart_data['Item_Outlet_Sales'])
plt.show()


# In[20]:


plt.figure(figsize=(6,6))
sns.countplot(x='Outlet_Establishment_Year', data=big_mart_data)
plt.show()


# In[21]:


plt.figure(figsize=(6,6))
sns.countplot(x='Item_Fat_Content', data=big_mart_data)
plt.show()


# In[22]:


plt.figure(figsize=(30,6))
sns.countplot(x='Item_Type', data=big_mart_data)
plt.show()


# In[23]:


plt.figure(figsize=(6,6))
sns.countplot(x='Outlet_Size', data=big_mart_data)
plt.show()


# In[24]:


big_mart_data.head()


# In[25]:


big_mart_data['Item_Fat_Content'].value_counts()


# In[26]:


big_mart_data.replace({'Item_Fat_Content': {'low fat':'Low Fat','LF':'Low Fat', 'reg':'Regular'}}, inplace=True)


# In[27]:


big_mart_data['Item_Fat_Content'].value_counts()


# In[28]:


encoder = LabelEncoder()


# In[29]:


big_mart_data['Item_Identifier'] = encoder.fit_transform(big_mart_data['Item_Identifier'])

big_mart_data['Item_Fat_Content'] = encoder.fit_transform(big_mart_data['Item_Fat_Content'])

big_mart_data['Item_Type'] = encoder.fit_transform(big_mart_data['Item_Type'])

big_mart_data['Outlet_Identifier'] = encoder.fit_transform(big_mart_data['Outlet_Identifier'])

big_mart_data['Outlet_Size'] = encoder.fit_transform(big_mart_data['Outlet_Size'])

big_mart_data['Outlet_Location_Type'] = encoder.fit_transform(big_mart_data['Outlet_Location_Type'])

big_mart_data['Outlet_Type'] = encoder.fit_transform(big_mart_data['Outlet_Type'])


# In[30]:


big_mart_data.head()


# In[31]:


X = big_mart_data.drop(columns='Item_Outlet_Sales', axis=1)
Y = big_mart_data['Item_Outlet_Sales']


# In[32]:


print(X)


# In[33]:


print(Y)


# In[34]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)


# In[35]:


print(X.shape, X_train.shape, X_test.shape)


# In[36]:


regressor = XGBRegressor()


# In[37]:


regressor.fit(X_train, Y_train)


# In[38]:


training_data_prediction = regressor.predict(X_train)


# In[39]:


r2_train = metrics.r2_score(Y_train, training_data_prediction)


# In[40]:


print('R Squared value = ', r2_train)


# In[41]:


test_data_prediction = regressor.predict(X_test)


# In[42]:


r2_test = metrics.r2_score(Y_test, test_data_prediction)


# In[43]:


print('R Squared value = ', r2_test)


# In[44]:


from sklearn.metrics import mean_absolute_error as mae
models = [LinearRegression(), Lasso(), RandomForestRegressor(), Ridge()]
for i in range(4):
    models[i].fit(X_train, Y_train)
    print(f'{models[i]} : ')
    train_preds = models[i].predict(X_train)
    print('Training Error : ', mae(Y_train, train_preds))
    val_preds = models[i].predict(X_test)
    print('Validation Error : ', mae(Y_test, val_preds))
    print()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




