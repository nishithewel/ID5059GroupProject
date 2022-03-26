#!/usr/bin/env python
# coding: utf-8

# # Load pandas, matplotlib

# In[48]:


import pandas as pd
import matplotlib.pyplot as plt


# # Load dataset & description

# In[60]:


#Load train_transaction.csv

trans_train = pd.read_csv('train_transaction.csv')
trans_train.shape


# In[50]:


trans_train.describe()


# # Boxplot & delete outliers

# In[51]:


#Define a function called "plot_boxplot"

def plot_boxplot(df, feature):
    df.boxplot(column=[feature])
    plt.grid(False)
    plt.show()


# In[52]:


#Identify outliers for "TransactionAmt"

plot_boxplot(trans_train, "TransactionAmt")


# In[53]:


#Create a function that returns a list of outliers

def outliers(df, feature):
    Q1 = df[feature].quantile(0.25)
    Q3 = df[feature].quantile(0.75)
    IQR = Q3-Q1
    
    lower_bound = Q1 - 0.5 * IQR
    upper_bound = Q3 + 0.5 * IQR
    
    ls = df.index[(df[feature]<lower_bound) | (df[feature]>upper_bound)]
    
    return ls


# In[78]:


#Create an empty list to store the output indices from multiple columns

outlier_list = []
outlier_list.extend(outliers(trans_train,"TransactionAmt"))
len(outlier_list)


# In[80]:


#Define a function to remove the outliers

def remove(df, ls):
    ls = sorted(set(ls))
    df = df.drop(ls)
    return df


# In[81]:


clean_trans_train = remove(trans_train, outlier_list)
clean_trans_train


# In[57]:


#Identify outliers for days(column D1-D15. If the period between transactions are too long, we assume the person might stop using the card or not really using the card)

plot_boxplot(trans_train, "D1")
plot_boxplot(trans_train, "D2")
plot_boxplot(trans_train, "D3")
plot_boxplot(trans_train, "D4")
plot_boxplot(trans_train, "D5")
plot_boxplot(trans_train, "D6")
plot_boxplot(trans_train, "D7")
plot_boxplot(trans_train, "D8")
plot_boxplot(trans_train, "D9")
plot_boxplot(trans_train, "D10")
plot_boxplot(trans_train, "D11")
plot_boxplot(trans_train, "D12")
plot_boxplot(trans_train, "D13")
plot_boxplot(trans_train, "D14")
plot_boxplot(trans_train, "D15")


# In[113]:


#Delete day columns (D1-D15) since we assume in-between transaction period should not be too long

outlier_list_days = []
for feature in ["D1","D2","D3","D4","D5","D6","D7","D8","D9","D10","D11","D12","D13","D14","D15"]:
    outlier_list_days.extend(outliers(clean_trans_train,feature))

clean_trans_train = remove(clean_trans_train, outlier_list_days)
clean_trans_train.shape


# In[126]:


plot_boxplot(clean_trans_train, "D13")

