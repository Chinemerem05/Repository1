#!/usr/bin/env python
# coding: utf-8

# In[6]:


# importing necesdsary libraries

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')


# In[7]:


data = pd.read_csv("heart_disease.csv")
data.head()


# In[8]:


data.shape


# In[9]:


data.describe()


# In[13]:


data = data.drop (['Unnamed: 0'], axis=1)


# In[14]:


data.shape


# In[15]:


# distribution of data

plt.figure(figsize=(20,25), facecolor='blue')
plotnumber = 1

for column in data:
    if plotnumber<=14 :
        ax = plt.subplot(4,4,plotnumber)
        sns.distplot(data[column])
        plt.xlabel(column,fontsize=20)
        
    plotnumber+=1
plt.show()


# In[17]:


df_features = data.drop('target', axis=1)


# In[18]:


# visualizing the outliers

plt.figure(figsize=(20,25))
graph = 1

for column in df_features:
    if graph<=14 :
        plt.subplot(4,4,graph)
        ax=sns.boxplot(data=df_features[column])
        plt.xlabel(column,fontsize=15)
        
    graph+=1
plt.show()


# In[19]:


data.shape


# In[20]:


# finding the IQR to identify outliers

#1st quantile
q1 = data.quantile(0.25)

#3rd quantile
q3 = data.quantile(0.75)

#IQR
iqr = q3-q1


# In[21]:


q1


# In[23]:


#validating outliers
trestbps_high = (q3.trestbps + (1.5 * iqr.trestbps))
trestbps_high


# In[24]:


#checking the indexes which have higher values
np_index = np.where (data['trestbps'] > trestbps_high)
np_index


# In[25]:


# droping the index which I found in the above cell
data=data.drop(data.index[np_index])


# In[26]:


data.reset_index()


# In[27]:


chol_high = (q3.chol + (1.5 * iqr.chol))
chol_high


# In[28]:


np_index = np.where (data['chol'] > chol_high)
np_index


# In[29]:


data=data.drop(data.index[np_index])


# In[30]:


data.reset_index()


# In[31]:


fbs_high = (q3.trestbps + (1.5 * iqr.fbs))
fbs_high


# In[32]:


np_index = np.where (data['fbs'] > fbs_high)
np_index


# In[33]:


data=data.drop(data.index[np_index])


# In[34]:


data.reset_index()


# In[35]:


oldpeak_high = (q3.oldpeak + (1.5 * iqr.oldpeak))
oldpeak_high


# In[36]:


np_index = np.where (data['oldpeak'] > oldpeak_high)
np_index


# In[37]:


data=data.drop(data.index[np_index])
data.reset_index()


# In[38]:


ca_high = (q3.ca + (1.5 * iqr.ca))
ca_high

np_index = np.where (data['ca'] > ca_high)
np_index

data=data.drop(data.index[np_index])

data.reset_index()


# In[39]:


thalach_low = (q1.thalach - (1.5 * iqr.thalach))
thalach_low

np_index = np.where (data['thalach'] < thalach_low)
np_index

data=data.drop(data.index[np_index])

data.reset_index()


# In[40]:


thal_low = (q1.thal - (1.5 * iqr.thal))
thal_low

np_index = np.where (data['thal'] < thal_low)
np_index

data=data.drop(data.index[np_index])

data.reset_index()


# In[41]:


plt.figure(figsize=(20,25), facecolor='blue')
plotnumber = 1

for column in data:
    if plotnumber<=14 :
        ax = plt.subplot(4,4,plotnumber)
        sns.distplot(data[column])
        plt.xlabel(column,fontsize=20)
        
    plotnumber+=1
plt.show()


# In[42]:


#finding the relationship
x = data.drop(columns = ['target'])
y = data['target']


# In[44]:


#checking how features are related
plt.figure(figsize=(20,25))
plotnumber = 1

for column in x:
    if plotnumber<=14 :
        ax = plt.subplot(4,4,plotnumber)
        sns.stripplot(x=y, y=x[column],hue=y)       
    plotnumber+=1
plt.show()


# In[45]:


#multicollinearity check
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)


# In[47]:


x_scaled.shape[1]


# In[48]:


#finding variance factor in the scaled column i.e x_scaled.shape[1] (1/(1-R2))


# In[49]:


vif = pd.DataFrame()
vif["vif"] = [variance_inflation_factor(x_scaled, i) for i in range(x_scaled.shape[1])]
vif["Features"] = x.columns
vif


# In[50]:


#using 5 as the vif score, there is no multicollinearity between the features, so all will be used


# In[52]:


x_train,x_test,y_train,y_test = train_test_split(x_scaled,y, test_size = 0.25, random_state = 355)


# In[53]:


#Model building


# In[54]:


log_reg = LogisticRegression()
log_reg.fit(x_train,y_train)


# In[55]:


y_pred = log_reg.predict(x_test)


# In[56]:


y_pred


# In[57]:


log_reg.predict_proba(x_test)


# In[58]:


#confusion matrix
conf_mat = confusion_matrix(y_test,y_pred)
conf_mat


# In[59]:


#model accuracy
accuracy = accuracy_score(y_test,y_pred)
accuracy


# In[60]:


#checking recall, precision and F1 score

from sklearn.metrics import classification_report


# In[61]:


print (classification_report(y_test,y_pred))


# In[62]:


#ROC  curve
fpr,tpr, thresholds = roc_curve(y_test, y_pred)


# In[63]:


print ('Threshold =', thresholds)
print ('True positive rate =', tpr)
print ('False positive rate = ',fpr)


# In[70]:


plt.plot(fpr,tpr, color ='blue', label = 'ROC')
plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')

plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('Receiver Operating Characteristics(ROC) Curve')
plt.legend()
plt.show()


# In[71]:


auc_score = roc_auc_score(y_test,y_pred)
print (auc_score)


# In[ ]:




