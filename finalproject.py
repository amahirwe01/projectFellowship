#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import warnings
warnings.filterwarnings("ignore")
#get_ipython().run_line_magic('matplotlib', 'inline')
import missingno
import warnings
warnings.filterwarnings("ignore")
import math, time, random, datetime
import featuretools as ft


# In[2]:


dataset=pd.read_csv('weatherAUS.csv')
dataset.head(10)


# In[3]:


dataset.head()


# # finding missing data

# In[4]:


dataset.isna().sum()


# In[5]:


dataset.info()


# In[6]:


dataset['Date'] = pd.to_datetime(dataset['Date'])
dataset['year'] = dataset['Date'].dt.year
dataset['month'] = dataset['Date'].dt.month
dataset['day'] = dataset['Date'].dt.day


# In[7]:


#dataset.drop('Date', axis = 1, inplace = True)
dataset['Date'] = pd.to_datetime(dataset['Date']).dt.month
dataset.rename(columns={'Date':'Month'}, inplace=True)
dataset.head(10)


# In[8]:


dataset.info()


# # encoding categorical and numerical data

# In[9]:


categorical_features = [column_name for column_name in dataset.columns if dataset[column_name].dtype == 'O']
dataset[categorical_features].isnull().sum()


# In[10]:


categorical_features_with_null = [feature for feature in categorical_features if dataset[feature].isnull().sum()]
for each_feature in categorical_features_with_null:
    mode_val = dataset[each_feature].mode()[0]
    dataset[each_feature].fillna(mode_val,inplace=True)


# In[11]:


numerical_features = [column_name for column_name in dataset.columns if dataset[column_name].dtype != 'O']
dataset[numerical_features].isnull().sum()


# In[12]:


numerical_features_with_null = [feature for feature in numerical_features if dataset[feature].isnull().sum()]
for feature in numerical_features_with_null:
    mean_value = dataset[feature].mean()
    dataset[feature].fillna(mean_value,inplace=True)


# In[13]:


def encode_data(feature_name):

    mapping_dict = {}

    unique_values = list(dataset[feature_name].unique())

    for idx in range(len(unique_values)):

        mapping_dict[unique_values[idx]] = idx

    return mapping_dict

dataset['RainToday'].replace({'No':0, 'Yes': 1}, inplace = True)

dataset['RainTomorrow'].replace({'No':0, 'Yes': 1}, inplace = True)

dataset['WindGustDir'].replace(encode_data('WindGustDir'),inplace = True)

dataset['WindDir9am'].replace(encode_data('WindDir9am'),inplace = True)

dataset['WindDir3pm'].replace(encode_data('WindDir3pm'),inplace = True)

dataset['Location'].replace(encode_data('Location'), inplace = True)

print (dataset['Location'])
# In[14]:


corr = dataset.corr()
corr.style.background_gradient(cmap='coolwarm')


# In[15]:


dataset.info()


# In[16]:


dataset = dataset.drop(['Evaporation', 'Sunshine', 'Cloud9am', 'Cloud3pm'], axis = 1)


# In[17]:


dataset.head(10)


# In[37]:


X = dataset[['Month','Location','MinTemp','MaxTemp','Rainfall','RainToday']]
Y = dataset['RainTomorrow']


# In[ ]:





# # splitting dataset into traing and testing

# In[38]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X,Y, test_size = 0.15, random_state = 0)


# # feature scaling

# In[39]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test= scaler.fit_transform(x_test)


# In[40]:


from sklearn.linear_model import LogisticRegression
classifier=LogisticRegression(random_state=0,C=1.0)
classifier.fit(x_train,y_train)


# In[41]:


y_pred=classifier.predict(x_test)
y_pred


# In[42]:


print("testing score")
print(classifier.score(x_test,y_test))
print('\r')
print("traing score")
print(classifier.score(x_train,y_train))


# In[43]:


from sklearn.metrics import confusion_matrix
cm= confusion_matrix(y_test,y_pred)
cm


# In[44]:


from sklearn.metrics import accuracy_score
print ("Accuracy : ", accuracy_score(y_test, y_pred))


# In[47]:


import pickle


# In[48]:


myfile= 'myfile_model.sav'
pickle.dump(classifier, open(myfile, 'wb'))


# In[ ]:

from matplotlib.colors import ListedColormap
X_set, y_set = x_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, 
                               stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, 
                               stop = X_set[:, 1].max() + 1, step = 0.01))

plt.contourf(X1, X2, classifier.predict(
             np.array([X1.ravel(), X2.ravel()]).T).reshape(
             X1.shape), alpha = 0.75, cmap = ListedColormap(('red', 'green')))
  
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
  
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('purple', 'green'))(i), label = j)
      
plt.title('Classifier (Test set)')
plt.xlabel('Month')
plt.ylabel('Location')
plt.legend()
plt.show()

# In[ ]:





# In[ ]:





# In[ ]:




