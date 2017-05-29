
# coding: utf-8

# In[976]:

import pandas as pd
import sklearn as sk
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
matplotlib.style.use("ggplot")
get_ipython().magic('matplotlib inline')
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score


# In[977]:

# df = pd.read_csv('https://www.kaggle.com/c/titanic/download/train.csv',index_col='PassengerId',names= ['PassengerId','Survived','Pclass','Name','Sex','Age','SibSp','Parch','Ticket','Fare','Cabin','Embarked'])
df = pd.read_csv('/home/abhiram/Downloads/train.csv',names= ['PassengerId','Survived','Pclass','Name','Sex','Age','SibSp','Parch','Ticket','Fare','Cabin','Embarked'],skiprows=1)
df.info()


# In[978]:

df[pd.isnull(df).any(axis=1)]
df.shape


# In[979]:

df["Sex"] = df["Sex"].map({'female': 1, 'male': 0})
df["Age"] = df["Age"].fillna(value=30)
df["Fare"] = df["Fare"].fillna(value=10.1)


# In[980]:

df["Embarked"] = df["Embarked"].map({'S': 1, 'Q': 2,'C':3})
df["Embarked"] = df["Embarked"].fillna(value=0)


# In[981]:

x = df[["Pclass","Sex","Age","SibSp","Parch","Fare","Embarked"]]
x.shape
x.sample()


# In[982]:

y = df["Survived"]
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.10,random_state=5)
reg=linear_model.LinearRegression()


# In[983]:

x_train.sample()


# In[984]:

y_train.shape


# In[985]:

x_train.info()


# In[ ]:




# In[986]:

reg.fit(x_train,y_train)


# In[987]:

y_predict=reg.predict(x_test)
y_pred = [round(i) for i in y_predict]  
y_test_mat = y_test.as_matrix()
plt.figure(figsize=(15,10))
plt.plot(y_test_mat,ms=50,alpha=1)
plt.plot(y_pred,ms=50,alpha=1)
legend_list=["test_data","prediction"]
plt.legend(legend_list,fontsize=25,loc=4)


# In[988]:

mean_squared_error(y_test,y_pred)


# In[989]:

type(y_pred)


# In[990]:

r2_score(y_test,y_pred)


# In[991]:

test_data_set1 = pd.read_csv("/home/abhiram/Downloads/test.csv",names= ['PassengerId','Pclass','Name','Sex','Age','SibSp','Parch','Ticket','Fare','Cabin','Embarked'],skiprows=1,index_col=False)
test_data_set1.info()


# In[992]:

test_data_set_write = pd.read_csv("/home/abhiram/Downloads/test.csv",names= ['PassengerId','Pclass','Name','Sex','Age','SibSp','Parch','Ticket','Fare','Cabin','Embarked'],skiprows=1)
test_data_set = pd.read_csv("/home/abhiram/Downloads/test.csv",names= ['PassengerId','Pclass','Name','Sex','Age','SibSp','Parch','Ticket','Fare','Cabin','Embarked'],skiprows=1)

test_data_set["Age"] = test_data_set["Age"].fillna(value=28)
test_data_set["Fare"] = test_data_set["Fare"].fillna(value=10.1)
test_data_set["Sex"] = test_data_set["Sex"].map({'female': 1, 'male': 0})
test_data_set["Embarked"] = test_data_set["Embarked"].map({'S': 1, 'Q': 2,'C':3})
x_test_data = test_data_set[["Pclass","Sex","Age","SibSp","Parch","Fare","Embarked"]]
x_test_data.shape
# x_test_data.sample()
# y_test_data = test_data_set["Survived"]


# In[993]:

# x_test_data["Sex"] = x_test_data["Sex"].map({'female': 1, 'male': 0})
# x_test_data["Embarked"] = x_test_data["Embarked"].map({'S': 1, 'Q': 2,'C':3})


# In[994]:

# x_test_data["Survival"]=[round(i) for i in reg.predict(x_test_data)].tolist()
y_predict_data=reg.predict(x_test_data)
y_pred_data = [int(round(i)) for i in y_predict_data]  
# pd.Series(y_pred_data).shape
test_data_set_write["Survival"]=pd.Series(y_pred_data).values
# type([y_pred_data])
# [y_pred_data].shape
# y_pred_data.info()
# y_test_mat_data = y_test_data.as_matrix()
# plt.figure(figsize=(15,10))
# plt.plot(y_test_mat_data,ms=50,alpha=1)
# plt.plot(y_pred_data,ms=50,alpha=1)
# legend_list=["test_data","prediction"]
# plt.legend(legend_list,fontsize=25,loc=4)
# test_data_set.shape
cols = test_data_set_write.columns.tolist()
cols = cols[-1:] + cols[:-1]
test_data_set_write = test_data_set_write[cols]
write_data = test_data_set_write[["Survival","Pclass","Sex","Age","SibSp","Parch","Ticket","Fare","Cabin","Embarked"]]
write_data.to_csv("/home/abhiram/Downloads/result.csv",index=False)


# In[ ]:



