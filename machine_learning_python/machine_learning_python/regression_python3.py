
# coding: utf-8

# In[71]:

import pandas as pd
import sklearn as sk
#import numpy as np
import matplotlib
import matplotlib.pyplot as plt
#import seaborn as sns
matplotlib.style.use("ggplot")
#get_ipython().magic('matplotlib inline')


# In[72]:

print(sk.__version__)


# In[127]:

df = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data",delimiter = r"\s+",names = ["CRIM","ZN","INDUS","CHAS","NOX","RM","AGE","DIS","RAD","TAX","PTRATIO","B","LSTAT","MEDV"])
df.sample()
#df.head


# In[ ]:




# In[125]:

df.shape


# In[77]:

df[pd.isnull(df).any(axis=1)]


# In[78]:

x=df[["CRIM","ZN","INDUS","CHAS","NOX","RM","AGE","DIS","RAD","TAX","PTRATIO","B","LSTAT"]]
x.shape


# In[79]:

y=df["MEDV"]
y.shape


# In[80]:

type(y)


# In[81]:

type(x)


# In[82]:

#df = pd.read_csv("http://archive.ics.uci.edu/ml/machine-learning-databases/car/car.data")


# In[86]:

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20,random_state=5)


# In[89]:

from sklearn import linear_model
reg=linear_model.LinearRegression()


# In[90]:

reg.fit(x_train,y_train)


# In[91]:

reg.intercept_


# In[92]:

reg.coef_


# In[103]:

x_test


# In[104]:

y_predict=reg.predict(x_test)


# In[105]:

y_predict


# In[106]:

y_test


# In[107]:

y_test_mat = y_test.as_matrix()


# In[109]:

plt.figure(figsize=(15,10))
plt.plot(y_test_mat,ms=50,alpha=1)
plt.plot(y_predict,ms=50,alpha=1)
legend_list=["test_data","prediction"]
print(plt.legend(legend_list,fontsize=25,loc=4))


# In[116]:

from sklearn.metrics import mean_squared_error
mean_squared_error(y_test,y_predict)


# In[118]:

from sklearn.metrics import r2_score
print("r2 scores: ",r2_score(y_test,y_predict))


# In[ ]:



