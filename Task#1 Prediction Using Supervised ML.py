#!/usr/bin/env python
# coding: utf-8

# ## <span style="color:blue"> Submitted By - SUMUNNA PAUL</span>
# 
# ## <span style="color:purple"> DATA SCIENCE AND ANALYTICS INTERN</span>
# 
# ## <span style="color:red"> THE SPARKS FOUNDATION INTERNSHIP PROGRAM JULY'21</span>
# 
# # Task:1 Prediction Using Supervised ML
# 
# In this case study, my task is to create a machine learning model which can predict the percentage of marks that a student is expected to score based upon the number of hours they studied.
# 
# In below case study I will discuss the step by step approach to solve any supervised ML Regression problem!
# 
# The flow of the case study is as below:
# * Reading the data in Python
# * Defining the problem statement
# * Identifying the Target variable
# * Basic Data Exploration
# * Relationship Exploration 
# * Statistical Feature Selection using Correlation value
# * Training the Algorithm
# * Making Predictions
# * Evaluating the Model

# ## Reading the data into Python
# The data has one file. This file contains 25 students score with no.of study hours data.
# 
# ## Data description
# The business meaning of each column in the data is as below
# 
# * <b>Hours</b>: No.of study hours of a student
# * <b>Score</b>: The obtained score of the student

# In[1]:


# Reading the dataset
import numpy as np
import pandas as pd
data_url= 'http://bit.ly/w-data'
StudentStudyData= pd.read_csv(data_url)
print('Shape of data before removing duplicate data :',StudentStudyData.shape)
StudentStudyData=StudentStudyData.drop_duplicates()
print('Shape of data after removing duplicate data :',StudentStudyData.shape)
StudentStudyData.head(10)


# ## Defining the Problem Statement 
# #### Create a ML model which can predict the percentage of marks that a student is expected to score based upon the number of hours they studied.
# * Target Variable: Scores
# * Predictor: Hours
# 
# ## Basic Data Exploration
# Initial assessment of the data should be done to identify which columns are Quantitative, Categorical or Qualitative.

# In[22]:


StudentStudyData.nunique()


# Based on the basic exploration above, you can now create a simple report of the data.
# * <b>Scores</b>: Continuous. Selected. This is the <b>Target Variable!</b>
# * <b>Hours</b>: Continuous. Selected.
# 
# ## Relationship Exploration 
# When the Target variable is continuous and the predictor is also continuous, we can visualize the relationship between the two variables using scatter plot.

# In[4]:


get_ipython().run_line_magic('matplotlib', 'inline')
StudentStudyData.plot.scatter(x='Hours',y='Scores',title="Hours VS Score-Percentage")


# ## Scatter Charts Interpretation
# Based on this chart, we can clearly see that there is a positive linear relation between the number of hours studied and percentage of score.
# 
# ## Statistical Feature Selection (Continuous Vs Continuous) using Correlation value
# 
# Pearson's correlation coefficient can simply be calculated as
# the covariance between two features $x$ and $y$ (numerator) divided by the product
# of their standard deviations (denominator)

# In[23]:


ContinuousCols=['Scores','Hours']
CorrelationData=StudentStudyData[ContinuousCols].corr()
CorrelationData


# In[25]:


# Filtering only those columns where absolute correlation > 0.5 with Target Variable
CorrelationData['Scores'][abs(CorrelationData['Scores'])>0.5]


# ## Machine Learning: Splitting the data into Training and Testing sample
# We dont use the full data for creating the model. Some data is randomly selected and kept aside for checking how good the model is. This is known as Testing Data and the remaining data is called Training data on which the model is built. Typically 70% of data is used as Training data and the rest 30% is used as Tesing data.

# In[ ]:


X = StudentStudyData.iloc[:, :-1].values  
y = StudentStudyData.iloc[:, 1].values
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=0.2,random_state=0)


# ### Training the Algorithm
# We have split our data into training and testing sets, and now is finally the time to train our algorithm.

# In[26]:


from sklearn.linear_model import LinearRegression
RegModel=LinearRegression()
LREG = RegModel.fit(X_train,y_train)
print('Training Complete')


# In[31]:


line = LREG.coef_*X+LREG.intercept_
import matplotlib.pyplot as plt
# Plotting for the test data
plt.scatter(X, y)
plt.plot(X, line,color='green');
plt.show()


# ### Making Predictions
# Now that we have trained our algorithm, it's time to make some predictions.

# In[28]:


# Comparing Actual vs Predicted
print(X_test)
Prediction= LREG.predict(X_test)
TestingDataResult=pd.DataFrame({'Actual':y_test, 'Predictor':Prediction})
TestingDataResult


# In[29]:


# Find the predicted Score if a student studies 9.25hours/day.
hours= 9.25
Own_Prediction= LREG.predict([[hours]])
print('No.of Hours:{}'.format(hours))
print('Predicted Score:{}'.format(Own_Prediction[0]))


# ## Evaluating the model
# The final step is to evaluate the performance of algorithm. This step is particularly important to compare how well different algorithms perform on a particular dataset. For simplicity here, we have chosen the mean square error. There are many such metrics.

# In[33]:


from sklearn import metrics
print('Mean Absolute Error :',round(metrics.mean_absolute_error(y_test,Prediction),2))

