#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np


# In[2]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[4]:


titanic=pd.read_csv('titanic.csv')


# In[5]:


titanic


# In[6]:


titanic['Family']=titanic['SibSp']+titanic['Parch']


# In[7]:


titanic


# In[27]:


titanic.drop(['SibSp','Parch','Cabin'],axis=1,inplace=True)


# In[28]:


titanic


# Corelation between Survived and Sex

# In[8]:


sns.countplot(titanic['Survived'],data=titanic,hue='Sex')


# In[9]:


p3=titanic[titanic['Pclass']==3]
t3=p3[p3['Survived']==0].count()['PassengerId']


# In[10]:


p2=titanic[titanic['Pclass']==2]
t2=p2[p2['Survived']==0].count()['PassengerId']


# In[11]:


p1=titanic[titanic['Pclass']==1]
t1=p1[p1['Survived']==0].count()['PassengerId']


# In[12]:


y=[t1,t2,t3]


# In[13]:


x=['Class 1','Class 2','Class 3']


# In[19]:


plt.title('Percentage of people who died travelling from respective classes')
plt.pie(y,labels=x,autopct='%0.2f%%')
plt.show()


# Corelation between Survived and Pclass

# In[18]:


survived=titanic[titanic['Survived']==1]['Pclass'].value_counts()
dead=titanic[titanic['Survived']==0]['Pclass'].value_counts()
df=pd.DataFrame([survived,dead])
df.index=['survived','dead']
df.plot(kind='bar')
plt.show()


# Corelation between Survived,Pclass and Sex

# In[33]:


sns.barplot(data=titanic, x="Pclass", hue='Sex', y='Survived', estimator=np.mean)
plt.ylabel('proportion of survival')


# Corelation between Embarked,Sex and Survived

# In[35]:


sns.barplot(data=titanic,x='Embarked',hue='Sex',y='Survived',estimator=np.mean)


# In[ ]:




