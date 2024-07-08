#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
df=pd.read_csv('C:\\Users\\DELL\Documents\\iris_flo.data.csv')
print(df)

df.head()

df.describe()

"""# **Exploratory Data Analysis using iris data**




"""

from sklearn.preprocessing import LabelEncoder
label_encoder=LabelEncoder()
df['Species']=label_encoder.fit_transform(df['Species'])
print(df)

print(df.columns)

df=pd.get_dummies(df,columns=['Species'],dtype=int)
print(df)
print(df.astype(int))

from sklearn.preprocessing import OneHotEncoder
import pandas as pd

# Initialize the OneHotEncoder
encoder = OneHotEncoder()

# Fit and transform the data
encoded_data = encoder.fit_transform(df[['sepal length', 'sepal width', 'petal length', 'petal width']])

# Convert the encoded data to a DataFrame
df_encoded = pd.DataFrame(encoded_data.toarray(), columns=encoder.get_feature_names_out(['sepal length', 'sepal width', 'petal length', 'petal width']))

print(df_encoded)

df.isna()

print(df.isnull())
print(df.isnull().sum())
print(df.fillna("Species"))

df['petal_sepal_lratio'] = df['petal length'] / df['sepal length']
print(df[['petal_sepal_lratio']])
df['petal_sepal_wratio']= df['petal width'] / df['sepal width']
print(df[['petal_sepal_wratio']])


# In[7]:


df=pd.read_csv('C:\\Users\\DELL\Documents\\iris_flo.data.csv')
# Pairplot to visualize relationships between features colored by 'Species'
sns.pairplot(df, hue='Species')
plt.show()


# In[11]:


from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score
X=df.drop("Species", axis=1)
print(X)


# In[9]:


y=df["Species"]
y


# In[17]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3, random_state=42)



# In[18]:


knn=KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train,y_train)


# In[19]:


y_pred = knn.predict(X_test)
print("Accuracy:",accuracy_score(y_test,y_pred))


# In[20]:


print(classification_report(y_test,y_pred))


# In[26]:


new_data=pd.DataFrame({"sepal length":[5.5],"sepal width":2.6,"petal length":1.7,"petal width":1.8})
pred=knn.predict(new_data)
pred[0]


# In[ ]:





# In[ ]:




