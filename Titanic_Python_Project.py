# Beginner (Data Exploration)
# Load the dataset using pandas and display the first 10 rows.
import pandas as pd
from pandas import value_counts

df=pd.read_excel("Titanic_Project.xlsx")
print(df.head(3))
# How many passengers are in the dataset?
k=df["Name"].count()
print(k)
h=len(df)
print(h)

# What are the data types of each column?
print(df.info())
# How many missing values are there in each column?
f=df.isna().sum()
print(f)
# What is the overall survival rate?
g=df.groupby("Name")["Survived"].sum()
print(df["Survived"].sum())
print(g)


# 🟡 Intermediate (Data Analysis)
# What is the survival rate by gender?
f=df.groupby("Sex")["Survived"].mean()
print(f)
# What is the survival rate by passenger class (Pclass)?
e=df.groupby("Pclass")["Survived"].mean()
print("the average survival rate by passenger class is",e)
# What is the average age of passengers who survived vs those who didn’t?
r=df.groupby("Survived")["Age"].mean()
print(r)
# Which embarkation port (Embarked) had the highest survival rate?
y=df.groupby("Embarked")["Survived"].mean()
print("The embarkation port with the highest survival rate is",y)
# print(y.max())
print("The max is",y.idxmax(),y.max())
# What is the average fare paid by survivors vs non-survivors?
h=df.groupby("Survived")["Fare"].mean()
print(h)
# 🔵 Data Cleaning & Feature Engineering
# Fill missing values in the Age column (try mean or median).
df["Age"].fillna(df["Age"].mean(),inplace=True)
# Drop rows with missing Embarked values.
df.dropna(subset=["Embarked","Age"],inplace=True)
# Create a new column FamilySize = SibSp + Parch.
df["FamilySize"]=df["SibSp"]+df["Parch"]
print(df)
# Create an IsAlone column (1 if alone, 0 otherwise).
import numpy as np
df["IsAlone"]=np.where(df["FamilySize"]==0,1,0)
# Convert categorical variables (Sex, Embarked) into numeric format.
df["Sex"]=df["Sex"].map({"male":0,"female":1})
df=pd.get_dummies(df,columns=["Embarked"],drop_first=True)
# 🟣 Advanced Analysis
# Which combination of gender + class had the highest survival rate?
x=df.groupby(["Sex","Pclass"])["Survived"].max()
print(x)
# Find the top 5 passengers who paid the highest fare.
fare=df[["Name","Fare"]].sort_values(ascending=False,by="Fare")
print(fare.head(5))
# Group passengers into age bins (e.g., child, adult, senior) and compare survival rates.
# df["Age"].fillna(32,inplace=True)
df.fillna({"Age":32},inplace=True)
category_list=[0,12,50,100]
category_bin=["Child","Adult","Senior"]
df["Age_Group"]=pd.cut(df["Age"],bins=category_list,labels=category_bin)
print(df)
# Create a pivot table showing survival rate by Sex and Pclass.
x=pd.pivot_table(df,index="Sex",values="Survived",columns="Pclass")
print(x)

# Identify correlations between numerical variables.
s=df.corr(numeric_only=True)
print(s)
# 🔴 Machine Learning (Real-World Style)
# Build a logistic regression model to predict survival.
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
df_ml=df.copy()
df_ml=df_ml.drop(["Name","PassengerId"],axis=1)
print(df_ml.head())
df_ml["Sex"]=df_ml["Sex"].map({"male":0,"female":1})

X=df_ml.drop("Survived",axis=1)
y=df_ml["Survived"]
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
print(X_train)
print(X_test)
print(y_train)
y=df_ml["Survived"]
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
# from sklearn.metrics import accuracy_score
# model=LogisticRegression(max_iter=1000)
# model.fit(X_train,y_train)
# y_pred=model.predict(X_test)#prediction
# Build a logistic regression model to predict survival.


from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import  LogisticRegression
df_ml = df.copy()


# Drop unwanted columns
df_ml.drop(["Name", "PassengerId", "Age_Group"], axis=1, inplace=True, errors="ignore")


# Convert categorical
#df_ml["Sex"] = df_ml["Sex"].map({"male": 0, "female": 1})
#df_ml = pd.get_dummies(df_ml, columns=["Embarked"], drop_first=True)


# Convert True/False to int (safe way)
df_ml["Embarked_Q"] = df_ml["Embarked_Q"].astype(int)
df_ml["Embarked_S"] = df_ml["Embarked_S"].astype(int)


# Fill missing values
df_ml.fillna({"Age":df_ml["Age"].median()}, inplace=True)


# Final check
#print(df_ml.isnull().sum())
X = df_ml.drop("Survived", axis=1)
y = df_ml["Survived"]


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
   X, y, test_size=0.2, random_state=42
)


from sklearn.linear_model import LogisticRegression
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)


from sklearn.metrics import accuracy_score
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

# 💡 Bonus (Visualization)
# Plot survival count by gender.
g=df.groupby("Sex")["Survived"].sum()
print(g)
import matplotlib.pyplot as plt
plt.g(kind="Bar")
