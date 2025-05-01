# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required libraries.
2.Upload and read the dataset.
3.Check for any null values using the isnull() function.
4.From sklearn.tree import DecisionTreeClassifier and use criterion as entropy.
5.Find the accuracy of the model and predict the required values by importing the required module from sklearn.

## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: KUKKADAPU CHARAN TEJ
RegisterNumber: 212224040167
*/
```
```py
import pandas as pd
data=pd.read_csv("Employee.csv")
data.head()

data.info()

data.isnull().sum()

data['left'].value_counts()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data['salary']=le.fit_transform(data['salary'])
data.head()

x=data[['satisfaction_level','last_evaluation','number_project','average_montly_hours','time_spend_company','Work_accident','promotion_last_5years','salary']]
x.head()

y=data['left']

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)
from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion='entropy')
dt.fit(x_train,y_train)
y_predict=dt.predict(x_test)

from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_predict)
accuracy

dt.predict([[0.5,0.8,9,206,6,0,1,2]])
```

## Output:
## Data Head:
![8 1 ml](https://github.com/user-attachments/assets/f3e43481-b3c2-432a-9f37-b930b54d7a60)


## Dataset info :
![8 2 ml](https://github.com/user-attachments/assets/a64fb8db-f0e4-46f9-8f03-6c3f067d72dd)


## Null Dataset:
![8 3 ml](https://github.com/user-attachments/assets/0310f5a1-2644-470a-8619-9dde7054af68)


## Values count in left column:
![8 4 ml](https://github.com/user-attachments/assets/68c0e605-323f-43a6-8e80-baf0ce0889da)


## Dataset transformed head:
![8 5 ml](https://github.com/user-attachments/assets/c9e22592-e4f6-4b59-91f9-fc9f29deecce)


## x.head:
![8 6 ml](https://github.com/user-attachments/assets/13397803-2f82-47f2-8873-ec5d94dbfbc8)


## Accuracy:
![8 7 ml](https://github.com/user-attachments/assets/ffdfd215-f829-4300-85f1-f2104b4d9e6e)


## Data prediction:
![8 8 ml](https://github.com/user-attachments/assets/70d12a62-bb3f-4d9a-a709-dd64d67c1f9c)

## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
