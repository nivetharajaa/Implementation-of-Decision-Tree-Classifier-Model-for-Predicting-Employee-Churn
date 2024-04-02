# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. import pandas module and import the required data set.
2.Find the null values and count them.
3.Count number of left values.
4.From sklearn import LabelEncoder to convert string values to numerical values.
5.From sklearn.model_selection import train_test_split.
6.Assign the train dataset and test dataset.
7.From sklearn.tree import DecisionTreeClassifier.
8.Use criteria as entropy.
9.From sklearn import metrics.
10.Find the accuracy of our model and predict the require values. 

## Program:
```

Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by:Nivetha A 
RegisterNumber:212222230101 

import pandas as pd
data=pd.read_csv("/content/Employee (1).csv")
data.head()

data.info()

data.isnull().sum()

data['left'].value_counts()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()

data['salary']=le.fit_transform(data['salary'])
data.head()

x = data[['satisfaction_level','last_evaluation','number_project','average_montly_hours','time_spend_company','Work_accident','promotion_last_5years','salary']]
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

dt.predict([[0.5,0.8,9,260,6,0,1,2]])

```

## Output:

![image](https://github.com/nivetharajaa/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/120543388/64462bca-ef79-4a6a-9001-429237ec561d)


![image](https://github.com/nivetharajaa/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/120543388/051219ca-58dd-43b1-bc4a-078a329a9378)


![image](https://github.com/nivetharajaa/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/120543388/ce61c6ae-0526-4e7e-bbd3-a35ca856ed5a)


![image](https://github.com/nivetharajaa/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/120543388/b2c9a12f-34f6-402b-bd1c-3b822e0c4232)


![image](https://github.com/nivetharajaa/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/120543388/7ea26869-f848-447e-a739-1078421e5b0c)


![image](https://github.com/nivetharajaa/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/120543388/b1086ed9-648f-49f9-a418-014357262c81)


![image](https://github.com/nivetharajaa/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/120543388/fe578dda-a511-44e5-afa5-0ad37776fe99)


![image](https://github.com/nivetharajaa/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/120543388/4a8a571a-fcad-41b0-85a4-ef45a7f15250)

## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
