import pandas as pd 
import numpy as np 
dataset = pd.read_csv(r"C:\Users\my pc\OneDrive\Pictures\AI AND ML\machine learning\linearregression (DAY-3)\DigitalAd_dataset.csv")
print(dataset.shape)
print(dataset.head(5))
x = dataset.iloc[: ,:-1].values
x

y = dataset.iloc[:,-1].values
y
from sklearn.model_selection import train_test_split
X_train , X_test , Y_train , Y_test = train_test_split(x,y,test_size = 0.25,random_state=0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train,Y_train)

age = int(input("Enter new customer age: "))
sal = int(input("Enter new customer salary:"))
newcus = [[age,sal]]
result = model.predict (sc.transform(newcus))
print(result)
if result == 1:
    print("Customer will buy")
else:
    print("Customer will not buy")
y_pred = model.predict(X_test)
print(np.concatenate((y_pred.reshape(len(y_pred),1), Y_test.reshape(len(Y_test),1)),1))

from sklearn.metrics import accuracy_score
print("Accuracy of the model :- {0}% ".format (accuracy_score(Y_test,y_pred)*100))