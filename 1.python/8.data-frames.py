import os
import pandas as pd

os.getcwd()
os.chdir("C:\\Users\\Thimma Reddy\\Documents")

titanic_train = pd.read_csv("titanic.csv")
titanic_train.shape
titanic_train.Pclass = titanic_train.Pclass.astype('category')
titanic_train.info()
titanic_train.dtypes

titanic_train.head()
titanic_train.tail()
titanic_train.describe()


titanic_train1 = titanic_train[0:4]
titanic_train["Age"]
titanic_train.Age
titanic_train[["PassengerId","Fare"]]
titanic_train[titanic_train.Age>70]

titanic_train1.set_index('PassengerId')
titanic_train1.set_index('PassengerId', inplace=True)
titanic_train1.reset_index()
titanic_train1.reset_index(inplace=True)

titanic_train1.loc[2:]


