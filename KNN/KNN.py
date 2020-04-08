import sklearn
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np
from sklearn import linear_model,preprocessing

data = pd.read_csv("car.data")


le = preprocessing.LabelEncoder()

buying = le.fit_transform(list(data["buying"]))
door = le.fit_transform(list(data["door"]))
maint = le.fit_transform(list(data["maint"]))
persons = le.fit_transform(list(data["persons"]))
lug_boot = le.fit_transform(list(data["lug_boot"]))
safety = le.fit_transform(list(data["safety"]))
cls = le.fit_transform(list(data["class"]))


print(buying)
predict = "class"
X = list(zip(buying,maint,door,persons,lug_boot,safety))
Y = list(cls)

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, Y, test_size=0.1)

model = KNeighborsClassifier(n_neighbors=5)
model.fit(x_train,y_train)
acc = model.score(x_test,y_test)
print("accuracy:" + str(acc))