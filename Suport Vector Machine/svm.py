import sklearn
from sklearn import datasets
from sklearn import svm
from sklearn import metrics


cancer = datasets.load_breast_cancer()

x = cancer["data"]
y = cancer["target"]

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x,y, test_size=0.4)


model = svm.SVC(kernel="linear",C=10000)
model.fit(x_train,y_train)

acc = model.score(x_test,y_test)
print(acc)



