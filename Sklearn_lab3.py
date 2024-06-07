import sklearn
from sklearn import  datasets
from sklearn import model_selection
from sklearn import tree
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn import ensemble
from sklearn.model_selection import KFold
import numpy as np


dane = datasets.load_iris()
x = dane["data"]
y = dane["target"]




X_train,X_test, Y_train,Y_test = model_selection.train_test_split(x,y,train_size=0.8)


model = tree.DecisionTreeClassifier()

model.fit(X_train,Y_train)

y_pred = model.predict(X_test)

metrics.accuracy_score(Y_test,y_pred)
matrix = metrics.confusion_matrix(Y_test,y_pred)

display = metrics.ConfusionMatrixDisplay(confusion_matrix=matrix,display_labels=model.classes_)

tree.plot_tree(model)
display.plot()
plt.show()



dane_wina = datasets.load_wine()

x2 = dane_wina['data']
y2 = dane_wina['target']

X_train,X_test, Y_train,Y_test = model_selection.train_test_split(x2,y2,train_size=0.7)

model_linear = sklearn.svm.SVC(kernel='linear')
model_rbf = sklearn.svm.SVC(kernel="rbf")
model_poly = sklearn.svm.SVC(kernel='poly')

model_linear.fit(X_train,Y_train)
model_rbf.fit(X_train,Y_train)
model_poly.fit(X_train,Y_train)

y_pred_lin = model_linear.predict(X_test)
y_pred_rbf = model_rbf.predict(X_test)
y_pred_poly = model_poly.predict(X_test)


metrics.accuracy_score(Y_test,y_pred_lin)
metrics.accuracy_score(Y_test,y_pred_rbf)
metrics.accuracy_score(Y_test,y_pred_poly)



matrix_lin = metrics.confusion_matrix(Y_test,y_pred_lin)
matrix_rbf = metrics.confusion_matrix(Y_test,y_pred_rbf)
matrix_poly = metrics.confusion_matrix(Y_test,y_pred_poly)



display1 = metrics.ConfusionMatrixDisplay(confusion_matrix=matrix_lin,display_labels=model.classes_)
display2 = metrics.ConfusionMatrixDisplay(confusion_matrix=matrix_rbf,display_labels=model.classes_)
display3 = metrics.ConfusionMatrixDisplay(confusion_matrix=matrix_poly,display_labels=model.classes_)

display1.plot()
display2.plot()
display3.plot()

plt.show()





dane_domy = datasets.fetch_california_housing()


x3 = dane_domy['data']
y3 = dane_domy['target']



X_train,X_test, Y_train,Y_test = model_selection.train_test_split(x3,y3,train_size=0.75)


model_boost = ensemble.GradientBoostingRegressor()

kf = KFold(n_splits=10, shuffle=True, random_state=50)


print(model_selection.cross_val_score(model_boost,x3,y3,cv=kf))

model_boost.fit(X_train,Y_train)

y_pred = model_boost.predict(X_test)

print(metrics.r2_score(Y_test,y_pred))
print(metrics.mean_absolute_error(Y_test,y_pred))

x5 = model_boost.feature_importances_
feature_names = dane_domy['feature_names']

# Plot the bar chart
plt.bar(feature_names, x5)
plt.xticks(rotation=45)
plt.ylabel('Feature Importance')
plt.xlabel('Feature Name')
plt.title('Feature Importances')


plt.show()