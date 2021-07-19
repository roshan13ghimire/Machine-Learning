#!/usr/bin/env python
# coding: utf-8

# In[1 a.]:missiing values


import pandas as pd
df = pd.read_csv("Data.csv")
df.fillna(0,inplace = False)
x= df.iloc[:,:2]
df.dropna()
x


# In[1 b.]:selecting meaningful features


from sklearn import preprocessing
min_max_scaler = preprocessing.MinMaxScaler(feature_range =(0,1))
#scalar feature
x_after_min_max_scaler = min_max_scaler.fit_transform(x)
print ("\nafter min max scaling :\n",x_after_min_max_scaler)
import pandas as pd
import numpy as nm
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
data = pd.read_csv("train.csv")
X = data.iloc[:, 0:20] #independent columns
y = data.iloc[:,-1] #target column i.e price range
#apply SelectKBest class to extract top 10 best features
bestfeatures = SelectKBest(score_func=chi2, k=10)
fit = bestfeatures.fit(X,y)
dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(X.columns)
#concat two dataframes for better visualization
featureScores = pd.concat([dfcolumns,dfscores],axis=1)
featureScores.columns = ['Specs', 'Score'] #naming the data frames columns
print(featureScores.nlargest(10,'Score')) #print 10 best features



# In[2 a.]:k fold


from numpy import array
from sklearn.model_selection import KFold
data = array([0.1,0.2,0.3,0.4,0.5,0.6])
kfold = KFold(3,True)
for train, test in kfold.split(data):
    print('train: %s, test: %s'%(data[train],data[test]))
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR
import numpy as np
dataset = pd.read_csv('housing1.csv')
print(dataset.shape)
(505, 14)
x=dataset.iloc[:,[0,12]]
y=dataset.iloc[:,13]
scaler = MinMaxScaler(feature_range=(0,1))
x=scaler.fit_transform(x)
scores= []
best_svr= SVR(kernel='rbf')
cv = KFold(n_splits=10, random_state=42, shuffle=False)
for train_index, test_index in cv.split(x):
    print("Train Index:",train_index, "\n")
    print("Test Index: ",test_index)
x_train,x_test, y_train, y_test = x[train_index],x[test_index],y[train_index],y[test_index]
best_svr.fit(x_train, y_train)
scores.append(best_svr.score(x_test, y_test))
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_digits
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import validation_curve
# Loading dataset
dataset = load_digits()
# X contains the data and y contains the labels
X, y = dataset.data, dataset.target
# Setting the range for the paramter (from 1 to 10)
parameter_range = np.arange(1, 10, 1)
# Calculate accuracy on training and test set using the
# gamma parameter with 5-fold cross validation
train_score, test_score = validation_curve(KNeighborsClassifier(), X, y, param_name = "n_neighbors",
param_range = parameter_range,cv = 5, scoring = "accuracy")
# Calculating mean and standard deviation of training score
mean_train_score = np.mean(train_score, axis = 1)
print(mean_train_score)
std_train_score = np.std(train_score, axis = 1)
# Calculating mean and standard deviation of testing score
mean_test_score = np.mean(test_score, axis = 1)
std_test_score = np.std(test_score, axis = 1)
# Plot mean accuracy scores for training and testing scores
plt.plot(parameter_range, mean_train_score,
label = "Training Score", color = 'b')
plt.plot(parameter_range, mean_test_score,
label = "Cross Validation Score", color = 'g')
# Creating the plot
plt.title("Validation Curve with KNN Classifier")
plt.xlabel("Number of Neighbours")
plt.ylabel("Accuracy")
plt.tight_layout()
plt.legend(loc = 'best')
plt.show()
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_digits
from sklearn.svm import SVC
from sklearn.model_selection import validation_curve
X, y = load_digits(return_X_y=True)
param_range = np.logspace(-6, -1, 5)
train_scores, test_scores = validation_curve(
SVC(), X, y, param_name="gamma", param_range=param_range,
scoring="accuracy", n_jobs=1)
train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)
plt.title("Validation Curve with SVM")
plt.xlabel(r"$\gamma$")
plt.ylabel("Score")
plt.ylim(0.0, 1.1)
lw = 2
plt.semilogx(param_range, train_scores_mean, label="Training score",
color="darkorange", lw=lw)
plt.fill_between(param_range, train_scores_mean - train_scores_std,
train_scores_mean + train_scores_std, alpha=0.2,
color="darkorange", lw=lw)
plt.semilogx(param_range, test_scores_mean, label="Cross-validation score",
color="navy", lw=lw)
plt.fill_between(param_range, test_scores_mean - test_scores_std,
test_scores_mean + test_scores_std, alpha=0.2,
color="navy", lw=lw)
plt.legend(loc="best")
plt.show()


# In[2 b.]:grod search


#grid search
#https://stackabuse.com/cross-validation-and-grid-search-for-model-selection-in-python/
import pandas as pd
import numpy as np
dataset = pd.read_csv("wineQualityReds.csv", sep=',')
dataset.head()
X = dataset.iloc[:, 0:11].values
y = dataset.iloc[:, 11].values
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=10, random_state=0)
from sklearn.preprocessing import StandardScaler
feature_scaler = StandardScaler()
X_train = feature_scaler.fit_transform(X_train)
X_test = feature_scaler.transform(X_test)
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators=300, random_state=0)
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators=300, random_state=0)
from sklearn.model_selection import cross_val_score
all_accuracies = cross_val_score(estimator=classifier, X=X_train, y=y_train, cv=5)
print(all_accuracies)
grid_param = {'n_estimators': [100, 300, 500, 800, 1000],'criterion': ['gini', 'entropy'], 'bootstrap': [True, False]}
from sklearn.model_selection import GridSearchCV
gd_sr = GridSearchCV(estimator=classifier,param_grid=grid_param, scoring='accuracy',cv=5,n_jobs=-1)
gd_sr.fit(X_train, y_train)
best_parameters = gd_sr.best_params_
print(best_parameters)
best_result = gd_sr.best_score_
print(best_result)


# In[3]:pca


import numpy as np
import pandas as pd
#Importing the Dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'Class']
dataset = pd.read_csv(url, names=names)
print(dataset)
X = dataset.iloc[:, 0:4].values
y = dataset.iloc[:, 4].values
#The following code divides data into training and test sets:
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
#Feature Scaling
#As was the case with PCA, we need to perform feature scaling for LDA too. Execute the following script to do so:
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
iris = datasets.load_iris()
X = iris.data
y = iris.target
target_names = iris.target_names
pca = PCA(n_components=2)
X_r = pca.fit(X).transform(X)
lda = LinearDiscriminantAnalysis(n_components=2)
X_r2 = lda.fit(X, y).transform(X)
# Percentage of variance explained for each components
print('explained variance ratio (first two components): %s'% str(pca.explained_variance_ratio_))
plt.figure()
colors = ['navy', 'turquoise', 'darkorange']
lw = 2
for color, i, target_name in zip(colors, [0, 1, 2], target_names):
    plt.scatter(X_r[y == i, 0], X_r[y == i, 1], color=color, alpha=.8, lw=lw,label=target_name)
plt.legend(loc='best', shadow=False, scatterpoints=1)
plt.title('PCA of IRIS dataset')
plt.figure()
for color, i, target_name in zip(colors, [0, 1, 2], target_names):
    plt.scatter(X_r2[y == i, 0], X_r2[y == i, 1], alpha=.8, color=color,label=target_name)
plt.legend(loc='best', shadow=False, scatterpoints=1)
plt.title('LDA of IRIS dataset')
plt.show()


# In[4]:data clustering


from sklearn import datasets
wine=datasets.load_wine()
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import seaborn as sns; sns.set() # for plot styling
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
X, y_true = make_blobs(n_samples=300, centers=4,
cluster_std=0.60, random_state=0)
plt.scatter(X[:, 0], X[:, 1], s=50)
kmeans = KMeans(n_clusters=4)
kmeans.fit(X)
y_kmeans = kmeans.predict(X)
plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=50, cmap='viridis')
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(wine.data,wine.target,test_size=0.3)
from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train,Y_train)
y_pred=knn.predict(X_test)
from sklearn import metrics
metrics.accuracy_score(Y_test,y_pred)
from sklearn.metrics import confusion_matrix,classification_report
from keras.datasets import mnist
(X_train,y_train),(X_test,y_test)=mnist.load_data()
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense,Conv2D,MaxPool2D,Dropout
from keras.utils import np_utils
X_train=X_train.reshape(60000,784)
X_test=X_test.reshape(10000,784)
X_train=X_train.astype('float32')
X_test=X_test.astype('float32')
X_train/=255
X_test/=255
n_classes=10
Y_train=np_utils.to_categorical(y_train,n_classes);
Y_test=np_utils.to_categorical(y_test,n_classes);
model=Sequential()
model.add(Dense(100,input_shape=(784,),activation='relu'))
model.add(Dense(10,activation='softmax'))
model.summary()
model.compile(loss='categorical_crossentropy',metrics=['accuracy'],optimizer='adam')
history=model.fit(X_train,Y_train,batch_size=128,epochs=10,validation_data=(X_test,Y_test))


# In[5]:classification model


from sklearn import datasets
#Load dataset
wine = datasets.load_wine()
print(wine.feature_names)
print(wine.target_names)
print(wine.data[0:5])
print(wine.target)
print(wine.data.shape)
# print target(or label)shape
print(wine.target.shape)
# Import train_test_split function
from sklearn.model_selection import train_test_split
# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(wine.data, wine.target, test_size=0.3) # 70% training and 30% test
from sklearn.neighbors import KNeighborsClassifier
#Create KNN Classifier
knn = KNeighborsClassifier(n_neighbors=5)
#Train the model using the training sets
knn.fit(X_train, y_train)
#Predict the response for test dataset
y_pred = knn.predict(X_test)
from sklearn import metrics
# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
from sklearn.neighbors import KNeighborsClassifier
#Create KNN Classifier
knn = KNeighborsClassifier(n_neighbors=7)
#Train the model using the training sets
knn.fit(X_train, y_train)
#Predict the response for test dataset
y_pred = knn.predict(X_test)
from sklearn import metrics
# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import seaborn as sns; sns.set() # for plot styling
import numpy as np
from sklearn.datasets import make_blobs
X, y_true = make_blobs(n_samples=300, centers=4,
cluster_std=0.60, random_state=0)
plt.scatter(X[:, 0], X[:, 1], s=50)
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=4)
kmeans.fit(X)
y_kmeans = kmeans.predict(X)
plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=50, cmap='viridis')
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5);


# In[6]: Vector Addition


import numpy as NP
A = [4, 8, 7]
B = [5, -4, 8]
print("The input arrays are :\n","A:",A ,"\n","B:",B)
Res= NP.add(A,B)
print("After addition the resulting array is :",Res)










