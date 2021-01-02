# Importing the libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
#get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


# Importing the dataset
train = pd.read_csv('train.csv').drop('Unnamed: 0', axis = 1)


# In[3]:


# For each column heading we replace "-" and convert the heading in lowercase 
cleancolumn = []
for i in range(len(train.columns)):
    cleancolumn.append(train.columns[i].replace('-', '').lower())
train.columns = cleancolumn


# In[4]:


train.head()


# In[5]:


train[train.columns[1:]].describe()


# In[6]:


train[1:].isna().tail(10)


# In[7]:


# This give you the calulation of the target lebels. Which category of the target lebel is how many percentage.
total_len = len(train['seriousdlqin2yrs'])
percentage_labels = (train['seriousdlqin2yrs'].value_counts()/total_len)*100
percentage_labels


# In[9]:


# Graphical representation of the target label percentage.
sns.set()
sns.countplot(train.seriousdlqin2yrs).set_title('Data Distribution')
ax = plt.gca()
for p in ax.patches:
    height = p.get_height()
    ax.text(p.get_x() + p.get_width()/2.,
            height + 2,
            '{:.2f}%'.format(100*(height/total_len)),
            fontsize=14, ha='center', va='bottom')
sns.set(font_scale=2)
sns.set(rc={'figure.figsize':(8,8)})
ax.set_xlabel("Labels for seriousdlqin2yrs attribute")
ax.set_ylabel("Numbers of records")
plt.show()


# ## Missing Values

# In[10]:


# You will get to know which column has missing value and it's give the count that how many records are missing 
train.isnull().sum()


# In[11]:


# Graphical representation of the missing values.
x = train.columns
y = train.isnull().sum()
sns.set()
sns.barplot(x,y)
ax = plt.gca()
for p in ax.patches:
    height = p.get_height()
    ax.text(p.get_x() + p.get_width()/2.,
            height + 2,
            int(height),
            fontsize=14, ha='center', va='bottom')
sns.set(font_scale=1.5)
sns.set(rc={'figure.figsize':(8,8)})
ax.set_xlabel("Data Attributes")
ax.set_ylabel("count of missing records for each attribute")
plt.xticks(rotation=90)
plt.show()


# In[12]:


# Actual replacement of the missing value using mean value.
train_mean = train.fillna((train.mean()))
train_mean.head()


# In[13]:


train_mean.isnull().sum()


# In[14]:


# Actual replacement of the missing value using median value.
train_median = train.fillna((train.median()))
train_median.head()


# In[15]:


train_median.isnull().sum()


# # Correlation

# In[33]:


train.fillna((train.median()), inplace=True)
# Get the correlation of the training dataset
correlation=train[train.columns[1:]].corr()
correlation


# In[20]:


sns.set()
sns.set(font_scale=1.25)
sns.heatmap(train[train.columns[1:]].corr(),annot=True,fmt=".1f")
sns.set(rc={'figure.figsize':(10,10)})
plt.show()


# In[23]:


train.columns


# # Feature Selection
# 

# In[38]:


X=train.drop('seriousdlqin2yrs',axis=1)
y=train.seriousdlqin2yrs


# In[37]:


train.columns[1:]


# In[39]:


features_label = train.columns[1:]


# In[42]:


#Fitting Random Forest Classification to the Training set
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 200, criterion = 'entropy', random_state = 0)
classifier.fit(X, y)
importances = classifier.feature_importances_
indices = np. argsort(importances)[::-1]
for i in range(X.shape[1]):
    print ("%2d) %-*s %f" % (i + 1, 30, features_label[i],importances[indices[i]]))


# In[43]:


plt.title('Feature Importances')
plt.bar(range(X.shape[1]),importances[indices], color="green", align="center")
plt.xticks(range(X.shape[1]),features_label, rotation=90)
plt.xlim([-1, X.shape[1]])
plt.show()


# 
# # Train and build baseline model

# #The algorithms that we are going to choose are as follows (this selection is based on intuition):
# 
# RandomForest
# 
# Logistic Regression
# 
# K-Nearest Neighbor (KNN)
# 
# AdaBoost
# 
# GradientBoosting
# 

# In[187]:


from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score


# In[188]:


X = train.drop('seriousdlqin2yrs', axis=1)
y = train.seriousdlqin2yrs


# In[189]:


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection  import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 42)


# In[190]:


# Fitting Random Forest Classification to the Training set
from sklearn.ensemble import RandomForestClassifier
Random = RandomForestClassifier(n_estimators=10, criterion='gini', max_depth=None, min_samples_split=2,
                               min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='auto',
                               max_leaf_nodes=None, bootstrap=True, oob_score=False, n_jobs=1, 
                               random_state=None, verbose=0)
Random.fit(X_train, y_train)


# In[191]:


# Predicting the Test set results
y_pred = Random.predict(X_test)

# Making the Confusion Matrix 
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
np.set_printoptions(precision=2)


# In[192]:


cm


# In[193]:


#Let's see how our model performed
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))


# In[194]:


test_labels=Random.predict_proba(np.array(X_test.values))[:,1]
roc_auc_score(y_test,test_labels , average='macro', sample_weight=None)


# In[195]:


import matplotlib.pyplot as plt
import itertools

from sklearn import svm, datasets
from sklearn.metrics import confusion_matrix

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()


# In[196]:


# Plot normalized confusion matrix
plot_confusion_matrix(cm,classes=[0,1])
sns.set(rc={'figure.figsize':(6,6)})
plt.show()


# In[197]:


# Predicting the Test set results
y_pred = Random.predict(X)

# Making the Confusion Matrix w
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y, y_pred.round())
np.set_printoptions(precision=2)


# In[198]:


# Plot non-normalized confusion matrix
plot_confusion_matrix(cm,classes=[0,1])
sns.set(rc={'figure.figsize':(6,6)})
plt.show()


# In[199]:


# Fitting Logistic Regression to the Training set
from sklearn.linear_model import LogisticRegression
Logistic = LogisticRegression(penalty='l1', dual=False, tol=0.0001, C=1.0, fit_intercept=True,
                            intercept_scaling=1, class_weight=None, 
                            random_state=None, solver='liblinear', max_iter=100,
                            multi_class='ovr', verbose=2)
Logistic.fit(X_train, y_train)


# In[200]:


# Predicting the Test set results
y_pred = Logistic.predict(X_test)

# Making the Confusion Matrix 
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
np.set_printoptions(precision=2)


# In[201]:


cm


# In[202]:


#Let's see how our model performed
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))


# In[203]:


test_labels=Logistic.predict_proba(np.array(X_test.values))[:,1]
roc_auc_score(y_test,test_labels , average='macro', sample_weight=None)


# In[204]:


# Plot normalized confusion matrix
plot_confusion_matrix(cm,classes=[0,1])
sns.set(rc={'figure.figsize':(6,6)})
plt.show()


# In[205]:


# Predicting the Test set results
y_pred = Logistic.predict(X)

# Making the Confusion Matrix w
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y, y_pred.round())
np.set_printoptions(precision=2)


# In[206]:


# Plot non-normalized confusion matrix
plot_confusion_matrix(cm,classes=[0,1])
sns.set(rc={'figure.figsize':(6,6)})
plt.show()


# In[207]:


# Fitting K-NN to the Training set
from sklearn.neighbors import KNeighborsClassifier
KNN = KNeighborsClassifier(n_neighbors=5, weights='uniform', algorithm='auto', leaf_size=30, p=2,
                             metric='minkowski', metric_params=None)
KNN.fit(X_train, y_train)


# In[208]:


# Predicting the Test set results
y_pred = KNN.predict(X_test)

# Making the Confusion Matrix 
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
np.set_printoptions(precision=2)


# In[209]:


cm


# In[210]:


#Let's see how our model performed
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))


# In[211]:


test_labels=KNN.predict_proba(np.array(X_test.values))[:,1]
roc_auc_score(y_test,test_labels , average='macro', sample_weight=None)


# In[212]:


# Plot normalized confusion matrix
plot_confusion_matrix(cm,classes=[0,1])
sns.set(rc={'figure.figsize':(6,6)})
plt.show()


# In[213]:


# Predicting the Test set results
y_pred = KNN.predict(X)

# Making the Confusion Matrix w
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y, y_pred.round())
np.set_printoptions(precision=2)


# In[214]:


# Plot non-normalized confusion matrix
plot_confusion_matrix(cm,classes=[0,1])
sns.set(rc={'figure.figsize':(6,6)})
plt.show()


# In[215]:


# Fitting Ada-boost to the Training set
from sklearn.neighbors import KNeighborsClassifier
ADA = AdaBoostClassifier(base_estimator=None, n_estimators=200, learning_rate=1.0)
ADA.fit(X_train, y_train)


# In[216]:


# Predicting the Test set results
y_pred = ADA.predict(X_test)

# Making the Confusion Matrix 
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
np.set_printoptions(precision=2)
cm


# In[217]:


#Let's see how our model performed
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))


# In[218]:


test_labels=ADA.predict_proba(np.array(X_test.values))[:,1]
roc_auc_score(y_test,test_labels , average='macro', sample_weight=None)


# In[219]:


# Plot normalized confusion matrix
plot_confusion_matrix(cm,classes=[0,1])
sns.set(rc={'figure.figsize':(6,6)})
plt.show()


# In[220]:


# Fitting GradientBoosting to the Training set
from sklearn.neighbors import KNeighborsClassifier
GradientBoo = GradientBoostingClassifier(loss='deviance', learning_rate=0.1, n_estimators=200, subsample=1.0,
                                   min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, 
                                   max_depth=3,
                                   init=None, random_state=None, max_features=None, verbose=0)
GradientBoo.fit(X_train, y_train)


# In[221]:


# Predicting the Test set results
y_pred = GradientBoo.predict(X_test)

# Making the Confusion Matrix 
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
np.set_printoptions(precision=2)
cm


# In[222]:


#Let's see how our model performed
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))


# In[223]:


test_labels=GradientBoo.predict_proba(np.array(X_test.values))[:,1]
roc_auc_score(y_test,test_labels , average='macro', sample_weight=None)


# In[224]:


# Plot normalized confusion matrix
plot_confusion_matrix(cm,classes=[0,1])
sns.set(rc={'figure.figsize':(6,6)})
plt.show()


# In[226]:


test_labels=classifier.predict_proba(np.array(X_test.values))[:,1]
roc_auc_score(y_test,test_labels , average='macro', sample_weight=None)


# In[227]:


from sklearn.model_selection import cross_val_score
def cvDictGen(functions, scr, X_train=X, y_train=y, cv=5, verbose=1):
    cvDict = {}
    for func in functions:
        cvScore = cross_val_score(func, X_train, y_train, cv=cv, verbose=verbose, scoring=scr)
        cvDict[str(func).split('(')[0]] = [cvScore.mean(), cvScore.std()]
    
    return cvDict

def cvDictNormalize(cvDict):
    cvDictNormalized = {}
    for key in cvDict.keys():
        for i in cvDict[key]:
            cvDictNormalized[key] = ['{:0.2f}'.format((cvDict[key][0]/cvDict[cvDict.keys()[0]][0])),
                                     '{:0.2f}'.format((cvDict[key][1]/cvDict[cvDict.keys()[0]][1]))]
    return cvDictNormalized


# In[228]:


cvD = cvDictGen(functions=[Random, Logistic, KNN, ADA, GradientBoo], scr='roc_auc')
cvD


# # Hyper parameter optimization using Randomized search

# In[230]:


from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint


# ADA Boosting
# 

# In[232]:


adaHyperParams = {'n_estimators': [10,50,100,200,420]}
gridSearchAda = RandomizedSearchCV(estimator=ADA, param_distributions=adaHyperParams, n_iter=5,
                                   scoring='roc_auc', fit_params=None, cv=None, verbose=2).fit(X_train, y_train)
gridSearchAda.best_params_, gridSearchAda.best_score_


# Gradient Boosting

# In[233]:


gbHyperParams = {'loss' : ['deviance', 'exponential'],
                 'n_estimators': randint(10, 500),
                 'max_depth': randint(1,10)}
gridSearchGB = RandomizedSearchCV(estimator=GradientBoo, param_distributions=gbHyperParams, n_iter=10,
                                   scoring='roc_auc', fit_params=None, cv=None, verbose=2).fit(X_train, y_train)
gridSearchGB.best_params_, gridSearchGB.best_score_


# # Train models with help of new hyper parameter

# In[234]:


#Fitting both ADA and Gradient
bestGbModFitted = gridSearchGB.best_estimator_.fit(X_train, y_train)
bestAdaModFitted = gridSearchAda.best_estimator_.fit(X_train, y_train)


# In[235]:


cvDictbestpara = cvDictGen(functions=[bestGbModFitted, bestAdaModFitted], scr='roc_auc')
cvDictbestpara


# In[236]:


test_labels=bestGbModFitted.predict_proba(np.array(X_test.values))[:,1]
roc_auc_score(y_test,test_labels , average='macro', sample_weight=None)


# In[237]:


test_labels=bestAdaModFitted.predict_proba(np.array(X_test.values))[:,1]
roc_auc_score(y_test,test_labels , average='macro', sample_weight=None)


# # Feature Transformation

# In[240]:


import numpy as np
from sklearn.preprocessing import FunctionTransformer

transformer = FunctionTransformer(np.log1p)
X_train_1 = np.array(X_train)
X_train_transform = transformer.transform(X_train_1)


# In[244]:


bestGbModFitted_transformed = gridSearchGB.best_estimator_.fit(X_train_transform, y_train)
bestAdaModFitted_transformed = gridSearchAda.best_estimator_.fit(X_train_transform, y_train)

cvDictbestpara_transform = cvDictGen(functions=[bestGbModFitted_transformed, bestAdaModFitted_transformed],
                                     scr='roc_auc')
cvDictbestpara_transform


# In[245]:


import numpy as np
from sklearn.preprocessing import FunctionTransformer

transformer = FunctionTransformer(np.log1p)
X_test_1 = np.array(X_test)
X_test_transform = transformer.transform(X_test_1)
X_test_transform


# In[246]:


test_labels=bestGbModFitted_transformed.predict_proba(np.array(X_test_transform))[:,1]
roc_auc_score(y_test,test_labels , average='macro', sample_weight=None)


# In[247]:


test_labels=bestAdaModFitted_transformed.predict_proba(np.array(X_test_transform))[:,1]
roc_auc_score(y_test,test_labels , average='macro', sample_weight=None)


# # Voting based ensamble model

# In[248]:


from sklearn.ensemble import VotingClassifier
votingMod = VotingClassifier(estimators=[('gb', bestGbModFitted_transformed), 
                                         ('ada', bestAdaModFitted_transformed)], voting='soft',weights=[2,1])
votingMod = votingMod.fit(X_train_transform, y_train)


# In[249]:


test_labels=votingMod.predict_proba(np.array(X_test_transform))[:,1]
votingMod.score(X_test_transform, y_test)


# In[250]:


roc_auc_score(y_test,test_labels , average='macro', sample_weight=None)


# In[253]:


#without transform voting
from sklearn.ensemble import VotingClassifier
votingMod_old = VotingClassifier(estimators=[('gb', bestGbModFitted), ('ada', bestAdaModFitted)], 
                                 voting='soft',weights=[2,1])
votingMod_old = votingMod.fit(X_train, y_train)


# In[254]:


test_labels=votingMod_old.predict_proba(np.array(X_test.values))[:,1]


# In[255]:


roc_auc_score(y_test,test_labels , average='macro', sample_weight=None)


# # Testing on Real Test Dataset

# In[294]:


# Read Training dataset as well as drop the index column
test = pd.read_csv('test.csv').drop('Unnamed: 0', axis = 1)
# For each column heading we replace "-" and convert the heading in lowercase 
cleancolumn = []
for i in range(len(test.columns)):
    cleancolumn.append(test.columns[i].replace('-', '').lower())
test.columns = cleancolumn


# In[295]:



test.head()


# In[296]:


test.drop(['seriousdlqin2yrs'], axis=1, inplace=True)


# In[297]:


test.fillna((train_median.median()), inplace=True)


# In[298]:


test.head()


# In[299]:


test_labels_votingMod_old = votingMod_old.predict_proba(np.array(test.values))[:,1]
print (len(test_labels_votingMod_old))


# In[300]:



output = pd.DataFrame({'ID':test.index, 'probability':test_labels_votingMod_old})


# In[301]:


output.to_csv("./predictions.csv", index=False)


# In[302]:


import numpy as np
from sklearn.preprocessing import FunctionTransformer

transformer = FunctionTransformer(np.log1p)
test_data_temp = np.array(test)
test_data_transform = transformer.transform(test_data_temp)


# In[303]:



test_labels_votingMod = votingMod.predict_proba(np.array(test.values))[:,1]
print (len(test_labels_votingMod_old))


# In[304]:


output = pd.DataFrame({'ID':test.index, 'probability':test_labels_votingMod})
output.to_csv("./predictions_voting_Feature_transformation.csv", index=False)


# In[ ]:




