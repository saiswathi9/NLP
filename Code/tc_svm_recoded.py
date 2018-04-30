
# coding: utf-8

# In[3]:


import pandas as pd
import itertools
from sklearn import svm    
import numpy as np
#import matplotlib.pyplot as plt 
from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import time
start_time = time.time()
#reading embedded datasets
bot_s=pd.read_excel('/home/sss26x/NLP/Data/recoded_botcompiledscores.xlsx').as_matrix()
nonbot_s=pd.read_excel('/home/sss26x/NLP/Data/recoded_nonbotcompiledscores.xlsx').as_matrix()
#bot_score=pd.read_excel('C:/Users/sivar/Desktop/recoded_results/recoded_nonbotscoring.xlsx').as_matrix()
#nonbot_score=pd.read_excel('C:/Users/sivar/Desktop/recoded_results/recoded_botscoringv2.xlsx').as_matrix()
#bot = np.concatenate((bot_score,bot_s))
#nonbot = np.concatenate((nonbot_score,nonbot_s))
bot_test = [1]*len(bot_s) + [0]*len(nonbot_s)
a = bot_test == 1
b = bot_test == 0
y = bot_test.copy()
y[a] = 0
y[b] = 1
X=np.concatenate((bot_s,nonbot_s))
#print(rev_label)
#y=X[:,[3]]
class_names = ['bot','nonbot']

#splitting train and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
C=1.0

#classifier
#svc = svm.LinearSVC(C=C).fit(X_train, y_train)
#svc = svm.SVC(kernel='rbf', gamma=0.7, C=C).fit(X_train, y_train)
svc = svm.SVC(kernel='poly', degree=3, C=C,decision_function_shape='ovr').fit(X_train, y_train)
#svc = svm.SVC(kernel='linear', C=C).fit(X_train, y_train)

predictions = svc.predict(X_test)

#scatter plot
#plt.scatter(X[:,0], X[:,1],X[:,2], c = y)
#plt.title('bot vs nonbot')
#plt.show()
#plt.scatter(X, y);

#plt.plot(predictions, y_test, 'r.', markersize=10, label=u'Observations')
print("Detailed classification report:")
print()
print("The model is trained on the full development set.")
print("The scores are computed on the full evaluation set.")
print()
y_true, y_pred = y_test, predictions
print(classification_report(y_true, y_pred))
print()

#plotting confusion matrix

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix' ):
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

    #plt.imshow(cm, interpolation='nearest', cmap=cmap)
    #plt.title('recoded')
    #plt.colorbar()
    #tick_marks = np.arange(len(classes))
    #plt.xticks(tick_marks, classes, rotation=45)
    #plt.yticks(tick_marks, classes)


# Compute confusion matrix
cnf_matrix = confusion_matrix(y_test, predictions)
np.set_printoptions(precision=2)


print("end")
print("--- %s seconds ---" % (time.time() - start_time))

