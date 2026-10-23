import numpy as np
import matplotlib.pyplot as plt
 
# --- Your Task --- #
# import necessary library 
# if you need more libraries, just import them
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import roc_auc_score
# --- end of task --- #
 
# load an imbalanced data set 
# there are 50 positive class instances 
# there are 500 negative class instances 
data = np.loadtxt('diabetes_new.csv', delimiter=',', skiprows=1)
[n,p] = np.shape(data)
 
# always use last 25% data for testing 
num_test = int(0.25*n)
sample_test = data[n-num_test:,0:-1]
label_test = data[n-num_test:,-1]
 
# vary the percentage of data for training
num_train_per = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8]

hyper_values = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100]
 
auc_hyper_per = []
 
for per in num_train_per: 
 
    # create training data and label
    num_train = int(n*per)
    sample_train = data[0:num_train,0:-1]
    label_train = data[0:num_train,-1]
 
    auc_list = []
    for C_yours in hyper_values:
 
        W = 5
        poly = PolynomialFeatures(2, include_bias=False)
        model_y = LogisticRegression(C=C_yours, class_weight={0: 1, 1: W}, max_iter=10000)
        model_y.fit(poly.fit_transform(sample_train), label_train)
        score_test = model_y.decision_function(poly.transform(sample_test))
 
        # evaluate model testing AUC score and stores it in "auc_yours"
        auc_yours = roc_auc_score(label_test, score_test)
        auc_list.append(auc_yours)
        # --- end of task --- #
 
    auc_hyper_per.append(auc_list)
 
# average AUC over all training sizes
auc_hyper = np.mean(auc_hyper_per, axis=0)
 
plt.figure()    
plt.plot(hyper_values,auc_hyper, label='Your AUC Score')
plt.xscale('log')
plt.xlabel('Hyper-parameter C')
plt.ylabel('Classification AUC Score')
plt.legend()
plt.show()