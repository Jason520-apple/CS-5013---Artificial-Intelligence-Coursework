# Task 1. Evaluate the Impact of Layer Size on Classification Accuracy

import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

data = np.loadtxt('diabetes.csv', delimiter=',', skiprows=1)
[n,p] = np.shape(data)

# training data 
num_train = int(0.5*n)
sample_train = data[0:num_train,0:-1]
label_train = data[0:num_train,-1]
# testing data 
num_test = int(0.5*n)
sample_test = data[n-num_test:,0:-1]
label_test = data[n-num_test:,-1]

# ----------------------- #
# --- Hyper-Parameter --- #
# ----------------------- #
k_values = [1, 4, 16, 64, 256] # want to be able to observe under and overfitting
# m_values = [0,0,0,0,0]

er_train_k = []
er_test_k = []
for k in k_values: 
    
    # train a MLP classification model 
    model = MLPClassifier (
        hidden_layer_sizes=(k,) * 5, #5 layers, vary number of neurons per layer k
        max_iter = 2000, 
        random_state = 42 #fixed seed
    )

    model.fit(sample_train, label_train)
    
    # evaluate training error and testing error 
    
    # need to do 1 - accuracy_score

    pred_train = model.predict(sample_train)
    pred_test = model.predict(sample_test)

    er_train = 1 - accuracy_score(label_train, pred_train)
    er_test = 1 - accuracy_score(label_test, pred_test)
    er_train_k.append(er_train)
    er_test_k.append(er_test)
   
plt.figure()
plt.plot(k_values,er_train_k, label='Training Error')
plt.plot(k_values,er_test_k, label='Testing Error')
plt.xlabel('k value') # need to change it to "m" value for figure 2
plt.ylabel('Classification Error')
plt.legend()
plt.xscale('log') #to spread out points for better visibility
plt.xticks(k_values, k_values)
plt.show()
