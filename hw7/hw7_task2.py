# Task 2. Evaluate the Impact of Layer Depth on Classification Accuracy

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
k = 16 #for task 2, have fixed neurons per layer rather than fixed layers m
m_values = [1, 3, 5, 10, 20]  # varying number of m layers

er_train_k = []
er_test_k = []
for m in m_values: 
    
    # train a MLP classification model 
    model = MLPClassifier (
        hidden_layer_sizes=(k,) * m, #m layers with fixed k neurons per layer
        max_iter=2000,
        random_state=42 #fixed seed
    )

    model.fit(sample_train, label_train)
    
    # evaluate training error and testing error 
    
    pred_train = model.predict (sample_train)
    pred_test = model.predict (sample_test)

    er_train = 1 - accuracy_score(label_train, pred_train)
    er_test = 1 - accuracy_score(label_test, pred_test)
    er_train_k.append(er_train)
    er_test_k.append(er_test)
   
plt.figure()
plt.plot(m_values,er_train_k, label='Training Error')
plt.plot(m_values,er_test_k, label='Testing Error')
plt.xlabel('m value') # need to change it to "m" value for figure 2
plt.ylabel('Classification Error')
plt.legend()
plt.show()


