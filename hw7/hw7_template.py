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
k_values = [0,0,0,0,0]
# m_values = [0,0,0,0,0]

er_train_k = []
er_test_k = []
for k in k_values: 
    
    # train a MLP classification model 
    # ......        
    # ......
    # ......
    
    # evaluate training error and testing error 
    # ......
    # ......
    # ......
    # er_train = ...
    # er_test = ...
    er_train_k.append(er_train)
    er_test_k.append(er_test)
   
plt.figure()
plt.plot(k_values,er_train_k, label='Training Error')
plt.plot(k_values,er_test_k, label='Testing Error')
plt.xlabel('k value') # need to change it to "m" value for figure 2
plt.ylabel('Classification Error')
plt.legend()


