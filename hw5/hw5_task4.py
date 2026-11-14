#
# Template for Task 4: kNN Classification 
#
import numpy as np
import matplotlib.pyplot as plt

# --- Your Task --- #
# import libraries as needed 
from scipy.spatial.distance import cdist #for distance function between neighbors
from sklearn.metrics import accuracy_score #compute test error
# --- end of task --- #

# -------------------------------------
# load data 
data = np.loadtxt('diabetes.csv', delimiter=',')
[n,p] = np.shape(data)
# 75% for training, 25% for testing 
num_train = int(0.75*n)
num_test = int(0.25*n)
sample_train = data[0:num_train,0:-1]
label_train = data[0:num_train,-1]
sample_test = data[n-num_test:,0:-1]
label_test = data[n-num_test:,-1]
# -------------------------------------


# --- Your Task --- #
# pick five values of k by yourself 
k_values = [1, 5, 19, 101, 401] #odd numbers to prevent ties
# --- end of task --- #

er_test = []
for k in k_values: 
    # --- Your Task --- #
    # implement the kNN classification method 

    # using formula provided in slides
    # works by calculating 
    label_pred = []
    
    for x in sample_test:
        # compute distance from input vector x to all training points x_i
        distances = cdist([x], sample_train, metric='euclidean')[0]
        
        # find N_k(x): indices of the neighborhood of k closest points to x
        Nk_x = np.argsort(distances)[:k]
        
        # get y_i for each x_i in N_k(x)
        y_i = label_train[Nk_x]
        
        # lecture formula: y_hat(x) = (1 / k) * sum(y_i)
        y_hat = (1.0 / k) * np.sum(y_i)
        
        # convert average into binary class prediction (1 if >= 0.5, else 0)
        label_pred.append(1 if y_hat >= 0.5 else 0)

    # store classification error on testing data here 
    er = 1 - accuracy_score(label_test, label_pred)
    er_test.append(er)
# --- end of task --- #
    
plt.figure()    
plt.plot(k_values, er_test, marker='o')
plt.xlabel('k')
plt.ylabel('Classification Error')
plt.show()