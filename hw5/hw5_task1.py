#
# Template for Task 1: Linear Regression 
#
import numpy as np
import matplotlib.pyplot as plt

# --- Your Task --- #
# import libraries as needed 
# .......
# --- end of task --- #

# -------------------------------------
# load data 
data = np.loadtxt('crimerate.csv', delimiter=',')
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

X_train = np.hstack([sample_train, np.ones((num_train,1))])
X_test  = np.hstack([sample_test,  np.ones((num_test,1))])


# pick a proper number of iterations 
num_iter = 5000
# learning rate (step size)
lr = 0.03
# randomly initialize your w 
np.random.seed(0)
w = np.random.randn(X_train.shape[1]) * 0.01
# --- end of task --- #

er_test = []


# --- Your Task --- #
# implement the iterative learning algorithm for w
# at the end of each iteration, evaluate the updated w 
for iter in range(num_iter): 

    ## update w
    pre_train = X_train @ w  # model prediction
    grad = (2.0/num_train) * X_train.T @ (pre_train - label_train)   # d(MSE)/dw
    w = w - lr * grad

    ## evaluate testing error of the updated w 
    # we should measure mean-square-error here
    pre_test = X_test @ w
    er = np.mean((pre_test - label_test)**2)
    er_test.append(er)
# --- end of task --- #

print('final test MSE: %.5f' % er_test[-1])

plt.figure()    
plt.plot(er_test)
plt.xlabel('Iteration')
plt.ylabel('Classification Error')
plt.title('Figure 1: Linear Regression')
plt.show()