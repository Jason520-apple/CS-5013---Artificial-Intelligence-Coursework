#
# Template for Task 2: Logistic Regression 
#
import numpy as np
import matplotlib.pyplot as plt

# --- Your Task --- #
# import libraries as needed 
# .......
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
def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))

X_train = np.hstack([sample_train, np.ones((num_train,1))])
X_test  = np.hstack([sample_test,  np.ones((num_test,1))])

# pick a proper number of iterations 
num_iter = 10000
# learning rate (step size)
lr = 0.01
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
    h_train = sigmoid(X_train @ w)
    w = w + lr * X_train.T @ ((label_train - h_train) * h_train * (1 - h_train))

    ## evaluate testing error of the updated w 
    # we should measure classification error here 
    prob_test = sigmoid(X_test @ w)
    label_pre = (prob_test >= 0.5).astype(float)
    er = np.mean(label_pre != label_test)
    er_test.append(er)
# --- end of task --- #
    
print('final test classification error: %.4f' % er_test[-1])

plt.figure()    
plt.plot(er_test)
plt.xlabel('Iteration')
plt.ylabel('Classification Error')
plt.title('Figure 2: Logistic Regression')
plt.show()