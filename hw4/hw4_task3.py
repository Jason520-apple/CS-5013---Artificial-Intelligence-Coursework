import numpy as np
import matplotlib.pyplot as plt

# --- Your Task --- #
# import necessary library 
# if you need more libraries, just import them
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
# --- end of task --- #

# load a data set for regression
# in array "data", each row represents a community 
# each column represents an attribute of community 
# last column is the continuous label of crime rate in the community
data = np.loadtxt('crimerate.csv', delimiter=',', skiprows=1)
[n,p] = np.shape(data)

# always use last 25% data for testing 
num_test = int(0.25*n)
sample_test = data[n-num_test:,0:-1]
label_test = data[n-num_test:,-1]

# --- Your Task --- #
# now, pick the percentage of data used for training 
# remember we should be able to observe overfitting with this pick 
# note: maximum percentage is 0.75 
per = 0.75 #kfold benefits from as much training data as possible
num_train = int(n*per)
sample_train = data[0:num_train,0:-1]
label_train = data[0:num_train,-1]
# --- end of task --- #


# --- Your Task --- #
# We will use a regression model called Ridge. 
# This model has a hyper-parameter alpha. Larger alpha means simpler model. 
# Pick 5 candidate values for alpha (in ascending order)
# Remember we should aim to observe both overfitting and underfitting from these values 
# Suggestion: the first value should be very small and the last should be large 
alpha_vec = [0.0001, 0.4, 16, 1000, 100000]
# --- end of task --- #

er_train_alpha = []
er_test_alpha = []

er_valid_alpha = []
for alpha in alpha_vec: 

    # pick ridge model, set its hyperparameter 
    model = Ridge(alpha = alpha)
    
    # --- Your Task --- #
    # now implement k-fold cross validation 
    # on the training set (which means splitting 
    # training set into k-folds) to get the 
    # validation error for each candidate alpha value 
    # store it in "er_valid"
    
    k = 5 # number of folds
    fold_size = num_train // k # rows per fold
    fold_errors = []

    for i in range(k):
        # create validation chunk
        start = i * fold_size
        end = (i + 1) * fold_size if i != k - 1 else num_train

        # Isolate the validation chunk
        val_X = sample_train[start:end]
        val_Y = label_train[start:end]

        # Stitch together the remaining data for training
        train_X = np.concatenate((sample_train[:start], sample_train[end:]), axis=0)
        train_y = np.concatenate((label_train[:start], label_train[end:]), axis=0)
        
        # Train and evaluate the fold
        model.fit(train_X, train_y)
        fold_errors.append(mean_squared_error(val_Y, model.predict(val_X)))

    # Average the 5 fold errors to get the final validation error for this alpha
    er_valid = np.mean(fold_errors)
    er_valid_alpha.append(er_valid)

    
    
    # --- end of task --- #

# output
print("Table 1 Values:")
for alpha, error in zip(alpha_vec, er_valid_alpha):
    print(f"Hyper-Parameter: {alpha} | Validation Error: {error}")

# Now you should have obtained a validation error for each alpha value 
# In the homework, you just need to report these values

# The following practice is only for your own learning purpose.
# Compare the candidate values and pick the alpha that gives the smallest error 
# set it to "alpha_opt"
alpha_opt = alpha_vec[np.argmin(er_valid_alpha)]

# now retrain your model on the entire training set using alpha_opt 
# then evaluate your model on the testing set 
model = Ridge(alpha = alpha_opt)
model.fit(sample_train, label_train)

er_train = mean_squared_error(label_train, model.predict(sample_train))
er_test = mean_squared_error(label_test, model.predict(sample_test))