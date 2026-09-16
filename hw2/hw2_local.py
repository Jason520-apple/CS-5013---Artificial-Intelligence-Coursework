# Josiah Abraham, Jason Vo
# CS 5013 - Fall 2026, HW2

# hill climbing is a local search method

import random #for initial w
import pandas as pd #parsing csv
import numpy as np #math function


# 1) parse the csv and encode to 1s and 0s

df = pd.read_csv('CreditCard.csv') #now in pandas dataframe object to manipulate

# encode text to 0/1
df['Gender'] = df['Gender'].map({'M': 1, 'F' : 0})
df['CarOwnder'] = df['CarOwnder'].map({'Y': 1, 'N' : 0})
df['PropertyOwner'] = df['PropertyOwner'].map({'Y': 1, 'N' : 0})

# we will use the attributes from gender...email
attributes_cols = ['Gender', 'CarOwner', 'PropertyOwner', '#Children', 'WorkPhone', 'Email_ID']

# 2d array to store the nx6 w's 
X = df[attributes_cols].values

# store the credit approve values for use later
y = df['CreditApprove'].values


# our goal is to minimize the apporximation error
# we will stop once get lower er(w), and return the w that we currently have

# 2) write the er(w) function; each time the 6 neighbors of w are checked it will 
# be passed into this function, then the one with the lowest value will be moved to
# if the lowest neighbor is higher than current error then we will end loop there

def errorFunction(w, X, y):
    # f(x) is the dot product of the w vector (ex: w = [1, 1, −1, −1, 1, −1]
    # along with the x which is each attribute, then added together

    predictions = np.dot(w, X) #f(x)

    # subtract y from this which is the application result / credit approve
    squared_errors = (predictions - y)**2
    error = np.mean(squared_errors)

    return error #this function will be called during hill climbing

# 3) hill climbing 

# requires an initial w, which will be random
w = np.array([random.choice([-1, 1]) for _ in range(6)])

# for use when plotting graph, store each round in
error_history = []

# when looking at adjacent neighbors, will have 6 neighbors, since there are 6 possible attributes to change one by one


# then since we are trying to minimize er(w), will do if neighbor value >= current return current

# 4) generate figure 2 graph, er(w) versus search round