# Josiah Abraham, Jason Vo
# CS 5013 - Fall 2026, HW2

# hill climbing is a local search method

import random #for initial w
import pandas as pd #parsing csv
import numpy as np #math function
import matplotlib.pyplot as plt


# 1) parse the csv and encode to 1s and 0s

df = pd.read_csv('hw2/CreditCard.csv') #now in pandas dataframe object to manipulate

# drop any rows with missing data (NaNs) to prevent math errors
df = df.dropna()

# encode text to 0/1
df['Gender'] = df['Gender'].map({'M': 1, 'F' : 0})
df['CarOwner'] = df['CarOwner'].map({'Y': 1, 'N' : 0})
df['PropertyOwner'] = df['PropertyOwner'].map({'Y': 1, 'N' : 0})

# we will use the attributes from gender...email
attributes_cols = ['Gender', 'CarOwner', 'PropertyOwner', '#Children', 'WorkPhone', 'Email_ID']

# GLOBAL X and y: will use in errorFunction

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

    predictions = np.dot(X, w) #f(x)

    # subtract y from this which is the application result / credit approve
    squared_errors = (predictions - y)**2
    error = np.mean(squared_errors)

    return error #this function will be called during hill climbing

# 3) hill climbing 

# requires an initial w, which will be random
initial_W = np.array([random.choice([-1, 1]) for _ in range(6)])

# get its initial er(w)

# for use when plotting graph, store each round's result in list
error_history = []

# when looking at adjacent neighbors, will have 6 neighbors, since there are 6 possible attributes to change one by one

# will be changed with each iteration
currentW = initial_W

while True:

    # will be compared against, updated by the end of the for loop
    currentErrorValue = errorFunction(currentW, X, y)
    error_history.append(currentErrorValue)

    # each round will have 6 neighbors, need to track best neighbor
    bestNeighbor = None
    bestNeighborErrorValue = float('inf') #temporary placeholders

    # check all 6 neighbors by flipping each bit in current W array
    for i in range(6):
        neighbor = currentW.copy()
        neighbor[i] *= -1

        # get error value of neighbor
        neighborErrorValue = errorFunction(neighbor, X, y)

        # compare neighbor and best neighbor, if neighbor less than becomes new bestNeighbor
        if neighborErrorValue < bestNeighborErrorValue:
            bestNeighbor = neighbor
            bestNeighborErrorValue = neighborErrorValue


    # after we get the best neighbor, we want to compare against our currentW
    if bestNeighborErrorValue < currentErrorValue:
        currentW = bestNeighbor

# if bestNeighbor > current, this means we have reached a local minimum so exit while loop 
    else:
        break

print(f"Optimal w: {currentW.tolist()}")
print(f"Final er(w): {currentErrorValue}")

# 4) generate figure 2 graph, er(w) versus search round

# x axis data: rounds of search
rounds = range(1, len(error_history) + 1)

# y axis is er(w) which is represented in error_history
# plot line graph
plt.plot(rounds, error_history, marker='o', linestyle='-', color='b')

# add the required titles and labels
plt.title('Figure 2: er(w) vs Round of Search') 
plt.xlabel('Round of Search') 
plt.ylabel('Error: er(w)') 

# add a grid for readability and display the graph
plt.grid(True)
plt.show()