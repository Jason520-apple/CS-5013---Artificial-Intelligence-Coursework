# Josiah Abraham, Jason Vo
# CS 5013 - Fall 2026, HW2

# hill climbing is a local search method

import random #for initial w
import pandas as pd #parsing csv
import numpy as np


# 1) parse the csv and encode to 1s and 0s

df = pd.read_csv('CreditCard.csv') #now in pandas dataframe object to manipulate

# encode text to 0/1
df['Gender'] = df['Gender'].map({'M': 1, 'F' : 0})
df['CarOwnder'] = df['CarOwnder'].map({'Y': 1, 'N' : 0})
df['PropertyOwner'] = df['PropertyOwner'].map({'Y': 1, 'N' : 0})

# we will use the attributes from gender...email, not including credit approve and ind id
print(df.head())

# our goal is to minimize the apporximation error
# we will stop once get lower er(w), and return the w that we currently have