# Josiah Abraham, Jason Vo
# CS 5013 - Fall 2026, HW2

#genetic algorithm

import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('CreditCard.csv') #adds data from csv file
df = df.dropna()  #drops rows with incomplete info


df['Gender'] = df['Gender'].map({'M': 1, 'F': 0}) #convert into bits
df['CarOwner'] = df['CarOwner'].map({'Y': 1, 'N': 0})
df['PropertyOwner'] = df['PropertyOwner'].map({'Y': 1, 'N': 0})

#categories
attribute_cols = ['Gender', 'CarOwner', 'PropertyOwner', '#Children', 'WorkPhone', 'Email_ID']
X = df[attribute_cols].values
y = df['CreditApprove'].values


def error_function(w): #gap between predicted and actual values
    #er(w) = mean((f(x_i) - y_i)^2)
    return np.mean((np.dot(X, w) - y) ** 2)


def fitness(w): #smaller error=larger fitness value
    #fitness is e^(-er(w))
    return np.exp(-error_function(w))



population_size = 6  #picked smaller value
number_generation = 30   
mutation_rate = 0.05   #probability of flipping each element of a child
crossover_point = 3    #splitting in the middle
elite_count = 1      #carrying over best chromosome to next generation
random_val = 7       #constant random value during each run      

random.seed(random_val)
np.random.seed(random_val)


def random_chromosome(): #chromosome with 6 values
    return np.array([random.choice([-1, 1]) for _ in range(6)])


def select_parents(population, fitnesses): #choose parents based on fitness
    probs = fitnesses / fitnesses.sum()
    idx = np.random.choice(len(population), size=2, p=probs)
    return population[idx[0]], population[idx[1]]


def crossover(p1, p2): #combine to create child
    return np.concatenate([p1[:crossover_point], p2[crossover_point:]])


def mutate(child): #mutate child's genes
    child = child.copy()
    for i in range(len(child)):
        if random.random() < mutation_rate:
            child[i] *= -1
    return child



population = [random_chromosome() for _ in range(population_size)]  #create first population of random chromosomes

best_w = None #tracks best one
best_error = float('inf')
error_history = []  #store best er(w) in each generation

for gen in range(number_generation): #repeat for total number of generations
    errors = np.array([error_function(w) for w in population])
    fitnesses = np.exp(-errors)

    #track best chromosome in this generation and overall
    gen_best_idx = int(np.argmin(errors))
    error_history.append(errors[gen_best_idx])
    if errors[gen_best_idx] < best_error:
        best_error = errors[gen_best_idx]
        best_w = population[gen_best_idx].copy()

    #build next generation
    elite_idx = np.argsort(errors)[:elite_count]
    next_population = [population[i].copy() for i in elite_idx]

    while len(next_population) < population_size:
        p1, p2 = select_parents(population, fitnesses)
        child = mutate(crossover(p1, p2))
        next_population.append(child)

    population = next_population

print(f"Optimal w: {best_w.tolist()}")
print(f"Final er(w): {best_error}")


#figure 3 (error vs generation plot)
generations = range(1, len(error_history) + 1)
plt.plot(generations, error_history, marker='o', linestyle='-', color='g')
plt.title('Figure 3: er(w) vs Generation')
plt.xlabel('Generation')
plt.ylabel('Error: er(w)')
plt.grid(True)
plt.show()