# Josiah Abraham, Jason Vo
# hw6_task1.py
# CS5013 HW 6 - Task 1

import random

random.seed(42)

state = (1, 1)
num_trials = 10000 #high number of experiments for better results
gamma = 0.8
step_reward = -0.04

# policy from figure 1 (left)
policy = {
    (1, 1): 'R',
    (1, 2): 'U',
    (1, 3): 'L',
    (2, 1): 'U',
    (2, 3): 'D',
    (3, 1): 'L',
    (3, 2): 'L',
    (3, 3): 'R',
    (4, 1): 'L'
}

# order of possible directions for each action: [0.6 target, 0.2 left, 0.1 right, 0.1 back]
directions = {
    'U': ['U', 'L', 'R', 'D'],
    'D': ['D', 'R', 'L', 'U'],
    'L': ['L', 'D', 'U', 'R'],
    'R': ['R', 'U', 'D', 'L']
}

def get_next_state(curr, action):
    # pick actual move based on stochastic probabilities (figure 1 right side)
    r = random.random()
    opts = directions[action]
    if r < 0.6: # targeted direction
        move = opts[0]
    elif r < 0.8: # relative left
        move = opts[1]
    elif r < 0.9: # relative right
        move = opts[2]
    else: # opposite direction
        move = opts[3]

    x, y = curr
    if move == 'U':
        y += 1
    elif move == 'D':
        y -= 1
    elif move == 'L':
        x -= 1
    elif move == 'R':
        x += 1

    # stay in current state if hitting a wall or the rock at (2, 2)
    if x < 1 or x > 4 or y < 1 or y > 3 or (x, y) == (2, 2):
        return curr
    return (x, y)

# run first-visit monte carlo
def run_mc(start_state, trials):
    total_utility = 0.0

    for _ in range(trials):
        curr = start_state
        episode_return = 0.0
        t = 0

        # keep moving until hitting a terminal state
        while curr != (4, 3) and curr != (4, 2):
            episode_return += (gamma ** t) * step_reward
            curr = get_next_state(curr, policy[curr])
            t += 1

        # add final terminal reward
        if curr == (4, 3):
            episode_return += (gamma ** t) * 1.0
        else:
            episode_return += (gamma ** t) * -1.0

        total_utility += episode_return

    return total_utility / trials


# run first-visit MC for all 9 non-terminal states
states = [
    (1, 1), (1, 2), (1, 3),
    (2, 1), (2, 3),
    (3, 1), (3, 2), (3, 3),
    (4, 1)
]

utilities = {}
for s in states:
    utilities[s] = run_mc(s, num_trials)

print(f"Utility for state {state}: {utilities[state]:.4f}")
print()
print("State\tUtility")
for s in states:
    print(f"{s}\t{utilities[s]:.4f}")