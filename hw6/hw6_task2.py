# Josiah Abraham, Jason Vo
# hw6_task2.py
# CS5013 HW 6 - Task 2

import random

random.seed(42)

gamma = 0.8
step_reward = -0.04
num_trials = 10000

# policy from figure 1
old_policy = {
    (1, 1): 'R', (1, 2): 'U', (1, 3): 'L',
    (2, 1): 'U', (2, 3): 'D',
    (3, 1): 'L', (3, 2): 'L', (3, 3): 'R',
    (4, 1): 'L'
}

# 0.6 target, 0.2 left, 0.1 right, 0.1 back
directions = {
    'U': ['U', 'L', 'R', 'D'],
    'D': ['D', 'R', 'L', 'U'],
    'L': ['L', 'D', 'U', 'R'],
    'R': ['R', 'U', 'D', 'L']
}
probs = [0.6, 0.2, 0.1, 0.1]

states = [(1, 1), (1, 2), (1, 3), (2, 1), (2, 3), (3, 1), (3, 2), (3, 3), (4, 1)]
terminal = {(4, 3): 1.0, (4, 2): -1.0}


def move(curr, m):
    x, y = curr
    if m == 'U': y += 1
    elif m == 'D': y -= 1
    elif m == 'L': x -= 1
    elif m == 'R': x += 1
    if x < 1 or x > 4 or y < 1 or y > 3 or (x, y) == (2, 2):
        return curr  # hit wall or rock: stay
    return (x, y)


def get_next_state(curr, action):
    r = random.random()
    opts = directions[action]
    if r < 0.6: m = opts[0]
    elif r < 0.8: m = opts[1]
    elif r < 0.9: m = opts[2]
    else: m = opts[3]
    return move(curr, m)


def run_mc(start, policy, trials):
    total = 0.0
    for _ in range(trials):
        curr, ret, t = start, 0.0, 0
        while curr not in terminal:
            ret += (gamma ** t) * step_reward
            curr = get_next_state(curr, policy[curr])
            t += 1
        ret += (gamma ** t) * terminal[curr]
        total += ret
    return total / trials


def transition_probs(s, a):
    """Return {s': P(s'|s,a)} (merging outcomes that land in the same state)."""
    out = {}
    for m, p in zip(directions[a], probs):
        s2 = move(s, m)
        out[s2] = out.get(s2, 0.0) + p
    return out


def q_value(s, a, U):
    # Bellman eq: R(s) + gamma * sum_s' P(s'|s,a) U(s')
    return step_reward + gamma * sum(p * U[s2] for s2, p in transition_probs(s, a).items())


# utilities under the old policy
U = dict(terminal)
for s in states:
    U[s] = run_mc(s, old_policy, num_trials)

print("Estimated utilities (old policy):")
for s in states:
    print(f"  U{s} = {U[s]:.4f}")

# policy improvement
new_policy = {}
for s in states:
    q = {a: q_value(s, a, U) for a in 'UDLR'}
    new_policy[s] = max(q, key=q.get)
    if s == (1, 2):
        print("\nBellman analysis for state (1,2):")
        for a in 'UDLR':
            print(f"  Q((1,2),{a}) = {q[a]:.4f}")

print("\nState\tOld\tNew")
for s in states:
    print(f"{s}\t{old_policy[s]}\t{new_policy[s]}")