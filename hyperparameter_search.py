from Learning import q_learning
from ADS_Environment import Environment
import numpy as np

env = Environment(is_deterministic=True)
max_weights = 15000
print("-------------------")
print("L(earning) Agent will learn now using Q-Learning in the Public Civility Game.")
print("-------------------")

vweights = [[1.0, 0.31, 0.013],
            [1.0, 0.372, 1.5],
            [1.0, 0.09, 0.000000001],
            [1.0, 0.045, 0.233],
            [1.0, 0.372, 1.5],
            [1.0, 0.005, 0.633]]

vvalues = [[7.0, 0.0, -3.375],
           [6.03125, 0.0, 0.0],
           [10.0, -20.0, - 4.0],
           [10.0, - 30.0, - 0.75],
           [6.03125, 0.0, 0.0],
           [9.625, - 30.0, 0.0]]
valpha = [[], [], [], [], [], []]

for i in range(0, 6):
    weights = vweights[i]
    value = vvalues[i]
    print("-----")
    print()
    print("-----")
    print("Policy with the weights: ", weights)
    print("Expected value is: ", value)
    for alpha in np.arange(0.7, 0.81, 0.01):
        policy, v, q = q_learning(env, weights, alpha=alpha, max_weights=max_weights, max_episodes=50000)
        print("-------------------")
        print("The Learnt Policy has the following Value for alpha = ", alpha, " is:")
        print(v[43, 45, 31])
        error = v[43, 45, 31] - value
        if sum(abs(error)) < 0.5:  # 0.5 ??
            print("This alpha works!")
            valpha[i].append(alpha)

print("-------------------")
print("Finnished!!!")
print(valpha)