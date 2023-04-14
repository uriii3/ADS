from Learning import q_learning
from ADS_Environment import Environment
import numpy as np

env = Environment(is_deterministic=True)
max_weights = 15000
print("-------------------")
print("L(earning) Agent will learn now using Q-Learning in the Public Civility Game.")
print("-------------------")

vweights = [[1.0, 0.255, 0.007],
             [1.0, 2.25, 7.6],
             [1.0, 0.052, 0.00001],
             [1.0, 0.001, 0.03],
             [1.0, 6.0, 20.0],
             [1.0, 0.001, 0.66]]

vvalues =[[7.0, 0.0, -8.1875],
            [5.765625, 0.0, 0.0],
            [10.0, -20.0, - 6.0],
            [10.0, - 30.0, - 1.125],
            [5.765625, 0.0, 0.0],
            [9.5625, - 30.0, 0.0]]


env.initial_pedestrian_1_position = env.translate_state_cell(31)
env.initial_pedestrian_2_position = env.translate_state_cell(38)
valpha = [[], [], [], [], [], []]

for i in range(0, 6):
    weights = vweights[i]
    value = vvalues[i]
    print("-----")
    print()
    print("-----")
    print("Policy with the weights: ", weights)
    print("Expected value is: ", value)
    for alpha in np.arange(0.1, 0.8, 0.1):
        policy, v, q = q_learning(env, weights, alpha=alpha, max_weights=max_weights, max_episodes=50000)
        print("-------------------")
        print("The Learnt Policy has the following Value for alpha = ", alpha, " is:")
        print(v[43, 38, 31])
        error = v[43, 38, 31] - value
        if sum(abs(error)) < 0.5:  # 0.5 ??
            print("This alpha works!")
            valpha[i].append(alpha)

print("-------------------")
print("Finnished!!!")
print(valpha)