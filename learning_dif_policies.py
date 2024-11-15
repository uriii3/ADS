"""
We have a convex hull (v_function.pickle, from 13/4/23 at 12:09) and will use it for different initial positions.
We will have the set of initial positions defined by the 2 positions that the pedestrians take:
p. ex: Policies_45_31 correspond to those done in the first part of the project and are supposed to work already.
We will add Policies_38_31 and maybe some other if the first one fits good in our methodology.
To do so, we will use 3 types of vectors, lexicographic order, final theoretical values and weights.
"""

from Learning import q_learning
from ADS_Environment import Environment
import numpy as np


def main():
    v_lexico = ['102', '120', '012', '021', '210', '201', 'unethical1']
    v_good = [False, False, False, False, False,
              False, False]  # if some lexicographic order has already a good policy mark it here

    v_weights = [[1.0, 0.50523247, 0.0013476],
                 [1.0, 1.37225, 5.53922],
                 [1.0, 0.09699965370370374, 1e-06],
                 [1.0, 0.02669135802468774, 0.09333333333332157],
                 [1.0, 0.6493272128519423, 2.9424395436216413],
                 [1.0, 0.0032263374485599463, 1.2381344307270263],
                 [1.0, 0.0, 0.0]]

    v_values = [[7.0, 0.0, -3.61688386],
                [5.40370368, 0.0, 0.0],
                [10.0, -20.0, -4.48148148],
                [10.0, -30.0, -1.40740741],
                [5.40370368,  0.0, 0.0],
                [9.26646091, -22.5, 0.0],
                [10, -30, 0.0]] # no estic segur dels valors hehe

    # Preparations before loop
    initial_pedestrian_1_cell = 31
    initial_pedestrian_2_cell = 45
    isMoreStochastic = True
    env = Environment(isMoreStochastic=isMoreStochastic,  initial_pedestrian_1_cell=initial_pedestrian_1_cell,
                      initial_pedestrian_2_cell=initial_pedestrian_2_cell)
    max_weights = 15000
    max_episodes = 200000
    alpha = 0.8

    for order, is_good, weights, exp_value in zip(v_lexico, v_good, v_weights, v_values):
        print('hola')
        if not is_good:
            print("Learning order ...", order)
            policy, v, q = q_learning(env, weights, alpha=alpha, max_weights=max_weights, max_episodes=max_episodes)
            print("-------------------")

            print("The Learnt Policy has the following Value:")
            print(v[43, initial_pedestrian_2_cell, initial_pedestrian_1_cell])
            error = v[43, initial_pedestrian_2_cell, initial_pedestrian_1_cell] - exp_value
            if sum(abs(error)) < 0.5:
                print("This policy works!: ", order)
                print(error)
            # Save policy
            np.save("./new_more_sto/policy_lex" + order + ".npy", policy)
            print("Saved!")


main()


''' For 45-31 non stochastic runs:
     v_good = [True, True, True, True, True, True, True]
    v_weights = [[1.0, 0.31, 0.06],
                 [1.0, 0.482, 2.97],
                 [1.0, 0.04, 0.015385],
                 [1.0, 0.004, 0.08],
                 [1.0, 1.45, 15.77],
                 [1.0, 0.001, 0.66],
                 [1.0, 0.0, 0.0]]
    v_values = [[7.0, 0.0, -3.375],
                [6.03125, 0.0, 0.0],
                [10.0, -20.0, -4.0],
                [10.0, -30.0, -0.75],
                [6.03125,  0.0, 0.0],
                [9.625, -30, 0.0],
                [10, -30, 0.0]]
---------------------------------------------------------------

For 38-31 non stochastic runs:         
    v_good = [True, True, True, True, True, True, True]  
    v_weights = [[1.0, 0.221, 0.0133333],
                 [1.0, 1.26, 7.9],
                 [1.0, 0.0267, 0.0074],
                 [1.0, 0.001, 0.0267],
                 [1.0, 0.86, 5.23],
                 [1.0, 0.001, 0.66],
                 [1.0, 0.0, 0.0]]
    v_values = [[7.0, 0.0, -5.6875],
                [5.765625, 0.0, 0.0],
                [10.0, -20.0, -6.0],
                [10.0, -30.0, -1.125],
                [5.765625,  0.0, 0.0],
                [9.5625, -30, 0.0],
                [10, -30, 0.0]] '''