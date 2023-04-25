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
    v_lexico = ['102', '120', '012', '021', '210', '201', 'unethical']
    v_good = [False, True, True, True, True,
              True, False]  # if some lexicographic order has already a good policy mark it here

    # Basics things to change manually:
    pedestrian1_position = 31
    pedestrian2_position = 38

    v_weights = [[1.0, 0.255, 0.007],
                 [1.0, 2.25, 7.6],
                 [1.0, 0.07, 0.00001],
                 [1.0, 0.001, 0.03],
                 [1.0, 6.0, 20.0],
                 [1.0, 0.001, 0.66],
                 [1.0, 0.0, 0.0]]

    v_values = [[7.0, 0.0, -8.1875],
                [5.765625, 0.0, 0.0],
                [10.0, -20.0, - 6.0],
                [10.0, - 30.0, - 1.125],
                [5.765625, 0.0, 0.0],
                [9.5625, - 30.0, 0.0],
                [10, -30, 0.0]] # no estic segur dels valors hehe

    # Preparations before loop
    env = Environment(is_deterministic=True)
    max_weights = 15000
    alpha = 0.7
    print(env.translate_state_cell(pedestrian1_position))
    env.initial_pedestrian_1_position = env.translate_state_cell(pedestrian1_position)
    env.initial_pedestrian_2_position = env.translate_state_cell(pedestrian2_position)

    for order, is_good, weights, exp_value in zip(v_lexico, v_good, v_weights, v_values):
        print('hola')
        if not is_good:
            print("Learning order ...", order)
            policy, v, q = q_learning(env, weights, alpha=alpha, max_weights=max_weights, max_episodes=60000)
            print("-------------------")

            print("The Learnt Policy has the following Value:")
            print(v[43, pedestrian2_position, pedestrian1_position])
            error = v[43, pedestrian2_position, pedestrian1_position] - exp_value
            if sum(abs(error)) < 0.5:
                print("This policy works!: ", order)
            # Save policy
            np.save("./Policies_" + str(pedestrian2_position) + "_" + str(pedestrian1_position) + "/policy_lex" + order + ".npy", policy)
            print("Saved!")


main()
